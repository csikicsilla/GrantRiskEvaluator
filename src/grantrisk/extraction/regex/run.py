"""The L1 regex extractor as a stage run (Spec_L1_ExtractFactors.md §3.3).

Reads the Markdown of one complete C2 run, applies the rules of every factor to each
document with text, and stores ten observations per document with source ``regex``.
The rules are deterministic and local, so a run is written in one transaction.
"""

from __future__ import annotations

import collections
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from grantrisk.extraction.regex.extractor import Observation, extract
from grantrisk.extraction.regex.rules import RULES_VERSION
from grantrisk.labelling.scoring import FACTORS
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

SOURCE = "regex"


def _documents(
    conn: sqlite3.Connection, c2_run_id: str, doc_ids: Sequence[str] | None
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """(doc_id, Markdown) of the documents with text, and the skipped ones (SPEC-L1-03, -14)."""
    rows = conn.execute(
        "SELECT doc_id, status, error, markdown FROM document_texts WHERE run_id = ? ORDER BY doc_id", (c2_run_id,)
    ).fetchall()
    if doc_ids is not None:
        wanted = set(doc_ids)
        rows = [r for r in rows if r["doc_id"] in wanted]
    texts = [(r["doc_id"], r["markdown"]) for r in rows if r["status"] == "ok"]
    skipped = [{"doc_id": r["doc_id"], "reason": f"C2 status {r['status']}: {r['error']}"} for r in rows if r["status"] != "ok"]
    if doc_ids is not None:
        present = {r["doc_id"] for r in rows}
        skipped += [{"doc_id": d, "reason": "not in the C2 run"} for d in sorted(set(doc_ids) - present)]
    return texts, skipped


def _insert(conn: sqlite3.Connection, run_id: str, doc_id: str, observations: Mapping[str, Observation]) -> None:
    for f, o in observations.items():
        conn.execute(
            "INSERT INTO factor_observations (run_id, doc_id, factor, source, value_json, evidence, evidence_page,"
            " status, warnings_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, doc_id, f, SOURCE, None if o.value is None else json.dumps(o.value, ensure_ascii=False),
                o.evidence or None, o.evidence_page, o.status, json.dumps(o.warnings, ensure_ascii=False),
            ),
        )


def _report(c2_run_id: str, results: Mapping[str, Mapping[str, Observation]], skipped: list[dict[str, Any]]) -> dict[str, Any]:
    status = {f: collections.Counter() for f in FACTORS}
    warnings = {f: collections.Counter() for f in FACTORS}
    for observations in results.values():
        for f, o in observations.items():
            status[f][o.status] += 1
            warnings[f].update(w.split(":")[0] for w in o.warnings)
    found = sum(c["found"] for c in status.values())
    not_in_text = sum(c["evidence_not_in_text"] for c in warnings.values())
    return {
        "input_runs": {"C2": c2_run_id},
        "rules_version": RULES_VERSION,
        "documents": len(results),
        "skipped": skipped,
        "status_by_factor": {f: dict(sorted(c.items())) for f, c in status.items()},
        "warnings_by_factor": {f: dict(sorted(c.items())) for f, c in warnings.items() if c},
        "evidence_not_in_text_rate": round(not_in_text / found, 4) if found else None,  # SPEC-L1-02
    }


def run(
    conn: sqlite3.Connection,
    config_values: Mapping[str, Any],
    data_root: Path,
    *,
    c2_run_id: str,
    doc_ids: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Extract the factors of a C2 run's documents with the regex rules. Returns the run id."""
    runs.require_complete(conn, c2_run_id, "C2")
    snapshot = {**config_values, "regex_run": {"rules_version": RULES_VERSION, "doc_filter": doc_ids is not None}}
    run_id = runs.start(conn, "L1", snapshot, inputs=[c2_run_id])
    report_path = None
    try:
        texts, skipped = _documents(conn, c2_run_id, doc_ids)
        results = {}
        for i, (doc_id, markdown) in enumerate(texts, start=1):
            results[doc_id] = extract(markdown)
            if progress and (i % 50 == 0 or i == len(texts)):
                progress(f"[{i}/{len(texts)}] regex")
        report_path = files.write_json(data_root, f"reports/{run_id}/l1_regex_report.json", _report(c2_run_id, results, skipped))
        with transaction(conn):
            for doc_id, observations in results.items():
                _insert(conn, run_id, doc_id, observations)
            runs.complete(conn, run_id, report_path)
    except BaseException as exc:
        if report_path:
            files.remove(data_root, report_path)
        runs.fail(conn, run_id, f"{type(exc).__name__}: {exc}")
        raise
    return run_id
