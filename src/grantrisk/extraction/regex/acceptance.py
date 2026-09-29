"""Acceptance of the regex rules against the gold set (SPEC-L1-06, DEC-28).

A factor's rule is accepted when its agreement with the expert's points on the gold
documents is at least that of the old regex. Agreement is counted at the level of
points, after L3's points function and the TOP rule (SPEC-L3-14): a value that was
not found counts against the extractor. The baseline is the old code line's regex
points (column ``regex`` of Gold_second_test_results_final.csv), compared with the
current gold file. Limitation: the rules were developed on the same 42 documents, so
the agreement is in-sample (ISS-26).
"""

from __future__ import annotations

import csv
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grantrisk.extraction.manual.gold import read_gold
from grantrisk.extraction.regex.extractor import extract
from grantrisk.labelling import scoring
from grantrisk.labelling.scoring import FACTORS


@dataclass
class GoldDocument:
    name: str  # the call as written in the current gold file
    doc_id: str
    programme: str
    gold: dict[str, int | None]
    old: dict[str, int | None]  # the old regex points; None where the old code gave none
    markdown: str | None = None  # None while the document has no text


def read_old_baseline(path: Path, renames: Mapping[str, str] | None = None) -> dict[str, dict[str, int | None]]:
    """The old regex points per call and factor; calls renamed since then get their current name."""
    renames = renames or {}
    result: dict[str, dict[str, int | None]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = renames.get(row["felhivas"].strip(), row["felhivas"].strip())
            cell = (row["regex"] or "").strip()
            result.setdefault(name, {})[row["factor"]] = int(float(cell)) if cell else None
    return result


def points_of(observations: Mapping[str, Any], programme: str | None) -> dict[str, int | None]:
    """The points of each factor from the extracted values, with the TOP rule, without imputation."""
    result = {}
    for f in FACTORS:
        o = observations[f]
        value = o.value if o.status == "found" else None
        result[f] = scoring.points(f, value, programme)[0]
    return result


def agreement(pairs: Sequence[tuple[int | None, int | None]]) -> dict[str, int]:
    """Counts of (gold points, extracted points) pairs, as E1 counts them (SPEC-E1-01)."""
    counts = {"agree": 0, "disagree": 0, "not_found": 0, "not_comparable": 0}
    for gold, extracted in pairs:
        if gold is None:
            counts["not_comparable"] += 1
        elif extracted is None:
            counts["not_found"] += 1
        elif extracted == gold:
            counts["agree"] += 1
        else:
            counts["disagree"] += 1
    return counts


def compare(docs: Sequence[GoldDocument]) -> dict[str, Any]:
    """Per-factor agreement of the new rules and the old baseline on the documents that have a text."""
    with_text = [d for d in docs if d.markdown is not None]
    new_points = {d.doc_id: points_of(extract(d.markdown), d.programme) for d in with_text}
    factors = {}
    for f in FACTORS:
        new = agreement([(d.gold[f], new_points[d.doc_id][f]) for d in with_text])
        old = agreement([(d.gold[f], d.old.get(f)) for d in with_text])
        factors[f] = {"new": new, "old": old, "accepted": new["agree"] >= old["agree"]}
    return {
        "documents": len(docs),
        "documents_with_text": len(with_text),
        "missing_texts": sorted(d.name for d in docs if d.markdown is None),
        "factors": factors,
        "total": {
            "new_agree": sum(v["new"]["agree"] for v in factors.values()),
            "old_agree": sum(v["old"]["agree"] for v in factors.values()),
            "cells": sum(len(with_text) - v["new"]["not_comparable"] for v in factors.values()),
        },
        "details": {
            d.name: {f: {"gold": d.gold[f], "new": new_points[d.doc_id][f], "old": d.old.get(f)} for f in FACTORS}
            for d in with_text
        },
        "limitation": "in-sample: the rules were developed on these gold documents (ISS-26)",
    }


def gold_documents(
    conn: sqlite3.Connection,
    *,
    c1_run_id: str,
    c2_run_id: str | None,
    gold_csv: Path,
    gold_pins: Mapping[str, str],
    old_baseline: Path,
    renames: Mapping[str, str] | None = None,
) -> list[GoldDocument]:
    """The gold documents with their expert points, old regex points, programme and (if any) text.

    Works on a connection opened read-only; it never writes.
    """
    old = read_old_baseline(old_baseline, renames)
    programmes = dict(conn.execute("SELECT doc_id, programme FROM documents WHERE run_id = ?", (c1_run_id,)).fetchall())
    texts = {}
    if c2_run_id:
        texts = dict(conn.execute(
            "SELECT doc_id, markdown FROM document_texts WHERE run_id = ? AND status = 'ok'", (c2_run_id,)
        ).fetchall())
    docs = []
    for row in read_gold(gold_csv):
        doc_id = gold_pins[row.name][:16]
        docs.append(GoldDocument(
            name=row.name, doc_id=doc_id, programme=programmes.get(doc_id), gold=dict(row.points),
            old=old.get(row.name, {}), markdown=texts.get(doc_id),
        ))
    return docs


def open_read_only(db_path: Path) -> sqlite3.Connection:
    """The database opened read-only: the acceptance check never migrates or writes it."""
    return sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)


def complete_c2_with(conn: sqlite3.Connection, doc_ids: Sequence[str]) -> str | None:
    """The newest complete C2 run in which every one of ``doc_ids`` has a text."""
    candidates = conn.execute(
        "SELECT run_id FROM runs WHERE stage = 'C2' AND status = 'complete' ORDER BY started_at DESC"
    ).fetchall()
    for (run_id,) in candidates:
        have = {r[0] for r in conn.execute(
            "SELECT doc_id FROM document_texts WHERE run_id = ? AND status = 'ok'", (run_id,))}
        if set(doc_ids) <= have:
            return run_id
    return None


def c1_of(conn: sqlite3.Connection, c2_run_id: str) -> str:
    """The C1 run a C2 run converted (its lineage, ARC-02)."""
    row = conn.execute(
        "SELECT i.input_run_id FROM run_inputs i JOIN runs r ON r.run_id = i.input_run_id"
        " WHERE i.run_id = ? AND r.stage = 'C1'", (c2_run_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"run {c2_run_id} has no C1 input")
    return row[0]


def format_table(report: Mapping[str, Any]) -> str:
    """The per-factor comparison as a Markdown table."""
    n = report["documents_with_text"]
    lines = [
        f"Gold documents with text: {n} of {report['documents']} (in-sample, ISS-26)",
        "",
        "| factor | new agree | new disagree | new not found | old agree | old disagree | old not found | accepted |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f, v in report["factors"].items():
        new, old = v["new"], v["old"]
        lines.append(
            f"| {f} | {new['agree']} | {new['disagree']} | {new['not_found']} | {old['agree']} | {old['disagree']}"
            f" | {old['not_found']} | {'yes' if v['accepted'] else 'NO'} |"
        )
    t = report["total"]
    lines.append(f"| **total** | {t['new_agree']} / {t['cells']} | | | {t['old_agree']} / {t['cells']} | | | |")
    return "\n".join(lines)
