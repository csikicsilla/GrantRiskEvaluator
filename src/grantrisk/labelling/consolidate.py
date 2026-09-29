"""Stage L2: consolidate (Spec_L2_Consolidate.md, DEC-18).

For each document and factor, choose one value from the manual, regex and LLM
observations: the expert's points first, then the preferred automated source, then
the other one. Values are copied unchanged; nothing is computed here.
"""

from __future__ import annotations

import collections
import json
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from grantrisk.labelling import scoring
from grantrisk.labelling.scoring import FACTORS
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

AUTOMATED = ("regex", "llm")
GOLD_ONLY = "gold_only"  # DEC-35: the document set is the documents of the manual run


class L2InputError(ValueError):
    """The inputs or the configuration violate the L2 contract; nothing is written."""


@dataclass(frozen=True)
class Observation:
    source: str
    status: str
    value: Any = None
    points: int | None = None
    evidence: str | None = None
    evidence_page: int | None = None

    def found(self, factor: str) -> bool:
        # DEC-33: an empty activity list counts as not found.
        return self.status == "found" and not (factor == "tam_tevekenyseg" and self.value == [])


@dataclass(frozen=True)
class Consolidated:
    doc_id: str
    factor: str
    value: Any
    points: int | None
    chosen_source: str | None
    rule: str  # manual, preferred, fallback or none
    evidence: str | None
    evidence_page: int | None


Observations = Mapping[tuple[str, str], Observation]  # (doc_id, factor) → observation


def check_preferences(preferences: Mapping[str, str]) -> None:
    """SPEC-L2-03: every factor needs a preferred source, llm or regex."""
    missing = [f for f in FACTORS if f not in preferences]
    wrong = {f: p for f, p in preferences.items() if p not in AUTOMATED}
    if missing or wrong:
        raise L2InputError(f"preferred_source: missing for {missing}; not llm or regex: {wrong}")


def consolidate(
    doc_ids: Iterable[str],
    manual: Observations,
    automated: Mapping[str, Observations],
    preferences: Mapping[str, str],
) -> list[Consolidated]:
    """SPEC-L2-01, -02, -04: exactly one row per document and factor."""
    rows = []
    for doc_id in sorted(set(doc_ids)):
        for f in FACTORS:
            m = manual.get((doc_id, f))
            if m is not None and m.found(f):
                rows.append(Consolidated(doc_id, f, None, m.points, m.source, "manual", None, None))
                continue
            preferred = preferences.get(f)
            order = [(preferred, "preferred")] + [(s, "fallback") for s in AUTOMATED if s != preferred]
            for source, rule in order:
                o = automated.get(source, {}).get((doc_id, f))
                if o is not None and o.found(f):
                    rows.append(Consolidated(doc_id, f, o.value, None, o.source, rule, o.evidence, o.evidence_page))
                    break
            else:
                rows.append(Consolidated(doc_id, f, None, None, None, "none", None, None))
    return rows


def disagreements(doc_ids: Iterable[str], automated: Mapping[str, Observations]) -> dict[str, dict[str, int]]:
    """SPEC-L2-05: per factor, documents where regex and LLM both found a value, and how many differ in points."""
    result = {}
    regex, llm = automated.get("regex", {}), automated.get("llm", {})
    for f in FACTORS:
        both = differ = 0
        for doc_id in doc_ids:
            r, l = regex.get((doc_id, f)), llm.get((doc_id, f))
            if r is not None and l is not None and r.found(f) and l.found(f):
                both += 1
                differ += scoring.points(f, r.value)[0] != scoring.points(f, l.value)[0]
        result[f] = {"both_found": both, "differ": differ}
    return result


# --- Database run (SPEC-L2-06) -------------------------------------------------------


def load_observations(conn: sqlite3.Connection, run_id: str) -> dict[tuple[str, str], Observation]:
    rows = conn.execute(
        "SELECT doc_id, factor, source, status, value_json, points, evidence, evidence_page"
        " FROM factor_observations WHERE run_id = ?",
        (run_id,),
    )
    return {
        (r["doc_id"], r["factor"]): Observation(
            source=r["source"],
            status=r["status"],
            value=None if r["value_json"] is None else json.loads(r["value_json"], parse_float=Decimal),
            points=r["points"],
            evidence=r["evidence"],
            evidence_page=r["evidence_page"],
        )
        for r in rows
    }


def _sources(conn: sqlite3.Connection, run_id: str) -> set[str]:
    return {r[0] for r in conn.execute("SELECT DISTINCT source FROM factor_observations WHERE run_id = ?", (run_id,))}


def _check_source(conn: sqlite3.Connection, run_id: str, expected: str) -> None:
    runs.require_complete(conn, run_id, "L1")
    sources = _sources(conn, run_id)
    ok = sources == {expected} if expected != "llm" else len(sources) == 1 and next(iter(sources)).startswith("llm:")
    if not ok:
        raise L2InputError(f"run {run_id} holds sources {sorted(sources)}, not {expected}")


def _number(x: Any) -> int | float:
    if isinstance(x, Decimal):
        # float(repr) round-trips the shortest decimal form, so "30.01" stays 30.01.
        return int(x) if x == x.to_integral_value() else float(x)
    raise TypeError(f"cannot store {type(x).__name__} as JSON")


def _json_value(value: Any) -> str | None:
    return None if value is None else json.dumps(value, default=_number, ensure_ascii=False)


def run(
    conn: sqlite3.Connection,
    config_values: Mapping[str, Any],
    data_root: Path,
    *,
    manual_run_id: str,
    regex_run_id: str | None = None,
    llm_run_id: str | None = None,
    c2_run_id: str | None = None,
    gold_only: bool = False,
) -> str:
    """Consolidate the observations of the named L1 runs. Returns the new run id.

    The document set is the C2 run's documents with text (SPEC-L2-01), or, in the
    gold-only mode of DEC-35, the documents of the manual run.
    """
    if gold_only == (c2_run_id is not None):
        raise L2InputError("name either a C2 run or the gold-only mode (DEC-35)")
    if c2_run_id is not None:
        raise L2InputError("C2 is not implemented yet; until then only the gold-only mode is available")
    _check_source(conn, manual_run_id, "manual")
    automated_runs = {s: r for s, r in (("regex", regex_run_id), ("llm", llm_run_id)) if r}
    for source, run_id in automated_runs.items():
        _check_source(conn, run_id, source)
    settings = config_values.get("consolidate", {})
    preferences = settings.get("preferred_source") or {}
    if automated_runs:
        check_preferences(preferences)

    inputs = [manual_run_id, *automated_runs.values()]
    run_id = runs.start(conn, "L2", config_values, inputs=inputs)
    report_path = None
    try:
        manual = load_observations(conn, manual_run_id)
        automated = {s: load_observations(conn, r) for s, r in automated_runs.items()}
        doc_ids = sorted({d for d, _ in manual})
        rows = consolidate(doc_ids, manual, automated, preferences)
        by_rule = {f: collections.Counter() for f in FACTORS}
        by_source = {f: collections.Counter() for f in FACTORS}
        for r in rows:
            by_rule[r.factor][r.rule] += 1
            by_source[r.factor][r.chosen_source or "none"] += 1
        report = {
            "input_runs": {"manual": manual_run_id, **automated_runs},
            "document_set": {"mode": GOLD_ONLY, "from_run": manual_run_id, "documents": len(doc_ids)},
            "policy": "DEC-18: manual points, then the preferred source, then the other source",
            "preferred_source": dict(preferences) if automated_runs else None,
            "preferences_from_e1_run": settings.get("preferences_from_e1_run") if automated_runs else None,
            "by_factor": {f: {"rule": dict(sorted(by_rule[f].items())), "source": dict(sorted(by_source[f].items()))}
                          for f in FACTORS},
            "disagreement_regex_llm": disagreements(doc_ids, automated),
            "documents_missing_from": {
                s: sorted({d for d in doc_ids} - {d for d, _ in obs}) for s, obs in automated.items()
            },
        }
        report_path = files.write_json(data_root, f"reports/{run_id}/l2_report.json", report)
        with transaction(conn):
            conn.executemany(
                "INSERT INTO consolidated_factors (run_id, doc_id, factor, value_json, points, chosen_source,"
                " rule, evidence, evidence_page) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (run_id, r.doc_id, r.factor, _json_value(r.value), r.points, r.chosen_source,
                     r.rule, r.evidence, r.evidence_page)
                    for r in rows
                ],
            )
            runs.complete(conn, run_id, report_path)
    except Exception as exc:
        if report_path:
            files.remove(data_root, report_path)
        runs.fail(conn, run_id, f"{type(exc).__name__}: {exc}")
        raise
    return run_id
