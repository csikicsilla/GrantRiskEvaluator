"""Stage L3: score and label (Spec_L3_ScoreAndLabel.md).

``compute`` is the pure computation, tested against Appendix A of the chapter.
``run`` reads one L2 run and the C1 documents from the database, and writes the
risk labels and the run report in one transaction.
"""

from __future__ import annotations

import collections
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from grantrisk.labelling import scoring
from grantrisk.labelling.scoring import FACTORS, HIGH, LOW, MEDIUM
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

DETERMINED = ("band", "manual")


class L3InputError(ValueError):
    """The input violates the L3 contract; the run writes nothing (SPEC-L3-03)."""


@dataclass(frozen=True)
class FactorInput:
    """One consolidated factor: a value, or the expert's points (never both)."""

    value: Any = None
    points: int | None = None


@dataclass(frozen=True)
class DocumentInput:
    doc_id: str
    programme: str | None
    factors: Mapping[str, FactorInput]


@dataclass(frozen=True)
class FactorResult:
    points: Fraction
    origin: str  # band, manual, mean or top_rule


@dataclass(frozen=True)
class DocumentResult:
    doc_id: str
    factors: dict[str, FactorResult]
    n_determined: int
    low_coverage: bool
    total: Fraction
    normalised: Fraction
    fixed_label: str
    tercile_label: str


@dataclass(frozen=True)
class L3Result:
    documents: list[DocumentResult]
    report: dict[str, Any] = field(default_factory=dict)


def _exact(x: Fraction | None) -> str | None:
    return None if x is None else str(x)


def validate(docs: Sequence[DocumentInput]) -> None:
    """SPEC-L3-03: raise L3InputError listing every violation."""
    problems = []
    seen = set()
    for d in docs:
        if d.doc_id in seen:
            problems.append(f"{d.doc_id}: appears twice")
        seen.add(d.doc_id)
        missing = [f for f in FACTORS if f not in d.factors]
        unknown = [f for f in d.factors if f not in FACTORS]
        if missing:
            problems.append(f"{d.doc_id}: missing factor rows {missing}")
        if unknown:
            problems.append(f"{d.doc_id}: unknown factors {unknown}")
        for f in FACTORS:
            fi = d.factors.get(f)
            if fi is None:
                continue
            if fi.points is not None and fi.value is not None:
                problems.append(f"{d.doc_id} / {f}: carries both a value and points")
            elif fi.points is not None:
                if isinstance(fi.points, bool) or fi.points not in scoring.ALLOWED_POINTS[f]:
                    problems.append(f"{d.doc_id} / {f}: manual points {fi.points!r} are not allowed")
            elif fi.value is not None:
                error = scoring.domain_error(f, fi.value)
                if error:
                    problems.append(f"{d.doc_id} / {f}: {error}")
    if problems:
        raise L3InputError("invalid L3 input:\n  " + "\n  ".join(problems))


def tercile_split(scores: Sequence[tuple[Fraction, str]]) -> tuple[dict[str, str], dict[str, Any]]:
    """SPEC-L3-10: the tercile label of each document, and a description of the cuts.

    ``scores`` holds (normalised score, doc_id) pairs. Equal scores always get the
    same label; each cut goes to the admissible position nearest to its target, and
    on a tie in distance to the smaller one (DEC-15, DEC-32).
    """
    ordered = sorted(scores)
    n = len(ordered)
    admissible = [c for c in range(n + 1) if c in (0, n) or ordered[c - 1][0] != ordered[c][0]]

    def nearest(target: Fraction, lowest: int = 0) -> int:
        return min((c for c in admissible if c >= lowest), key=lambda c: (abs(c - target), c))

    def describe(target: Fraction, chosen: int) -> dict[str, Any]:
        tied = None
        if target not in admissible:
            below = max(c for c in admissible if c < target)
            above = min(c for c in admissible if c > target)
            tied = {
                "score": str(ordered[below][0]),
                "size": above - below,
                "went_to": "higher class" if chosen == below else "lower class",
            }
        return {"target": float(target), "position": chosen, "tied_group": tied}

    t1, t2 = Fraction(n, 3), Fraction(2 * n, 3)
    c1 = nearest(t1)
    c2 = nearest(t2, lowest=c1)
    labels = {}
    for position, (_, doc_id) in enumerate(ordered, start=1):
        labels[doc_id] = LOW if position <= c1 else MEDIUM if position <= c2 else HIGH
    info = {
        "n": n,
        "cut_1": describe(t1, c1),
        "cut_2": describe(t2, c2),
        "class_sizes": {LOW: c1, MEDIUM: c2 - c1, HIGH: n - c2},
        # E1 labels the gold documents with these scores (SPEC-E1-04).
        "last_low_score": str(ordered[c1 - 1][0]) if c1 > 0 else None,
        "last_medium_score": str(ordered[c2 - 1][0]) if c2 > 0 else None,
    }
    return labels, info


def compute(docs: Sequence[DocumentInput], low_coverage_threshold: int = 5) -> L3Result:
    """SPEC-L3-01 … -11: points, means, imputation, totals and both labels."""
    validate(docs)
    if not docs:
        raise L3InputError("the L2 run has no documents")
    docs = sorted(docs, key=lambda d: d.doc_id)

    # Points from the expert, the bands and the TOP rule; None where nothing applies.
    found: dict[str, dict[str, FactorResult | None]] = {}
    for d in docs:
        row: dict[str, FactorResult | None] = {}
        for f in FACTORS:
            fi = d.factors[f]
            if fi.points is not None:
                row[f] = FactorResult(Fraction(fi.points), "manual")
            else:
                p, origin = scoring.points(f, fi.value, d.programme)
                row[f] = None if p is None else FactorResult(Fraction(p), origin)
        found[d.doc_id] = row

    # SPEC-L3-05: means over the determined documents only; the TOP rule does not count.
    means: dict[str, Fraction] = {}
    for f in FACTORS:
        determined = [r[f].points for r in found.values() if r[f] is not None and r[f].origin in DETERMINED]
        if not determined:
            raise L3InputError(f"factor {f} is determined in no document")
        means[f] = sum(determined, Fraction(0)) / len(determined)

    # SPEC-L3-06 … -09: imputation, coverage, totals, fixed label.
    partial = []
    for d in docs:
        factors = {f: found[d.doc_id][f] or FactorResult(means[f], "mean") for f in FACTORS}
        n_determined = sum(r.origin in DETERMINED for r in factors.values())
        total = sum((r.points for r in factors.values()), Fraction(0))
        norm = scoring.normalised(total)
        _, fixed = scoring.fixed_label(norm)
        partial.append((d.doc_id, factors, n_determined, total, norm, fixed))

    # SPEC-L3-10: tercile label.
    terciles, cuts = tercile_split([(p[4], p[0]) for p in partial])
    results = [
        DocumentResult(
            doc_id=doc_id,
            factors=factors,
            n_determined=n_det,
            low_coverage=n_det < low_coverage_threshold,
            total=total,
            normalised=norm,
            fixed_label=fixed,
            tercile_label=terciles[doc_id],
        )
        for doc_id, factors, n_det, total, norm, fixed in partial
    ]
    return L3Result(results, _report(results, means, cuts, low_coverage_threshold))


def _report(
    results: list[DocumentResult],
    means: dict[str, Fraction],
    cuts: dict[str, Any],
    low_coverage_threshold: int,
) -> dict[str, Any]:
    n = len(results)
    by_factor = {
        f: dict(sorted(collections.Counter(r.factors[f].origin for r in results).items())) for f in FACTORS
    }
    warnings = []
    if n < 3:
        warnings.append("fewer than 3 documents: terciles are not meaningful")
    if len({r.normalised for r in results}) == 1:
        warnings.append("all documents have the same score")
    for label, size in cuts["class_sizes"].items():
        if size == 0:
            warnings.append(f"the tercile class '{label}' is empty")
    return {
        "scoring_rules_version": scoring.RULES_VERSION,
        "n_documents": n,
        "low_coverage_threshold": low_coverage_threshold,
        "factor_means": {
            f: {
                "mean": float(means[f]),
                "exact": str(means[f]),
                "n_determined": sum(r.factors[f].origin in DETERMINED for r in results),
            }
            for f in FACTORS
        },
        "coverage": {
            "by_factor": by_factor,
            "documents_by_n_determined": dict(sorted(collections.Counter(r.n_determined for r in results).items())),
            "low_coverage_documents": sum(r.low_coverage for r in results),
        },
        "tercile_cuts": cuts,
        "fixed_label_sizes": {label: sum(r.fixed_label == label for r in results) for label in scoring.LABELS},
        "warnings": warnings,
    }


# --- Database run (SPEC-L3-12) -------------------------------------------------------


def load_inputs(conn: sqlite3.Connection, l2_run_id: str, c1_run_id: str) -> list[DocumentInput]:
    """The consolidated factors of the L2 run, with each document's programme from the C1 run."""
    programmes = {
        r["doc_id"]: r["programme"]
        for r in conn.execute("SELECT doc_id, programme FROM documents WHERE run_id = ?", (c1_run_id,))
    }
    factors: dict[str, dict[str, FactorInput]] = collections.defaultdict(dict)
    for r in conn.execute(
        "SELECT doc_id, factor, value_json, points FROM consolidated_factors WHERE run_id = ?", (l2_run_id,)
    ):
        value = None if r["value_json"] is None else json.loads(r["value_json"], parse_float=Decimal)
        factors[r["doc_id"]][r["factor"]] = FactorInput(value=value, points=r["points"])
    without_document = sorted(d for d in factors if d not in programmes)
    if without_document:
        raise L3InputError(f"invalid L3 input: no Document record in run {c1_run_id} for {without_document}")
    return [DocumentInput(d, programmes[d], f) for d, f in factors.items()]


def _insert_document(conn: sqlite3.Connection, run_id: str, r: DocumentResult) -> None:
    conn.execute(
        "INSERT INTO risk_labels VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id, r.doc_id, r.n_determined, int(r.low_coverage),
            float(r.total), str(r.total), float(r.normalised), str(r.normalised),
            r.fixed_label, r.tercile_label,
        ),
    )
    conn.executemany(
        "INSERT INTO risk_label_factors VALUES (?, ?, ?, ?, ?, ?)",
        [(run_id, r.doc_id, f, float(fr.points), str(fr.points), fr.origin) for f, fr in r.factors.items()],
    )


def run(
    conn: sqlite3.Connection,
    config_values: Mapping[str, Any],
    data_root: Path,
    *,
    l2_run_id: str,
    c1_run_id: str,
) -> str:
    """Label the documents of an L2 run. Returns the new run id.

    On any error the run is marked failed, and neither rows nor a report remain.
    """
    runs.require_complete(conn, l2_run_id, "L2")
    runs.require_complete(conn, c1_run_id, "C1")
    threshold = config_values.get("label", {}).get("low_coverage_threshold", 5)
    run_id = runs.start(conn, "L3", config_values, inputs=[l2_run_id, c1_run_id])
    report_path = None
    try:
        result = compute(load_inputs(conn, l2_run_id, c1_run_id), threshold)
        report = {"input_runs": {"L2": l2_run_id, "C1": c1_run_id}, **result.report}
        report_path = files.write_json(data_root, f"reports/{run_id}/l3_report.json", report)
        with transaction(conn):
            for r in result.documents:
                _insert_document(conn, run_id, r)
            runs.complete(conn, run_id, report_path)
    except Exception as exc:
        if report_path:
            files.remove(data_root, report_path)
        runs.fail(conn, run_id, f"{type(exc).__name__}: {exc}")
        raise
    return run_id
