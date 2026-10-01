"""The chain of runs one E3 report is built from, and the stored results it reads (Spec_E3_Report.md §2.1).

E3 creates no new numbers: everything here is read from stored runs. The chain is
found through the lineage the runs record (ARC-02): the E2 run names its M2, L3 and
C1 runs, the L3 run its L2 and C1 runs, and the L2 run its L1 and C2 runs.
"""

from __future__ import annotations

import collections
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grantrisk.labelling import consolidate
from grantrisk.labelling.scoring import LABELS
from grantrisk.store import runs

STRATIFIED = "stratified"
BASELINE = "majority"


class E3InputError(ValueError):
    """The named runs do not form one chain; nothing is written."""


@dataclass
class Chain:
    """The runs of one chain, by stage. ``l1`` maps each L1 source (manual, regex, llm:<model>) to its run."""

    c1: str
    l2: str
    l3: str
    c2: str | None = None
    l1: dict[str, str] = field(default_factory=dict)
    m1: str | None = None
    m2: str | None = None
    e1: str | None = None
    e1_l1: dict[str, str] = field(default_factory=dict)  # the L1 runs E1 evaluated
    e2: str | None = None

    def runs(self) -> dict[str, str]:
        """Label → run id, in the order of the pipeline."""
        out = {"C1": self.c1}
        if self.c2:
            out["C2"] = self.c2
        out.update({f"L1 {s}": r for s, r in sorted(self.l1.items())})
        out.update({"L2": self.l2, "L3": self.l3})
        for label, run_id in (("M1", self.m1), ("M2", self.m2)):
            if run_id:
                out[label] = run_id
        out.update({f"L1 {s} (E1)": r for s, r in sorted(self.e1_l1.items()) if self.l1.get(s) != r})
        for label, run_id in (("E1", self.e1), ("E2", self.e2)):
            if run_id:
                out[label] = run_id
        return out

    def missing(self) -> list[str]:
        """The parts of the chain that were not run (§4 of the chapter)."""
        out = []
        if not self.m1:
            out.append("M1 (represent)")
        if not self.m2:
            out.append("M2 (train & predict)")
        if not self.e2:
            out.append("E2 (evaluate models)")
        if not self.e1:
            out.append("E1 (validate extraction)")
        if "regex" not in self.l1:
            out.append("L1 regex in the labelling chain")
        if not any(s.startswith("llm:") for s in self.l1):
            out.append("L1 LLM in the labelling chain")
        return out


def _inputs_of(conn: sqlite3.Connection, run_id: str, stage: str) -> list[str]:
    return [r for r in runs.inputs(conn, run_id) if runs.get(conn, r)["stage"] == stage]


def _single(conn: sqlite3.Connection, run_id: str, stage: str, required: bool = True) -> str | None:
    found = _inputs_of(conn, run_id, stage)
    if len(found) > 1 or (required and not found):
        raise E3InputError(f"run {run_id} has no single {stage} input run")
    return found[0] if found else None


def _l1_sources(conn: sqlite3.Connection, run_ids: list[str]) -> dict[str, str]:
    out = {}
    for run_id in run_ids:
        sources = [r[0] for r in conn.execute(
            "SELECT DISTINCT source FROM factor_observations WHERE run_id = ? ORDER BY 1", (run_id,))]
        if len(sources) != 1:
            raise E3InputError(f"L1 run {run_id} holds sources {sources}, not one")
        out[sources[0]] = run_id
    return out


def resolve(
    conn: sqlite3.Connection,
    *,
    e2_run_id: str | None = None,
    e1_run_id: str | None = None,
    l3_run_id: str | None = None,
) -> Chain:
    """Find the chain from the E2 run (or, without models, from the L3 run) through the lineage."""
    if e2_run_id:
        runs.require_complete(conn, e2_run_id, "E2")
        m2 = _single(conn, e2_run_id, "M2")
        l3_of_e2 = _single(conn, e2_run_id, "L3")
        if l3_run_id and l3_run_id != l3_of_e2:
            raise E3InputError(f"E2 run {e2_run_id} evaluated L3 run {l3_of_e2}, not {l3_run_id}")
        l3_run_id = l3_of_e2
    elif not l3_run_id:
        raise E3InputError("name an E2 run, or an L3 run for a report without models")
    else:
        m2 = None
    runs.require_complete(conn, l3_run_id, "L3")
    l2 = _single(conn, l3_run_id, "L2")
    c1 = _single(conn, l3_run_id, "C1")
    chain = Chain(c1=c1, l2=l2, l3=l3_run_id, c2=_single(conn, l2, "C2", required=False),
                  l1=_l1_sources(conn, _inputs_of(conn, l2, "L1")), m2=m2, e2=e2_run_id)
    if m2:
        chain.m1 = _single(conn, m2, "M1", required=False)
    if e1_run_id:
        runs.require_complete(conn, e1_run_id, "E1")
        chain.e1 = e1_run_id
        chain.e1_l1 = {s: r for s, r in _l1_sources(conn, _inputs_of(conn, e1_run_id, "L1")).items() if s != "manual"}
    for run_id in chain.runs().values():
        row = runs.get(conn, run_id)
        if row["status"] != "complete":
            raise E3InputError(f"run {run_id} of the chain is {row['status']}, not complete")
    return chain


# --- The stored results -------------------------------------------------------------------------


@dataclass
class ModelData:
    """What the dashboard shows for one model run (SPEC-E3-01)."""

    key: tuple[str, str, str]
    fold_f1: list[tuple[str, float | None]]
    confusion: dict[str, dict[str, float]]  # true → predicted → count / repeats
    confusion_share: dict[str, dict[str, float | None]]
    roc: dict[str, list[float]]
    roc_auc: dict[str, float | None]  # pooled, per class and macro
    errors: dict[int, dict[str, float | None]]
    periods: dict[str, dict[str, Any]]


@dataclass
class Data:
    chain: Chain
    code_version: str
    documents: dict[str, sqlite3.Row]
    labels: dict[str, sqlite3.Row]
    factor_points: dict[tuple[str, str], sqlite3.Row]
    consolidated: dict[tuple[str, str], sqlite3.Row]
    l3_report: dict[str, Any]
    observations: dict[str, dict]  # regex / llm → (doc_id, factor) → consolidate.Observation
    comparison: list[sqlite3.Row] = field(default_factory=list)
    grid: list[sqlite3.Row] = field(default_factory=list)
    significance: list[sqlite3.Row] = field(default_factory=list)
    tfidf_comparisons: list[sqlite3.Row] = field(default_factory=list)  # DEC-63
    label_distributions: list[sqlite3.Row] = field(default_factory=list)
    models: dict[tuple[str, str, str], ModelData] = field(default_factory=dict)
    e2_report: dict[str, Any] | None = None
    best: tuple[str, str, str] | None = None
    tied_best: list[tuple[str, str, str]] = field(default_factory=list)
    best_predictions: dict[str, list[str]] = field(default_factory=dict)  # doc_id → predicted label per repeat
    top_terms: dict[str, list[sqlite3.Row]] = field(default_factory=dict)
    e1_agreements: list[sqlite3.Row] = field(default_factory=list)
    e1_label_agreements: list[sqlite3.Row] = field(default_factory=list)
    e1_report: dict[str, Any] | None = None
    run_info: dict[str, sqlite3.Row] = field(default_factory=dict)

    @property
    def n_repeats(self) -> int:
        return max((len(v) for v in self.best_predictions.values()), default=0)


def _report(conn: sqlite3.Connection, data_root: Path, run_id: str) -> dict[str, Any]:
    path = runs.get(conn, run_id)["report_path"]
    return json.loads((data_root / path).read_text(encoding="utf-8")) if path else {}


def _model_data(conn: sqlite3.Connection, e2: str, key: tuple[str, str, str]) -> ModelData:
    args = (e2, *key)
    where = "run_id = ? AND scheme = ? AND representation = ? AND classifier = ?"
    values: dict[str, dict[str, float | None]] = collections.defaultdict(dict)
    for r in conn.execute(f"SELECT scope, name, value FROM metrics WHERE {where}"
                          " AND (scope LIKE 'fold:%' OR scope LIKE 'class:%' OR scope LIKE 'period:%'"
                          " OR scope = 'pooled')", args):
        values[r[0]][r[1]] = r[2]

    def fold_order(scope: str) -> tuple[int, int]:
        r, k = scope.split(":")[1].split(".")
        return int(r), int(k)

    folds = sorted((s for s in values if s.startswith("fold:")), key=fold_order)
    roc: dict[str, list[float]] = collections.defaultdict(list)
    for r in conn.execute(f"SELECT curve, tpr FROM roc_curves WHERE {where} ORDER BY curve, point", args):
        roc[r[0]].append(r[1])
    errors = {r[0]: {"mean_count": r[1], "sd_count": r[2], "share": r[3]} for r in conn.execute(
        f"SELECT error, mean_count, sd_count, share FROM error_sizes WHERE {where} ORDER BY error", args)}
    pooled = values.get("pooled", {})
    return ModelData(
        key=key,
        fold_f1=[(s.split(":")[1], values[s].get("f1_macro")) for s in folds],
        confusion={t: {p: values[f"class:{t}"].get(f"confusion:{p}") for p in LABELS} for t in LABELS},
        confusion_share={t: {p: values[f"class:{t}"].get(f"confusion_share:{p}") for p in LABELS} for t in LABELS},
        roc=dict(roc),
        roc_auc={**{c: pooled.get(f"roc_auc_ovr:{c}") for c in LABELS}, "macro": pooled.get("roc_auc_ovr_macro")},
        errors=errors,
        periods={s.split(":", 1)[1]: dict(v) for s, v in sorted(values.items()) if s.startswith("period:")},
    )


def load(conn: sqlite3.Connection, data_root: Path, chain: Chain) -> Data:
    """Read every stored result the report shows."""
    c = chain
    data = Data(
        chain=c,
        code_version=runs.code_version(),
        documents={r["doc_id"]: r for r in conn.execute(
            "SELECT * FROM documents WHERE run_id = ? ORDER BY doc_id", (c.c1,))},
        labels={r["doc_id"]: r for r in conn.execute(
            "SELECT * FROM risk_labels WHERE run_id = ? ORDER BY doc_id", (c.l3,))},
        factor_points={(r["doc_id"], r["factor"]): r for r in conn.execute(
            "SELECT * FROM risk_label_factors WHERE run_id = ? ORDER BY doc_id, factor", (c.l3,))},
        consolidated={(r["doc_id"], r["factor"]): r for r in conn.execute(
            "SELECT * FROM consolidated_factors WHERE run_id = ? ORDER BY doc_id, factor", (c.l2,))},
        l3_report=_report(conn, data_root, c.l3),
        observations={
            ("llm" if s.startswith("llm:") else s): consolidate.load_observations(conn, r)
            for s, r in c.l1.items() if s != "manual"
        },
    )
    lacking = sorted(d for d in data.labels if d not in data.documents)
    if lacking:
        raise E3InputError(f"C1 run {c.c1} has no document record for {lacking[:10]}")
    data.run_info = {label: runs.get(conn, r) for label, r in c.runs().items()}

    if c.e2:
        data.comparison = conn.execute(
            "SELECT * FROM model_comparison WHERE run_id = ? ORDER BY scheme != 'stratified', scheme, rank,"
            " representation, classifier", (c.e2,)).fetchall()
        data.grid = conn.execute("SELECT * FROM model_grid WHERE run_id = ? ORDER BY scheme, axis, key",
                                 (c.e2,)).fetchall()
        data.significance = conn.execute(
            "SELECT * FROM significance WHERE run_id = ? ORDER BY scheme, p_value, representation, classifier",
            (c.e2,)).fetchall()
        data.tfidf_comparisons = conn.execute(
            "SELECT * FROM tfidf_comparisons WHERE run_id = ? ORDER BY in_family DESC, rowid", (c.e2,)).fetchall()
        data.label_distributions = conn.execute(
            "SELECT * FROM label_distributions WHERE run_id = ? ORDER BY subset, tercile_label, fixed_label",
            (c.e2,)).fetchall()
        data.e2_report = _report(conn, data_root, c.e2)
        data.models = {k: _model_data(conn, c.e2, k) for k in
                       ((r["scheme"], r["representation"], r["classifier"]) for r in data.comparison)}
        best = [(r["scheme"], r["representation"], r["classifier"]) for r in data.comparison
                if r["is_best"] and r["scheme"] == STRATIFIED]
        if best:
            data.best, data.tied_best = best[0], best[1:]
            preds: dict[str, list[str]] = collections.defaultdict(list)
            for r in conn.execute(
                "SELECT doc_id, predicted_label FROM predictions WHERE run_id = ? AND scheme = ? AND representation = ?"
                " AND classifier = ? ORDER BY doc_id, repeat", (c.m2, *data.best)):
                preds[r[0]].append(r[1])
            data.best_predictions = dict(preds)
        for r in conn.execute("SELECT * FROM top_terms WHERE run_id = ? AND scheme = ? ORDER BY class_label, rank",
                              (c.m2, STRATIFIED)):
            data.top_terms.setdefault(r["class_label"], []).append(r)
    if c.e1:
        data.e1_agreements = conn.execute(
            "SELECT * FROM extraction_agreements WHERE run_id = ? AND subset = 'all' ORDER BY source, factor",
            (c.e1,)).fetchall()
        data.e1_label_agreements = conn.execute(
            "SELECT * FROM extraction_label_agreements WHERE run_id = ? ORDER BY source, label_kind", (c.e1,)).fetchall()
        data.e1_report = _report(conn, data_root, c.e1)
    return data


def sort_sources(sources: Mapping[str, Any] | list[str]) -> list[str]:
    """Regex first, then the LLM models, then others, and the old baseline last (as E1 orders them)."""
    def key(s: str) -> tuple[int, str]:
        return (0 if s == "regex" else 1 if s.startswith("llm:") else 3 if s == "old_regex" else 2, s)

    return sorted(set(sources), key=key)
