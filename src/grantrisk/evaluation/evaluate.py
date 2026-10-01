"""Stage E2: evaluate models (Spec_E2_EvaluateModels.md).

Measures how well each model run of an M2 run reproduces the tercile label: the metrics
of SPEC-E2-01 per fold, per repeat and pooled, their spread, the error sizes, the ROC
curves, the comparison with the majority baseline and the best model run, and the test of
INT-RQ-B: each transformer representation against TF-IDF (DEC-63). ``compute`` is the pure
computation; ``run`` reads the M2 run and its L3 and C1 inputs from the database and writes
the metric rows and the report in one transaction.
"""

from __future__ import annotations

import collections
import json
import sqlite3
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grantrisk.evaluation import metrics as m
from grantrisk.evaluation.metrics import LABELS, Scores
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

PERIODS = ("2021-2027", "2014-2020", "RRF", "VP")
STRATIFIED, GROUPED = "stratified", "grouped"
BASELINE = "majority"
ALL = "all"

ModelKey = tuple[str, str, str]  # (scheme, representation, classifier)

# DEC-63, fixed before the corpus results: the family of confirmatory comparisons, the level and the margin.
TFIDF = "tfidf"
FAMILY_REPRESENTATIONS = ("hubert", "e5", "bge_m3", "qwen3_8b")
FAMILY_CLASSIFIERS = ("logreg", "svm", "rf")
ALPHA = 0.05  # family-wise, with Holm's method
MARGIN = 0.02  # macro-F1: "practically equivalent" when the 95% interval lies within ±MARGIN
OUTCOMES = {
    "embedding_outperforms": "embeddings outperform TF-IDF",
    "tfidf_outperforms": "TF-IDF outperforms",
    "practically_equivalent": "practically equivalent",
    "inconclusive": "inconclusive",
}
NEGLIGIBLE = "practically negligible"
EXPLORATORY = "exploratory: outside the family of DEC-63"


class E2InputError(ValueError):
    """The input runs violate the E2 contract; nothing is written."""


@dataclass(frozen=True)
class Prediction:
    doc_id: str
    repeat: int
    fold: int
    true_label: str
    predicted_label: str
    probabilities: tuple[float, float, float]  # low, medium, high


@dataclass
class ModelEvaluation:
    key: ModelKey
    scores: dict[str, Scores]  # scope → metrics
    errors: dict[str, dict[str, float | None]]
    roc: dict[str, list[float] | None]
    n_documents: int
    test_train_ratio: float

    @property
    def fold_f1(self) -> dict[str, float]:
        return {s: sc.values["f1_macro"] for s, sc in self.scores.items() if s.startswith("fold:")}

    def summary(self, name: str) -> float | None:
        return self.scores["summary"].values.get(name)


@dataclass(frozen=True)
class E2Result:
    evaluations: list[ModelEvaluation]
    comparison: dict[str, list[dict[str, Any]]]  # scheme → rows in table order
    grid: dict[str, list[dict[str, Any]]]
    significance: dict[str, list[dict[str, Any]]]
    label_distributions: dict[str, dict[tuple[str, str], int]]  # subset → (tercile, fixed) → n
    tfidf_comparison: dict[str, Any] | None = None  # DEC-63; None when the tests are switched off


# --- One model run (SPEC-E2-01 … -04, -08) -----------------------------------------------------


def _scores(preds: Sequence[Prediction], labels: Sequence[str] = LABELS, probabilities: bool = True) -> Scores:
    return m.classification(
        [p.true_label for p in preds], [p.predicted_label for p in preds],
        [p.probabilities for p in preds] if probabilities else None, labels,
    )


def _summary(folds: Mapping[str, Scores], repeats: Mapping[str, Scores]) -> Scores:
    """SPEC-E2-02: the mean and the standard deviation (n - 1) over the folds and over the repeats."""
    out = Scores()
    for name in m.summary_names():
        for level, scopes in (("fold", folds), ("repeat", repeats)):
            mean, sd, n = m.mean_sd([s.values.get(name) for s in scopes.values()])
            out.values[f"{name}:{level}_mean"] = mean
            out.values[f"{name}:{level}_sd"] = sd
            if n < len(scopes):
                note = f"over {n} of {len(scopes)} {level}s; the others are undefined"
                out.flags[f"{name}:{level}_mean"] = out.flags[f"{name}:{level}_sd"] = note
    return out


def _class_scopes(pooled: Scores, n_repeats: int) -> dict[str, Scores]:
    """SPEC-E2-03: per class, pooled over the repeats; the confusion row as counts / repeats and row shares."""
    out = {}
    for c in LABELS:
        s = Scores()
        v = pooled.values
        for name in ("precision", "recall", "f1", "sensitivity", "specificity", "roc_auc_ovr"):
            s.values[name] = v.get(f"{name}:{c}")
            if f"{name}:{c}" in pooled.flags:
                s.flags[name] = pooled.flags[f"{name}:{c}"]
        row = [v[f"confusion:{c}:{p}"] for p in LABELS]
        s.values["documents"] = sum(row) / n_repeats
        for p, count in zip(LABELS, row):
            s.values[f"confusion:{p}"] = count / n_repeats
            s.values[f"confusion_share:{p}"] = count / sum(row) if sum(row) else None
        out[f"class:{c}"] = s
    return out


def _period_scopes(preds: Sequence[Prediction], periods: Mapping[str, str]) -> dict[str, Scores]:
    """SPEC-E2-08: macro-F1 per period, pooled over the repeats, with the number of documents."""
    out = {}
    for period in PERIODS:
        sub = [p for p in preds if periods[p.doc_id] == period]
        s = Scores()
        s.values["documents"] = len({p.doc_id for p in sub})
        s.values["n_predictions"] = len(sub)
        if not sub:
            s.values["f1_macro"] = s.values["accuracy"] = None
            s.flags["f1_macro"] = s.flags["accuracy"] = "no documents in this period"
        else:
            present = [c for c in LABELS if any(p.true_label == c or p.predicted_label == c for p in sub)]
            scores = _scores(sub, present, probabilities=False)
            s.values["f1_macro"] = scores.values["f1_macro"]
            s.values["accuracy"] = scores.values["accuracy"]
            if len(present) < len(LABELS):
                s.flags["f1_macro"] = "the mean over the classes present: " + ", ".join(present)
        out[f"period:{period}"] = s
    return out


def evaluate_model(key: ModelKey, preds: Sequence[Prediction], periods: Mapping[str, str]) -> ModelEvaluation:
    """Every metric scope of one model run."""
    preds = sorted(preds, key=lambda p: (p.repeat, p.fold, p.doc_id))
    by_fold: dict[tuple[int, int], list[Prediction]] = collections.defaultdict(list)
    by_repeat: dict[int, list[Prediction]] = collections.defaultdict(list)
    for p in preds:
        by_fold[(p.repeat, p.fold)].append(p)
        by_repeat[p.repeat].append(p)
    folds = {f"fold:{r}.{k}": _scores(ps) for (r, k), ps in sorted(by_fold.items())}
    repeats = {f"repeat:{r}": _scores(ps) for r, ps in sorted(by_repeat.items())}
    pooled = _scores(preds)
    n_documents = len({p.doc_id for p in preds})
    scores = {**folds, **repeats, "pooled": pooled, "summary": _summary(folds, repeats),
              **_class_scopes(pooled, len(repeats)), **_period_scopes(preds, periods)}
    ratios = [len(ps) / (n_documents - len(ps)) for ps in by_fold.values() if n_documents > len(ps)]
    return ModelEvaluation(
        key=key,
        scores=scores,
        errors=m.error_table({r: s for r, s in enumerate(repeats.values())}),
        roc=m.roc_curves([p.true_label for p in preds], [p.probabilities for p in preds]),
        n_documents=n_documents,
        test_train_ratio=statistics.fmean(ratios) if ratios else 0.0,
    )


# --- Across model runs (SPEC-E2-05, -06, -09) ---------------------------------------------------


def _baseline_of(key: ModelKey, by_key: Mapping[ModelKey, ModelEvaluation]) -> ModelEvaluation | None:
    """The majority baseline on the same representation, or else the first baseline of the scheme."""
    scheme, representation, _ = key
    same = by_key.get((scheme, representation, BASELINE))
    if same is not None:
        return same
    others = sorted(k for k in by_key if k[0] == scheme and k[2] == BASELINE)
    return by_key[others[0]] if others else None


def comparison(evaluations: Sequence[ModelEvaluation]) -> dict[str, list[dict[str, Any]]]:
    """SPEC-E2-05, -06: one row per model run and scheme, sorted by mean macro-F1; the best is marked.

    The best model run is chosen in the standard (stratified) scheme only, among the model
    runs that are not the baseline; equal means are all marked best (§4).
    """
    by_key = {e.key: e for e in evaluations}
    out: dict[str, list[dict[str, Any]]] = {}
    for scheme in sorted({e.key[0] for e in evaluations}, key=lambda s: (s != STRATIFIED, s)):
        rows = []
        for e in (e for e in evaluations if e.key[0] == scheme):
            f1 = e.summary("f1_macro:fold_mean")
            is_baseline = e.key[2] == BASELINE
            base = None if is_baseline else _baseline_of(e.key, by_key)
            base_f1 = None if base is None else base.summary("f1_macro:fold_mean")
            rows.append({
                "scheme": scheme, "representation": e.key[1], "classifier": e.key[2],
                "f1_macro_mean": f1, "f1_macro_sd": e.summary("f1_macro:fold_sd"),
                "accuracy_mean": e.summary("accuracy:fold_mean"),
                "roc_auc_ovr_macro_mean": e.summary("roc_auc_ovr_macro:fold_mean"),
                "kappa_quadratic_mean": e.summary("kappa_quadratic:fold_mean"),
                "error2_share": e.summary("error2_share:repeat_mean"),
                "is_baseline": is_baseline,
                "baseline": None if base is None else "/".join(base.key[1:]),
                "not_above_baseline": None if is_baseline or base_f1 is None or f1 is None else f1 <= base_f1,
            })
        rows.sort(key=lambda r: (r["f1_macro_mean"] is None, -(r["f1_macro_mean"] or 0.0),
                                 r["representation"], r["classifier"]))
        rank = 0
        for i, r in enumerate(rows):
            if i == 0 or r["f1_macro_mean"] != rows[i - 1]["f1_macro_mean"]:
                rank = i + 1
            r["rank"] = rank
        candidates = [r["f1_macro_mean"] for r in rows if not r["is_baseline"] and r["f1_macro_mean"] is not None]
        best = max(candidates) if candidates and scheme == STRATIFIED else None
        for r in rows:
            r["is_best"] = best is not None and not r["is_baseline"] and r["f1_macro_mean"] == best
        out[scheme] = rows
    return out


def grid(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """SPEC-E2-06: the mean macro-F1 of each representation (without the baseline) and of each classifier."""
    out = []
    for axis, index in (("representation", "representation"), ("classifier", "classifier")):
        for key in sorted({r[index] for r in rows}):
            values = [r["f1_macro_mean"] for r in rows if r[index] == key and r["f1_macro_mean"] is not None
                      and not (axis == "representation" and r["is_baseline"])]
            out.append({"axis": axis, "key": key, "f1_macro_mean": statistics.fmean(values) if values else None,
                        "n_models": len(values)})
    return out


def significance(evaluations: Sequence[ModelEvaluation], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """SPEC-E2-09: the first best model run against every other one, on the paired fold differences."""
    best = next((r for r in rows if r["is_best"]), None)
    if best is None:
        return []
    by_key = {e.key: e for e in evaluations}
    reference = by_key[(best["scheme"], best["representation"], best["classifier"])]
    ref_folds = reference.fold_f1
    out = []
    for r in rows:
        key = (r["scheme"], r["representation"], r["classifier"])
        if key == reference.key:
            continue
        other = by_key[key].fold_f1
        if set(other) != set(ref_folds) or None in other.values() or None in ref_folds.values():
            continue  # cannot happen with one shared fold assignment and stratified folds
        diffs = [ref_folds[f] - other[f] for f in sorted(ref_folds)]
        test = m.corrected_t_test(diffs, reference.test_train_ratio)
        out.append({"scheme": r["scheme"], "representation": r["representation"], "classifier": r["classifier"],
                    "reference_representation": reference.key[1], "reference_classifier": reference.key[2],
                    "n_folds": len(diffs), **test})
    for t, p in zip(out, m.holm([t["p_value"] for t in out])):  # DEC-52: over the comparisons of this scheme
        t["p_holm"] = p
    return out


def outcome(mean_diff: float, ci_low: float, ci_high: float, p_holm: float) -> tuple[str, str | None]:
    """DEC-63: the outcome of one comparison (representation minus TF-IDF), and its note.

    Significant (Holm-adjusted p < ALPHA): the better one outperforms, "practically negligible" if
    the 95% interval lies within ±MARGIN. Not significant: practically equivalent if the interval lies
    within ±MARGIN, otherwise inconclusive.
    """
    within = -MARGIN <= ci_low and ci_high <= MARGIN
    if p_holm < ALPHA:
        return ("embedding_outperforms" if mean_diff > 0 else "tfidf_outperforms"), (NEGLIGIBLE if within else None)
    if within:
        return "practically_equivalent", None
    if ci_low > 0 or ci_high < 0:  # possible: the interval is not adjusted for the 12 comparisons
        return "inconclusive", "the unadjusted 95% interval excludes 0, but the difference is not significant after Holm"
    return "inconclusive", None


def tfidf_comparison(evaluations: Sequence[ModelEvaluation]) -> dict[str, Any]:
    """DEC-63: each transformer representation against TF-IDF with the same classifier (INT-RQ-B).

    The stratified scheme only. The difference is representation minus TF-IDF, per fold. Holm's
    adjustment runs over the family of the representations and classifiers present; a representation
    outside the family is reported beside it as exploratory. H1 is supported if at least one
    comparison of the family has the outcome "embeddings outperform TF-IDF".
    """
    by_key = {e.key: e for e in evaluations}
    rows = []
    for (scheme, rep, clf), e in by_key.items():
        reference = by_key.get((scheme, TFIDF, clf))
        if scheme != STRATIFIED or rep == TFIDF or clf == BASELINE or reference is None:
            continue
        folds, ref_folds = e.fold_f1, reference.fold_f1
        if set(folds) != set(ref_folds) or None in folds.values() or None in ref_folds.values():
            continue  # cannot happen with one shared fold assignment and stratified folds
        diffs = [folds[f] - ref_folds[f] for f in sorted(ref_folds)]
        test = m.corrected_t_test(diffs, reference.test_train_ratio)
        rows.append({"scheme": scheme, "representation": rep, "classifier": clf, "n_folds": len(diffs), **test,
                     "in_family": rep in FAMILY_REPRESENTATIONS and clf in FAMILY_CLASSIFIERS})

    def order(r):
        rep, clf = r["representation"], r["classifier"]
        return (not r["in_family"], FAMILY_REPRESENTATIONS.index(rep) if rep in FAMILY_REPRESENTATIONS else 0, rep,
                FAMILY_CLASSIFIERS.index(clf) if clf in FAMILY_CLASSIFIERS else 0, clf)

    rows.sort(key=order)
    family = [r for r in rows if r["in_family"]]
    for r, p in zip(family, m.holm([r["p_value"] for r in family])):
        r["p_holm"] = p
        r["outcome"], r["note"] = outcome(r["mean_diff"], r["ci_low"], r["ci_high"], p)
    for r in rows:
        if not r["in_family"]:
            r["p_holm"], r["outcome"], r["note"] = None, None, EXPLORATORY
    present = {(r["representation"], r["classifier"]) for r in family}
    return {
        "rows": rows,
        "family_size": len(family),
        "absent_from_family": [f"{rep}/{clf}" for rep in FAMILY_REPRESENTATIONS for clf in FAMILY_CLASSIFIERS
                               if (rep, clf) not in present],
        "outcome_counts": {k: sum(r["outcome"] == k for r in family) for k in OUTCOMES},
        "h1_supported": any(r["outcome"] == "embedding_outperforms" for r in family) if family else None,
        "alpha": ALPHA,
        "margin": MARGIN,
    }


def label_distributions(labels: Sequence[tuple[str, str, str]]) -> dict[str, dict[tuple[str, str], int]]:
    """SPEC-E2-07: the cross-table of (tercile, fixed) labels, overall and per period.

    ``labels`` holds (tercile label, fixed label, period) per document.
    """
    out = {}
    for subset in (ALL, *PERIODS):
        counts = collections.Counter((t, f) for t, f, p in labels if subset in (ALL, p))
        out[subset] = {(t, f): counts.get((t, f), 0) for t in LABELS for f in LABELS}
    return out


def compute(
    predictions: Mapping[ModelKey, Sequence[Prediction]],
    periods: Mapping[str, str],
    labels: Sequence[tuple[str, str, str]],
    with_significance: bool = True,
) -> E2Result:
    """SPEC-E2-01 … -10 for every model run of one M2 run."""
    evaluations = [evaluate_model(k, predictions[k], periods) for k in sorted(predictions)]
    table = comparison(evaluations)
    return E2Result(
        evaluations=evaluations,
        comparison=table,
        grid={s: grid(rows) for s, rows in table.items()},
        significance={s: significance(evaluations, rows) for s, rows in table.items()} if with_significance else {},
        label_distributions=label_distributions(labels),
        tfidf_comparison=tfidf_comparison(evaluations) if with_significance else None,
    )


# --- Report -------------------------------------------------------------------------------------


def _fmt(x: float | None, digits: int = 3) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def markdown_report(result: E2Result, meta: Mapping[str, Any]) -> str:
    """A short summary; E3 builds the full tables and the dashboard."""
    lines = [
        "# E2: evaluation of the models",
        "",
        f"M2 run `{meta['input_runs']['M2']}`, L3 run `{meta['input_runs']['L3']}` (reference: the tercile label, "
        f"DEC-07), {meta['documents']} documents. Metric definitions version {m.METRICS_VERSION}.",
        "",
        "macro-F1, accuracy, ROC-AUC (one-vs-rest, macro) and κw (quadratic weights): mean over the "
        f"{meta['n_folds']} test folds; ± the standard deviation (n − 1). |e| = 2: the share of documents whose "
        "predicted label is two classes away from the tercile label, averaged over the repeats.",
    ]
    for scheme, rows in result.comparison.items():
        lines += ["", f"## Scheme `{scheme}`", "",
                  "| Rank | Representation | Classifier | macro-F1 | Accuracy | ROC-AUC | κw | abs(e) = 2 | Note |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            note = "best" if r["is_best"] else "baseline" if r["is_baseline"] else (
                "not above the baseline" if r["not_above_baseline"] else "")
            lines.append(
                f"| {r['rank']} | {r['representation']} | {r['classifier']} | {_fmt(r['f1_macro_mean'])} ± "
                f"{_fmt(r['f1_macro_sd'])} | {_fmt(r['accuracy_mean'])} | {_fmt(r['roc_auc_ovr_macro_mean'])} | "
                f"{_fmt(r['kappa_quadratic_mean'])} | {_fmt(r['error2_share'])} | {note} |"
            )
    if meta["failed_in_m2"]:
        lines += ["", "Model runs that failed in M2 and are not evaluated: "
                  + ", ".join(f"`{'/'.join(k)}`" for k in meta["failed_in_m2"]) + "."]
    lines += tfidf_comparison_markdown(result.tfidf_comparison, meta.get("m2_tuning"))
    return "\n".join(lines) + "\n"


def _signed(x: float) -> str:
    return f"{x:+.3f}"


def tfidf_comparison_markdown(c: Mapping[str, Any] | None, tuning: Mapping[str, Any] | None) -> list[str]:
    """The section of the E2 report on the test of INT-RQ-B (DEC-63)."""
    if c is None:
        return []
    analysis = (f"`C` of {', '.join(tuning['classifiers'])} tuned inside each training fold over {tuning['C']} "
                "(DEC-63 (b), DEC-67): the confirmatory analysis" if tuning else
                "the fixed defaults of SPEC-M2-04 (DEC-63 (a)), reported beside the confirmatory analysis")
    lines = ["", "## Embeddings against TF-IDF (DEC-63)", "",
             f"Hyperparameters: {analysis}. Difference: mean macro-F1 of the representation minus that of TF-IDF with "
             "the same classifier, over the paired folds; 95% interval and p from the corrected resampled t-test "
             f"(Nadeau and Bengio, 2003), two-sided; p (Holm) over the {c['family_size']} comparisons of the family. "
             f"Margin of practical equivalence: ±{c['margin']}.", ""]
    if not c["rows"]:
        return lines + ["No comparison: the M2 run has no TF-IDF model run beside a transformer representation."]
    lines += ["| Representation | Classifier | Difference [95% interval] | p | p (Holm) | Outcome |",
              "|---|---|---|---|---|---|"]
    for r in c["rows"]:
        result = OUTCOMES[r["outcome"]] if r["outcome"] else ""
        if r["note"]:
            result = f"{result} ({r['note']})" if result else r["note"]
        holm = "–" if r["p_holm"] is None else f"{r['p_holm']:.4f}"
        lines.append(f"| {r['representation']} | {r['classifier']} | {_signed(r['mean_diff'])} "
                     f"[{_signed(r['ci_low'])}, {_signed(r['ci_high'])}] | {r['p_value']:.4f} | {holm} | {result} |")
    verdict = ("supported: at least one comparison has the outcome \"embeddings outperform TF-IDF\"" if c["h1_supported"]
               else "not supported: no comparison has the outcome \"embeddings outperform TF-IDF\"")
    lines += ["", f"H1 (INT-RQ-B) is {verdict}. Outcomes: "
              + ", ".join(f"{OUTCOMES[k]} {n}" for k, n in c["outcome_counts"].items()) + "."]
    if c["absent_from_family"]:
        lines.append("Not computed, so outside the family: " + ", ".join(f"`{k}`" for k in c["absent_from_family"]) + ".")
    return lines


def json_report(result: E2Result, meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{k: v for k, v in meta.items() if k != "run_id"},
        "metrics_version": m.METRICS_VERSION,
        "definitions": {
            **m.DEFINITIONS,
            "scopes": {
                "fold:<r>.<k>": "the test fold k of repeat r (as stored by M2, from 0)",
                "repeat:<r>": "the out-of-fold predictions of repeat r, every document once",
                "pooled": "the out-of-fold predictions of all repeats together, every document once per repeat",
                "summary": "<m>:fold_mean/_sd over the folds, <m>:repeat_mean/_sd over the repeats; sd with n - 1",
                "class:<c>": "pooled; confusion:<p> is the pooled count divided by the number of repeats",
                "period:<p>": "pooled; macro-F1 over the classes present in the period",
            },
            "reference": "the tercile label of the L3 run (DEC-07, DEC-12)",
            "best": "the highest mean macro-F1 over the folds in the stratified scheme, the baseline excluded",
            "not_above_baseline": "mean macro-F1 not above that of the majority baseline on the same representation",
            "grid": "representation means leave out the baseline; classifier means run over the representations",
            "significance": "corrected resampled t-test (Nadeau and Bengio, 2003) on the paired fold differences of "
                            "macro-F1, the first best model run minus the other; two-sided; p_holm: adjusted for the "
                            "comparisons of the scheme with Holm's step-down method (DEC-52)",
            "tfidf_comparison": "DEC-63: each transformer representation minus TF-IDF with the same classifier, "
                                "stratified scheme; corrected resampled t-test and its 95% interval; p_holm over the "
                                f"family {list(FAMILY_REPRESENTATIONS)} × {list(FAMILY_CLASSIFIERS)}; outcome: "
                                "embedding_outperforms / tfidf_outperforms (p_holm < 0.05; note 'practically "
                                "negligible' if the interval lies within ±0.02), otherwise practically_equivalent "
                                "(interval within ±0.02) or inconclusive",
            "roc_curves": f"pooled over the repeats; TPR at {m.ROC_GRID} FPR points, linear between the ROC vertices",
        },
        "comparison": result.comparison,
        "grid": result.grid,
        "best": {s: ["/".join((r["representation"], r["classifier"])) for r in rows if r["is_best"]]
                 for s, rows in result.comparison.items()},
        "not_above_baseline": {s: ["/".join((r["representation"], r["classifier"])) for r in rows
                                   if r["not_above_baseline"]] for s, rows in result.comparison.items()},
        "significance": result.significance or "not run",
        "tfidf_comparison": result.tfidf_comparison or "not run",
        "label_distributions": {
            s: {f"{t}/{f}": n for (t, f), n in cells.items()} for s, cells in result.label_distributions.items()
        },
    }


# --- Database run (SPEC-E2-11) ----------------------------------------------------------------


def input_of(conn: sqlite3.Connection, run_id: str, stage: str) -> str:
    """The single input run of ``stage`` that ``run_id`` records."""
    found = [r for r in runs.inputs(conn, run_id) if runs.get(conn, r)["stage"] == stage]
    if len(found) != 1:
        raise E2InputError(f"run {run_id} has no single {stage} input run")
    return found[0]


def load_predictions(conn: sqlite3.Connection, m2_run_id: str) -> dict[ModelKey, list[Prediction]]:
    """The predictions of the complete model runs of a complete M2 run."""
    rows = conn.execute(
        "SELECT p.scheme, p.representation, p.classifier, p.doc_id, p.repeat, p.fold, p.true_label,"
        " p.predicted_label, p.p_low, p.p_medium, p.p_high"
        " FROM predictions p JOIN model_runs mr USING (run_id, scheme, representation, classifier)"
        " WHERE p.run_id = ? AND mr.status = 'complete'"
        " ORDER BY p.scheme, p.representation, p.classifier, p.doc_id, p.repeat",
        (m2_run_id,),
    )
    out: dict[ModelKey, list[Prediction]] = collections.defaultdict(list)
    for r in rows:
        out[(r[0], r[1], r[2])].append(
            Prediction(r[3], r[4], r[5], r[6], r[7], (r[8], r[9], r[10]))
        )
    return dict(out)


def _check(predictions: Mapping[ModelKey, Sequence[Prediction]], terciles: Mapping[str, str]) -> None:
    problems = []
    for key, preds in predictions.items():
        docs = {p.doc_id for p in preds}
        if docs != set(terciles):
            problems.append(f"{'/'.join(key)}: its documents differ from those of the L3 run")
        wrong = sorted({p.doc_id for p in preds if terciles.get(p.doc_id) not in (None, p.true_label)})
        if wrong:
            problems.append(f"{'/'.join(key)}: true labels differ from the L3 tercile labels for {wrong[:10]}")
    if problems:
        raise E2InputError("the M2 run does not match its L3 run:\n  " + "\n  ".join(problems))


def _metric_rows(run_id: str, evaluations: Sequence[ModelEvaluation]) -> list[tuple]:
    rows = []
    for e in evaluations:
        for scope, scores in e.scores.items():
            for name, value in scores.values.items():
                rows.append((run_id, *e.key, scope, name, value, scores.flags.get(name)))
    return rows


def _insert(conn: sqlite3.Connection, run_id: str, result: E2Result) -> None:
    conn.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)", _metric_rows(run_id, result.evaluations))
    conn.executemany(
        "INSERT INTO error_sizes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(run_id, *e.key, int(err), t["mean_count"], t["sd_count"], t["share"])
         for e in result.evaluations for err, t in e.errors.items()],
    )
    conn.executemany(
        "INSERT INTO roc_curves VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(run_id, *e.key, curve, i, i / (m.ROC_GRID - 1), tpr)
         for e in result.evaluations for curve, tprs in e.roc.items() if tprs is not None
         for i, tpr in enumerate(tprs)],
    )
    conn.executemany(
        "INSERT INTO model_comparison VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(run_id, r["scheme"], r["representation"], r["classifier"], r["rank"], r["f1_macro_mean"], r["f1_macro_sd"],
          r["accuracy_mean"], r["roc_auc_ovr_macro_mean"], r["kappa_quadratic_mean"], r["error2_share"],
          int(r["is_baseline"]), int(r["is_best"]),
          None if r["not_above_baseline"] is None else int(r["not_above_baseline"]))
         for rows in result.comparison.values() for r in rows],
    )
    conn.executemany(
        "INSERT INTO model_grid VALUES (?, ?, ?, ?, ?, ?)",
        [(run_id, s, g["axis"], g["key"], g["f1_macro_mean"], g["n_models"])
         for s, rows in result.grid.items() for g in rows],
    )
    conn.executemany(
        "INSERT INTO significance (run_id, scheme, representation, classifier, reference_representation,"
        " reference_classifier, n_folds, mean_diff, sd_diff, t_stat, df, p_value, p_holm)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(run_id, t["scheme"], t["representation"], t["classifier"], t["reference_representation"],
          t["reference_classifier"], t["n_folds"], t["mean_diff"], t["sd_diff"], t["t"], t["df"], t["p_value"],
          t["p_holm"])
         for rows in result.significance.values() for t in rows],
    )
    conn.executemany(
        "INSERT INTO tfidf_comparisons (run_id, scheme, representation, classifier, n_folds, mean_diff, sd_diff,"
        " ci_low, ci_high, t_stat, df, p_value, in_family, p_holm, outcome, note)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(run_id, t["scheme"], t["representation"], t["classifier"], t["n_folds"], t["mean_diff"], t["sd_diff"],
          t["ci_low"], t["ci_high"], t["t"], t["df"], t["p_value"], int(t["in_family"]), t["p_holm"], t["outcome"],
          t["note"])
         for t in (result.tfidf_comparison or {}).get("rows", [])],
    )
    conn.executemany(
        "INSERT INTO label_distributions VALUES (?, ?, ?, ?, ?)",
        [(run_id, s, t, f, n) for s, cells in result.label_distributions.items() for (t, f), n in cells.items()],
    )


def run(
    conn: sqlite3.Connection,
    config_values: Mapping[str, Any],
    data_root: Path,
    *,
    m2_run_id: str,
) -> str:
    """Evaluate every complete model run of an M2 run. Returns the new run id.

    The L3 run (the reference) and the C1 run (the periods) are the ones the M2 run and
    its L3 run record. On any error the run is marked failed, and neither rows nor
    report files remain.
    """
    runs.require_complete(conn, m2_run_id, "M2")
    l3_run_id = input_of(conn, m2_run_id, "L3")
    runs.require_complete(conn, l3_run_id, "L3")
    c1_run_id = input_of(conn, l3_run_id, "C1")
    runs.require_complete(conn, c1_run_id, "C1")
    settings = config_values.get("evaluate") or {}
    m2_tuning = (json.loads(runs.get(conn, m2_run_id)["config_json"]).get("m2_run") or {}).get("tuning")
    snapshot = {**config_values, "e2_run": {
        "metrics_version": m.METRICS_VERSION,
        "tfidf_comparison": {"family": [list(FAMILY_REPRESENTATIONS), list(FAMILY_CLASSIFIERS)], "alpha": ALPHA,
                             "margin": MARGIN, "m2_tuning": m2_tuning}}}
    run_id = runs.start(conn, "E2", snapshot, inputs=sorted({m2_run_id, l3_run_id, c1_run_id}))
    written: list[str] = []
    try:
        label_rows = conn.execute(
            "SELECT r.doc_id, r.tercile_label, r.fixed_label, d.period FROM risk_labels r"
            " LEFT JOIN documents d ON d.run_id = ? AND d.doc_id = r.doc_id WHERE r.run_id = ? ORDER BY r.doc_id",
            (c1_run_id, l3_run_id),
        ).fetchall()
        lacking = [r[0] for r in label_rows if r[3] is None]
        if lacking:
            raise E2InputError(f"C1 run {c1_run_id} has no period for {lacking[:10]}")
        terciles = {r[0]: r[1] for r in label_rows}
        periods = {r[0]: r[3] for r in label_rows}
        predictions = load_predictions(conn, m2_run_id)
        if not predictions:
            raise E2InputError(f"M2 run {m2_run_id} has no complete model run")
        _check(predictions, terciles)
        result = compute(predictions, periods, [(r[1], r[2], r[3]) for r in label_rows],
                         with_significance=settings.get("significance", True))

        failed = [tuple(r) for r in conn.execute(
            "SELECT scheme, representation, classifier FROM model_runs WHERE run_id = ? AND status != 'complete'"
            " ORDER BY 1, 2, 3", (m2_run_id,))]
        n_folds = max(len([s for s in e.scores if s.startswith("fold:")]) for e in result.evaluations)
        meta = {
            "run_id": run_id,
            "input_runs": {"M2": m2_run_id, "L3": l3_run_id, "C1": c1_run_id},
            "documents": len(terciles),
            "documents_by_period": dict(sorted(collections.Counter(periods.values()).items())),
            "n_folds": n_folds,
            "model_runs": ["/".join(e.key) for e in result.evaluations],
            "failed_in_m2": failed,
            "m2_tuning": m2_tuning,
        }
        folder = f"reports/{run_id}"
        written.append(files.write_text(data_root, f"{folder}/e2_report.md", markdown_report(result, meta)))
        report = json_report(result, meta)
        report["files"] = ["e2_report.md"]
        report_path = files.write_json(data_root, f"{folder}/e2_report.json", report)
        written.append(report_path)
        with transaction(conn):
            _insert(conn, run_id, result)
            runs.complete(conn, run_id, report_path)
    except Exception as exc:
        for path in written:
            files.remove(data_root, path)
        runs.fail(conn, run_id, f"{type(exc).__name__}: {exc}")
        raise
    return run_id
