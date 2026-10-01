"""The thesis tables of the explanatory analyses (DEC-64, DEC-70), read from the E4 run of the chain.

Each function returns (CSV text or None, Markdown text). The numbers are those E4 stored:
the mean and SD over the folds (``explain_results``) and the paired tests (``explain_tests``).
"""

from __future__ import annotations

import collections
import json
from typing import Any

from grantrisk.labelling.scoring import FACTORS
from grantrisk.reporting.chain import Data
from grantrisk.reporting.common import csv_text, fmt, md_document, md_table, rounded

STAGES = ["E4", "E2", "M2", "M1", "L3"]
SCORE = "normalised_score"
NOTE = ("Exploratory (DEC-64): specified before the corpus results and reported in full; these analyses do not decide "
        "H1 of DEC-63. Differences: mean over the paired folds with the 95% interval of the corrected resampled t-test; "
        "p (Holm): adjusted within the analysis.")


def _ran(data: Data, analysis: str) -> bool:
    return bool(data.chain.e4) and analysis in ((data.e4_report or {}).get("settings") or {}).get("analyses", [])


def _reps(data: Data) -> list[str]:
    return list((data.e4_report or {}).get("representations") or [])


def _means(data: Data, analysis: str) -> dict[tuple[str, str, str, str, str], Any]:
    return {(r["representation"], r["classifier"], r["target"], r["setting"], r["metric"]): r
            for r in data.e4_results if r["analysis"] == analysis}


def _tests(data: Data, analysis: str) -> list[Any]:
    return [t for t in data.e4_tests if t["analysis"] == analysis]


def _diff(t) -> str:
    return f"{t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]"


def _results_csv(data: Data, analysis: str) -> str:
    return csv_text(["analysis", "representation", "classifier", "target", "setting", "metric", "mean", "sd", "n"],
                    [[r["analysis"], r["representation"], r["classifier"], r["target"], r["setting"], r["metric"],
                      rounded(r["mean"]), rounded(r["sd"]), r["n"]] for r in data.e4_results if r["analysis"] == analysis])


def _doc(data: Data, title: str, body: list[str], today: str) -> str:
    return md_document(title, [NOTE, "", *body], data, STAGES, today)


def factor_probe_tables(data: Data, today: str) -> tuple[str | None, str]:
    """DEC-64 no. 1: which factor points, and the normalised score, each representation predicts."""
    title = "Factor probes"
    if not _ran(data, "factor_probe"):
        return None, md_document(title, ["Not run: " + ("no E4 run in this chain." if not data.chain.e4 else
                                                         "the E4 run did not run `factor_probe`.")], data, STAGES, today)
    reps, means = _reps(data), _means(data, "factor_probe")
    origins = ", ".join(((data.e4_report or {}).get("settings") or {}).get("probe_origins", []))

    def table(metric: str) -> list[list[str]]:
        rows = []
        for target in (*FACTORS, SCORE):
            m, clf = ("spearman", "ridge") if target == SCORE else (metric, "logreg")
            if target == SCORE and metric != "f1_macro":
                continue
            if not any((r, clf, target, "", m) in means for r in reps):
                continue
            base = means.get(("majority", "majority", target, "", m))
            rows.append([target + (" (Spearman)" if target == SCORE else ""), fmt(base["mean"]) if base else "–",
                         *[fmt(means[(r, clf, target, "", m)]["mean"]) if (r, clf, target, "", m) in means else "–"
                           for r in reps]])
        return rows

    tests = _tests(data, "factor_probe")
    body = [f"Logistic regression predicts each factor's points (5 × 5 CV stratified by the points, over the documents "
            f"whose points come from {origins}). The normalised score: ridge regression on the folds of M2, Spearman's "
            "correlation. Mean over the folds.", "", "## Macro-F1", "",
            *md_table(["Target", "Majority", *reps], table("f1_macro")), "", "## Balanced accuracy", "",
            *md_table(["Target", "Majority", *reps], table("balanced_accuracy")), "", "## Each embedding minus TF-IDF",
            "", *md_table(["Target", "Representation", "Difference [95% interval]", "p", "p (Holm)"],
                          [[t["target"], t["representation"], _diff(t), fmt(t["p_value"], 4), fmt(t["p_holm"], 4)]
                           for t in sorted(tests, key=lambda t: (t["target"] == SCORE, t["target"], t["representation"]))])]
    notes = ((data.e4_report or {}).get("notes") or {}).get("factor_probe")
    if notes:
        body += ["", "Notes: " + "; ".join(notes) + "."]
    return _results_csv(data, "factor_probe"), _doc(data, title, body, today)


def combination_tables(data: Data, today: str) -> tuple[str | None, str]:
    """DEC-64 no. 2: TF-IDF beside each embedding, against TF-IDF alone on the same folds."""
    title = "TF-IDF combined with each embedding"
    if not _ran(data, "combination"):
        return None, md_document(title, ["Not run: " + ("no E4 run in this chain." if not data.chain.e4 else
                                                         "the E4 run did not run `combination`.")], data, STAGES, today)
    means = _means(data, "combination")
    clfs = sorted({k[1] for k in means})
    reps = sorted({k[0] for k in means}, key=lambda r: (r != "tfidf", r))
    def cell(rep: str, clf: str) -> str:
        row = means.get((rep, clf, "tercile_label", "", "f1_macro"))
        return fmt(row["mean"]) if row else "–"

    body = ["The TF-IDF row beside the standardised embedding scaled by 1/√d (DEC-69), fitted in each training fold of "
            "M2 with the defaults of SPEC-M2-04; macro-F1, mean over the folds.", "",
            *md_table(["Features", *clfs], [[r, *[cell(r, c) for c in clfs]] for r in reps]),
            "", "## Each combination minus TF-IDF alone", "",
            *md_table(["Combination", "Classifier", "Difference [95% interval]", "p", "p (Holm)"],
                      [[t["representation"], t["classifier"], _diff(t), fmt(t["p_value"], 4), fmt(t["p_holm"], 4)]
                       for t in _tests(data, "combination")])]
    return _results_csv(data, "combination"), _doc(data, title, body, today)


def learning_curve_tables(data: Data, today: str) -> tuple[str | None, str]:
    """DEC-64 no. 3: logreg on shares of each training fold."""
    title = "Learning curve"
    if not _ran(data, "learning_curve"):
        return None, md_document(title, ["Not run: " + ("no E4 run in this chain." if not data.chain.e4 else
                                                         "the E4 run did not run `learning_curve`.")], data, STAGES, today)
    means = _means(data, "learning_curve")
    shares = sorted({k[3] for k in means}, key=float)
    body = ["logreg on a stratified share of each training fold of M2, tested on the whole test fold; macro-F1, mean "
            "over the folds.", "",
            *md_table(["Representation", *shares],
                      [[r, *[fmt(means[(r, "logreg", "tercile_label", s, "f1_macro")]["mean"])
                             if (r, "logreg", "tercile_label", s, "f1_macro") in means else "–" for s in shares]]
                       for r in _reps(data)]),
            "", "## Each embedding minus TF-IDF, per share", "",
            *md_table(["Share", "Representation", "Difference [95% interval]", "p", "p (Holm)"],
                      [[t["setting"], t["representation"], _diff(t), fmt(t["p_value"], 4), fmt(t["p_holm"], 4)]
                       for t in _tests(data, "learning_curve")])]
    return _results_csv(data, "learning_curve"), _doc(data, title, body, today)


def error_overlap_tables(data: Data, today: str) -> tuple[str | None, str]:
    """DEC-64 no. 4: whether TF-IDF and an embedding misclassify the same documents."""
    title = "Error overlap with TF-IDF"
    if not _ran(data, "error_overlap"):
        return None, md_document(title, ["Not run: " + ("no E4 run in this chain." if not data.chain.e4 else
                                                         "the E4 run did not run `error_overlap`.")], data, STAGES, today)
    md_rows, csv_rows = [], []
    for t in _tests(data, "error_overlap"):
        d = json.loads(t["details_json"])
        mo = d["modal"]
        overlaps = [x["overlap"] for x in d["per_repeat"] if x["overlap"] is not None]
        overlap = sum(overlaps) / len(overlaps) if overlaps else None
        md_rows.append([t["representation"], t["classifier"], str(mo["both_wrong"]), str(mo["only_embedding_wrong"]),
                        str(mo["only_tfidf_wrong"]), str(mo["both_right"]), fmt(overlap), fmt(t["p_value"], 4),
                        fmt(t["p_holm"], 4)])
        csv_rows.append([t["representation"], t["classifier"], t["n"], mo["both_wrong"], mo["only_embedding_wrong"],
                         mo["only_tfidf_wrong"], mo["both_right"], rounded(overlap), rounded(t["p_value"], 6),
                         rounded(t["p_holm"], 6)])
    body = ["The modal prediction of each document over the repeats (a tie goes to the higher risk), against the tercile "
            "label. Overlap: both wrong / either wrong, mean over the repeats. McNemar's exact test on the modal "
            "predictions.", "",
            *md_table(["Representation", "Classifier", "Both wrong", "Only embedding wrong", "Only TF-IDF wrong",
                       "Both right", "Overlap", "p", "p (Holm)"], md_rows)]
    csv = csv_text(["representation", "classifier", "documents", "both_wrong", "only_embedding_wrong",
                    "only_tfidf_wrong", "both_right", "overlap_mean", "p_value", "p_holm"], csv_rows)
    return csv, _doc(data, title, body, today)


def context_length_tables(data: Data, today: str) -> tuple[str | None, str]:
    """DEC-64 no. 5: bge-m3 with 512-token chunks against its 8,192-token chunks."""
    title = "Context length within bge-m3"
    if not _ran(data, "context_length"):
        return None, md_document(title, ["Not run: " + ("no E4 run in this chain." if not data.chain.e4 else
                                                         "the E4 run did not run `context_length`.")], data, STAGES, today)
    tests = _tests(data, "context_length")
    if not tests:
        notes = ((data.e4_report or {}).get("notes") or {}).get("context_length") or ["no comparison"]
        return None, _doc(data, title, ["Not compared: " + "; ".join(notes) + "."], today)
    body = ["`bge_m3_512` (512-token chunks) minus `bge_m3` (8,192-token chunks), macro-F1 on the folds of M2 (from E2).",
            "", *md_table(["Classifier", "Difference [95% interval]", "p", "p (Holm)"],
                          [[t["classifier"], _diff(t), fmt(t["p_value"], 4), fmt(t["p_holm"], 4)] for t in tests])]
    csv = csv_text(["representation", "reference", "classifier", "n_folds", "mean_diff", "ci_low", "ci_high", "p_value",
                    "p_holm"], [[t["representation"], t["reference"], t["classifier"], t["n"], rounded(t["mean_diff"]),
                                 rounded(t["ci_low"]), rounded(t["ci_high"]), rounded(t["p_value"], 6),
                                 rounded(t["p_holm"], 6)] for t in tests])
    return csv, _doc(data, title, body, today)


def tests_csv(data: Data, today: str) -> tuple[str | None, str]:
    """Every paired test of the E4 run, for the thesis appendix."""
    title = "Tests of the explanatory analyses"
    if not data.chain.e4:
        return None, md_document(title, ["Not run: no E4 run in this chain."], data, STAGES, today)
    counts = collections.Counter(t["analysis"] for t in data.e4_tests)
    body = ["Every test of the E4 run is in `e4_tests.csv`: " + ", ".join(f"{a} {n}" for a, n in counts.items()) + "."]
    csv = csv_text(["analysis", "representation", "reference", "classifier", "target", "setting", "metric", "test", "n",
                    "mean_diff", "ci_low", "ci_high", "statistic", "p_value", "p_holm"],
                   [[t["analysis"], t["representation"], t["reference"], t["classifier"], t["target"], t["setting"],
                     t["metric"], t["test"], t["n"], rounded(t["mean_diff"]), rounded(t["ci_low"]), rounded(t["ci_high"]),
                     rounded(t["statistic"]), rounded(t["p_value"], 6), rounded(t["p_holm"], 6)] for t in data.e4_tests])
    return csv, _doc(data, title, body, today)


EXPLAIN_TABLES = {
    "e4_factor_probes": factor_probe_tables,
    "e4_combination": combination_tables,
    "e4_learning_curve": learning_curve_tables,
    "e4_error_overlap": error_overlap_tables,
    "e4_context_length": context_length_tables,
    "e4_tests": tests_csv,
}
