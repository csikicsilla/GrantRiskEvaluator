"""The per-document risk dataset (SPEC-E3-02), the thesis tables (SPEC-E3-03) and the error analysis (SPEC-E3-04).

Each function returns the text of its files; ``report.run`` writes them. The numbers
come from stored runs; the error analysis only counts the stored predictions of the
best model run by group.
"""

from __future__ import annotations

import collections
import json
from typing import Any

from grantrisk.evaluation.metrics import ERROR_SIZES, signed
from grantrisk.labelling import scoring
from grantrisk.labelling.scoring import FACTORS, LABELS
from grantrisk.reporting.chain import Data, sort_sources
from grantrisk.reporting.common import (
    LABEL_RANK, PERIODS, csv_text, fmt, hu_number, hu_value, md_document, md_table, model_name, rounded,
)

NOT_RUN = "Not available: {what} is not part of this chain."
KIND_ORDER = ("low_coverage", "imputed_factors", "period", "programme")


# --- SPEC-E3-02: the per-document risk dataset ------------------------------------------------


def modal_label(predicted: list[str]) -> str:
    """The label predicted most often over the repeats; a tie goes to the higher-risk class (as DEC-32)."""
    counts = collections.Counter(predicted)
    return max(LABELS, key=lambda c: (counts.get(c, 0), LABEL_RANK[c]))


def _ranks(data: Data) -> dict[str, int]:
    """1 for the highest normalised score; equal scores share a rank."""
    ordered = sorted(data.labels.values(), key=lambda r: (-r["normalised_score"], r["doc_id"]))
    ranks, previous, rank = {}, None, 0
    for i, r in enumerate(ordered, start=1):
        if r["normalised_exact"] != previous:
            rank, previous = i, r["normalised_exact"]
        ranks[r["doc_id"]] = rank
    return ranks


def document_rows(data: Data) -> list[dict[str, str]]:
    """One row per document of the L3 run, ranked by normalised score; every value formatted once."""
    ranks = _ranks(data)
    rows = []
    for doc_id in sorted(data.labels, key=lambda d: (ranks[d], d)):
        lab, doc = data.labels[doc_id], data.documents[doc_id]
        row = {"rank": str(ranks[doc_id]), "doc_id": doc_id, "call_code": doc["call_code"],
               "programme": doc["programme"], "period": doc["period"], "doc_type": doc["doc_type"]}
        for f in FACTORS:
            cf = data.consolidated.get((doc_id, f))
            fp = data.factor_points[(doc_id, f)]
            row[f"{f}_value"] = "" if cf is None else hu_value(cf["value_json"])
            row[f"{f}_source"] = "" if cf is None or cf["chosen_source"] is None else cf["chosen_source"]
            row[f"{f}_points"] = hu_number(fp["points"])
            row[f"{f}_origin"] = fp["origin"]
        row.update({
            "n_determined": str(lab["n_determined"]), "low_coverage": str(lab["low_coverage"]),
            "total_score": hu_number(lab["total_score"]), "normalised_score": hu_number(lab["normalised_score"]),
            "fixed_label": lab["fixed_label"], "tercile_label": lab["tercile_label"],
        })
        predicted = data.best_predictions.get(doc_id)
        row["predicted_label"] = modal_label(predicted) if predicted else ""
        row["repeats_correct"] = str(sum(p == lab["tercile_label"] for p in predicted)) if predicted else ""
        row["repeats"] = str(len(predicted)) if predicted else ""
        rows.append(row)
    return rows


def risk_dataset(data: Data) -> str:
    """The CSV text; the caller writes it as UTF-8 with BOM, ';'-separated, for a Hungarian Excel."""
    rows = document_rows(data)
    header = list(rows[0]) if rows else []
    return csv_text(header, ([r[h] for h in header] for r in rows), delimiter=";")


# --- SPEC-E3-03: the thesis tables ------------------------------------------------------------


def _comparison_rows(data: Data) -> list[list[Any]]:
    out = []
    for r in data.comparison:
        out.append([r["scheme"], r["rank"], r["representation"], r["classifier"], rounded(r["f1_macro_mean"]),
                     rounded(r["f1_macro_sd"]), rounded(r["accuracy_mean"]), rounded(r["roc_auc_ovr_macro_mean"]),
                     rounded(r["kappa_quadratic_mean"]), rounded(r["error2_share"]), r["is_baseline"], r["is_best"],
                     r["not_above_baseline"]])
    return out


COMPARISON_HEADER = ["scheme", "rank", "representation", "classifier", "f1_macro_mean", "f1_macro_sd",
                     "accuracy_mean", "roc_auc_ovr_macro_mean", "kappa_quadratic_mean", "error2_share",
                     "is_baseline", "is_best", "not_above_baseline"]

MODEL_DEFINITIONS = (
    "macro-F1, accuracy, ROC-AUC (one-vs-rest, macro average) and κw (Cohen's kappa with quadratic weights) are "
    "the means over the 25 test folds of the repeated stratified 5 × 5 cross-validation; ± is the standard "
    "deviation over the folds (n − 1). The reference is the tercile label (DEC-07). |e| = 2 is the share of "
    "documents whose predicted label is two classes away from the tercile label, averaged over the repeats. "
    "The majority-class baseline predicts the most frequent class of each training fold (DEC-25)."
)


def model_comparison(data: Data, today: str) -> tuple[str | None, str]:
    """SPEC-E2-06 via SPEC-E3-03: the comparison table, per scheme."""
    stages = ["E2", "M2", "L3"]
    if not data.chain.e2:
        return None, md_document("Model comparison", [NOT_RUN.format(what="E2")], data, stages, today)
    body = [MODEL_DEFINITIONS, ""]
    for scheme in dict.fromkeys(r["scheme"] for r in data.comparison):
        title = "Standard scheme (stratified)" if scheme == "stratified" else f"Robustness check: `{scheme}` scheme"
        body += [f"## {title}", ""]
        rows = []
        for r in (r for r in data.comparison if r["scheme"] == scheme):
            note = "**best**" if r["is_best"] else "baseline" if r["is_baseline"] else (
                "not above the baseline" if r["not_above_baseline"] else "")
            rows.append([str(r["rank"]), r["representation"], r["classifier"],
                         f"{fmt(r['f1_macro_mean'])} ± {fmt(r['f1_macro_sd'])}", fmt(r["accuracy_mean"]),
                         fmt(r["roc_auc_ovr_macro_mean"]), fmt(r["kappa_quadratic_mean"]), fmt(r["error2_share"]), note])
        body += md_table(["Rank", "Representation", "Classifier", "macro-F1", "Accuracy", "ROC-AUC", "κw",
                          "abs(e) = 2", "Note"], rows)
        body.append("")
    if data.tied_best:
        body.append("Tied best model runs: " + ", ".join(f"`{model_name(k)}`" for k in [data.best, *data.tied_best]) + ".")
    csv = csv_text(COMPARISON_HEADER, _comparison_rows(data))
    return csv, md_document("Model comparison", body, data, stages, today)


def model_grid(data: Data, today: str) -> tuple[str | None, str]:
    """SPEC-E2-06: macro-F1 as representations × classifiers, with the row and column means."""
    stages = ["E2", "M2"]
    if not data.chain.e2:
        return None, md_document("Model grid", [NOT_RUN.format(what="E2")], data, stages, today)
    rows = [r for r in data.comparison if r["scheme"] == "stratified"]
    reps = list(dict.fromkeys(g["key"] for g in data.grid if g["axis"] == "representation" and g["scheme"] == "stratified"))
    clfs = list(dict.fromkeys(g["key"] for g in data.grid if g["axis"] == "classifier" and g["scheme"] == "stratified"))
    cell = {(r["representation"], r["classifier"]): r["f1_macro_mean"] for r in rows}
    mean = {(g["axis"], g["key"]): g["f1_macro_mean"] for g in data.grid if g["scheme"] == "stratified"}
    csv_rows = [[rep, *[rounded(cell.get((rep, c))) for c in clfs], rounded(mean.get(("representation", rep)))]
                for rep in reps]
    csv_rows.append(["classifier_mean", *[rounded(mean.get(("classifier", c))) for c in clfs], None])
    body = [
        "Mean macro-F1 over the 25 test folds (standard scheme). The row mean of a representation leaves out the "
        "majority baseline; the column mean of a classifier runs over the representations (INT-RQ-B1).",
        "",
        *md_table(["Representation", *clfs, "Row mean"],
                  [[rep, *[fmt(cell.get((rep, c))) for c in clfs], fmt(mean.get(("representation", rep)))]
                   for rep in reps] + [["**Column mean**", *[fmt(mean.get(("classifier", c))) for c in clfs], ""]]),
    ]
    return (csv_text(["representation", *clfs, "row_mean"], csv_rows),
            md_document("Model grid (macro-F1)", body, data, stages, today))


def label_distribution_tables(data: Data, today: str) -> tuple[str | None, str]:
    """SPEC-E2-07: the tercile and the fixed label, and their cross-table, overall and per period."""
    stages = ["E2", "L3", "C1"]
    if not data.chain.e2:
        return None, md_document("Label distributions", [NOT_RUN.format(what="E2")], data, stages, today)
    cells = {(r["subset"], r["tercile_label"], r["fixed_label"]): r["n"] for r in data.label_distributions}
    body = ["Rows: the tercile label (the ML target, DEC-07); columns: the fixed-threshold label (0–33 low, "
            "34–66 medium, 67–100 high; for documentation only). The margins are the counts per label.", ""]
    for subset in ("all", *PERIODS):
        n = sum(cells.get((subset, t, f), 0) for t in LABELS for f in LABELS)
        body += [f"## {'All documents' if subset == 'all' else 'Period ' + subset} ({n} documents)", ""]
        rows = [[t, *[str(cells.get((subset, t, f), 0)) for f in LABELS],
                 str(sum(cells.get((subset, t, f), 0) for f in LABELS))] for t in LABELS]
        rows.append(["**Total**", *[str(sum(cells.get((subset, t, f), 0) for t in LABELS)) for f in LABELS], str(n)])
        body += md_table(["Tercile \\ fixed", *LABELS, "Total"], rows) + [""]
    csv = csv_text(["subset", "tercile_label", "fixed_label", "n"],
                   [[r["subset"], r["tercile_label"], r["fixed_label"], r["n"]] for r in data.label_distributions])
    return csv, md_document("Label distributions", body, data, stages, today)


def error_size_tables(data: Data, today: str) -> tuple[str | None, str]:
    """SPEC-E2-04: documents per error size, averaged over the repeats, per model run."""
    stages = ["E2", "M2"]
    if not data.chain.e2:
        return None, md_document("Error sizes", [NOT_RUN.format(what="E2")], data, stages, today)
    sizes = ERROR_SIZES
    label = {e: signed(e) for e in sizes}
    csv_rows, md_rows = [], []
    for r in data.comparison:
        md = data.models[(r["scheme"], r["representation"], r["classifier"])]
        errs = md.errors
        csv_rows.append([r["scheme"], r["representation"], r["classifier"],
                         *[rounded(errs[e]["mean_count"]) for e in sizes], *[rounded(errs[e]["sd_count"]) for e in sizes],
                         rounded(r["error2_share"]), rounded(r["kappa_quadratic_mean"])])
        md_rows.append([model_name((r["scheme"], r["representation"], r["classifier"])),
                        *[f"{fmt(errs[e]['mean_count'], 1)} ± {fmt(errs[e]['sd_count'], 1)}" for e in sizes],
                        fmt(r["error2_share"]), fmt(r["kappa_quadratic_mean"])])
    body = ["Error = predicted label − tercile label on low = 0, medium = 1, high = 2 (INT-EVAL-15); e.g. a high "
            "document predicted medium is −1. Cells: documents per error size, the mean over the 5 repeats ± their "
            "standard deviation. |e| = 2: share of documents with an error of size 2. κw: quadratically weighted "
            "kappa, mean over the folds.", "",
            *md_table(["Model run", *[label[e] for e in sizes], "abs(e) = 2", "κw"], md_rows)]
    header = ["scheme", "representation", "classifier", *[f"mean_{label[e]}" for e in sizes],
              *[f"sd_{label[e]}" for e in sizes], "error2_share", "kappa_quadratic_mean"]
    return csv_text(header, csv_rows), md_document("Error sizes", body, data, stages, today)


def period_tables(data: Data, today: str) -> tuple[str | None, str]:
    """SPEC-E2-08: macro-F1 per period, pooled over the repeats."""
    stages = ["E2", "M2", "C1"]
    if not data.chain.e2:
        return None, md_document("Macro-F1 by period", [NOT_RUN.format(what="E2")], data, stages, today)
    csv_rows, md_rows = [], []
    for r in data.comparison:
        key = (r["scheme"], r["representation"], r["classifier"])
        p = data.models[key].periods
        csv_rows += [[*key, period, p[period]["documents"], rounded(p[period].get("f1_macro"))] for period in PERIODS]
        md_rows.append([model_name(key), *[f"{fmt(p[period].get('f1_macro'))} (n = {fmt(p[period]['documents'])})"
                                           for period in PERIODS]])
    body = ["macro-F1 of the out-of-fold predictions of all repeats together, per period, with the number of "
            "documents; the mean runs over the classes present in the period. Small periods carry no interval.", "",
            *md_table(["Model run", *PERIODS], md_rows)]
    return (csv_text(["scheme", "representation", "classifier", "period", "documents", "f1_macro"], csv_rows),
            md_document("Macro-F1 by period", body, data, stages, today))


def significance_tables(data: Data, today: str) -> tuple[str | None, str]:
    """SPEC-E2-09 (if run): the best model run against every other one."""
    stages = ["E2"]
    if not data.chain.e2 or not data.significance:
        return None, md_document("Significance", ["Not run."], data, stages, today)
    rows = data.significance
    body = ["Corrected resampled t-test (Nadeau and Bengio, 2003) on the 25 paired fold differences of macro-F1 "
            "(the best model run minus the other); two-sided. p: unadjusted. p (Holm): adjusted for the "
            f"{len(rows)} comparisons with Holm's step-down method (DEC-52); read this one when judging a difference.", "",
            *md_table(["Model run", "Reference", "Mean diff.", "SD diff.", "t", "df", "p", "p (Holm)"],
                      [[model_name((r["scheme"], r["representation"], r["classifier"])),
                        f"{r['reference_representation']}/{r['reference_classifier']}", fmt(r["mean_diff"]),
                        fmt(r["sd_diff"]), fmt(r["t_stat"], 2), str(r["df"]), fmt(r["p_value"], 4),
                        fmt(r["p_holm"], 4)] for r in rows])]
    csv = csv_text(["scheme", "representation", "classifier", "reference_representation", "reference_classifier",
                    "n_folds", "mean_diff", "sd_diff", "t_stat", "df", "p_value", "p_holm"],
                   [[r["scheme"], r["representation"], r["classifier"], r["reference_representation"],
                     r["reference_classifier"], r["n_folds"], rounded(r["mean_diff"]), rounded(r["sd_diff"]),
                     rounded(r["t_stat"]), r["df"], rounded(r["p_value"], 6), rounded(r["p_holm"], 6)] for r in rows])
    return csv, md_document("Significance", body, data, stages, today)


def gold_validation(data: Data, today: str) -> tuple[str | None, str]:
    """E1 via SPEC-E3-03 (INT-OUT-05): the agreement with the gold points per factor and source."""
    stages = ["E1", "L1"]
    if not data.chain.e1:
        return None, md_document("Gold validation", [NOT_RUN.format(what="E1")], data, stages, today)
    by = {(r["source"], r["factor"]): r for r in data.e1_agreements}
    sources = sort_sources([r["source"] for r in data.e1_agreements])

    def cell(r) -> str:
        if r is None or r["agreement"] is None:
            return "–"
        n = r["agree"] + r["disagree"] + r["not_found"]
        return f"{fmt(r['agreement'])} ({r['agree']}/{n}) [{fmt(r['agreement_low'], 2)}, {fmt(r['agreement_high'], 2)}]"

    body = ["Agreement with the expert's points on the gold documents: agree / (agree + disagree + not found); "
            "a value the source did not find counts against it. Brackets: 95% Wilson interval (SPEC-E1-05). "
            "The gold points come from one expert, and the extractors were developed on these documents, so the "
            "agreement is optimistic (see the E1 report).", "",
            *md_table(["Factor", *[f"`{s}`" for s in sources]],
                      [[f if f != "all" else "**all factors**", *[cell(by.get((s, f))) for s in sources]]
                       for f in (*FACTORS, "all")])]
    csv = csv_text(["source", "factor", "agree", "disagree", "not_found", "not_comparable", "agreement",
                    "agreement_low", "agreement_high", "coverage", "mean_abs_diff"],
                   [[r["source"], r["factor"], r["agree"], r["disagree"], r["not_found"], r["not_comparable"],
                     rounded(r["agreement"]), rounded(r["agreement_low"]), rounded(r["agreement_high"]),
                     rounded(r["coverage"]), rounded(r["mean_abs_diff"])] for r in data.e1_agreements])
    return csv, md_document("Gold validation per factor and source", body, data, stages, today)


def gold_label_agreement(data: Data, today: str) -> tuple[str | None, str]:
    """SPEC-E1-04 via SPEC-E3-03: the label-level agreement of each source with the expert's points."""
    stages = ["E1"]
    if not data.chain.e1:
        return None, md_document("Gold label agreement", [NOT_RUN.format(what="E1")], data, stages, today)
    order = sort_sources([r["source"] for r in data.e1_label_agreements])
    rows = sorted(data.e1_label_agreements, key=lambda r: (order.index(r["source"]), r["label_kind"]))
    md_rows, csv_rows = [], []
    for r in rows:
        errors = json.loads(r["error_sizes_json"])
        md_rows.append([f"`{r['source']}`", r["label_kind"],
                        f"{fmt(r['agreement'])} ({r['agree']}/{r['n_documents']}) "
                        f"[{fmt(r['agreement_low'], 2)}, {fmt(r['agreement_high'], 2)}]",
                        fmt(r["kappa_quadratic"]), *[str(errors[str(e)]) for e in range(-2, 3)],
                        fmt(r["mean_total_diff"], 2)])
        csv_rows.append([r["source"], r["label_kind"], r["n_documents"], r["agree"], rounded(r["agreement"]),
                         rounded(r["agreement_low"]), rounded(r["agreement_high"]), rounded(r["kappa_quadratic"]),
                         *[errors[str(e)] for e in range(-2, 3)], rounded(r["mean_total_diff"])])
    body = ["Each source's label on the gold documents, computed the way L3 would, against the label computed from "
            "the expert's points (SPEC-E1-04). Error: source label − reference label (low = 0, medium = 1, high = 2). "
            "Δ total: the mean difference of the total score (0–30).", "",
            *md_table(["Source", "Label", "Agreement", "κw", "−2", "−1", "0", "+1", "+2", "Δ total"], md_rows)]
    csv = csv_text(["source", "label_kind", "n_documents", "agree", "agreement", "agreement_low", "agreement_high",
                    "kappa_quadratic", "error_-2", "error_-1", "error_0", "error_+1", "error_+2", "mean_total_diff"],
                   csv_rows)
    return csv, md_document("Gold label agreement", body, data, stages, today)


THESIS_TABLES = {
    "model_comparison": model_comparison,
    "model_grid": model_grid,
    "label_distributions": label_distribution_tables,
    "error_sizes": error_size_tables,
    "period_breakdown": period_tables,
    "significance": significance_tables,
    "gold_validation": gold_validation,
    "gold_label_agreement": gold_label_agreement,
}


# --- SPEC-E3-04: the error analysis -----------------------------------------------------------


def disagreeing_factors(data: Data, doc_id: str) -> list[str] | None:
    """SPEC-L2-05 for one document: the factors where regex and the LLM both found a value with different points.

    None when the chain has no regex or no LLM run.
    """
    regex, llm = data.observations.get("regex"), data.observations.get("llm")
    if regex is None or llm is None:
        return None
    out = []
    for f in FACTORS:
        r, l = regex.get((doc_id, f)), llm.get((doc_id, f))
        if r is None or l is None or not r.found(f) or not l.found(f):
            continue
        try:
            if scoring.points(f, r.value)[0] != scoring.points(f, l.value)[0]:
                out.append(f)
        except ValueError:  # a value outside the domain gives no points to compare
            continue
    return out


def misclassified(data: Data) -> list[dict[str, str]]:
    """The documents the best model run misclassifies in more than half of the repeats (3 of 5)."""
    if data.best is None:
        return []
    by_doc = {r["doc_id"]: r for r in document_rows(data)}
    out = []
    for doc_id, row in by_doc.items():
        predicted = data.best_predictions[doc_id]
        wrong = sum(p != row["tercile_label"] for p in predicted)
        if wrong * 2 <= len(predicted):
            continue
        truth = LABEL_RANK[row["tercile_label"]]
        disagree = disagreeing_factors(data, doc_id)
        imputed = [f for f in FACTORS if row[f"{f}_origin"] == "mean"]
        out.append({
            **row,
            "repeats_wrong": str(wrong),
            "error_size": signed(LABEL_RANK[row["predicted_label"]] - truth),
            "errors_per_repeat": " ".join(signed(LABEL_RANK[p] - truth) for p in predicted),
            "imputed_factors": ", ".join(imputed),
            "regex_llm_disagree": "n/a" if disagree is None else ", ".join(disagree),
        })
    return out


def error_rates(data: Data) -> list[dict[str, Any]]:
    """The error rate of the best model run by coverage, imputed factors, period and programme."""
    if data.best is None:
        return []
    groups: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0, 0])  # docs, preds, wrong
    for doc_id, lab in data.labels.items():
        predicted = data.best_predictions[doc_id]
        wrong = sum(p != lab["tercile_label"] for p in predicted)
        imputed = sum(data.factor_points[(doc_id, f)]["origin"] == "mean" for f in FACTORS)
        doc = data.documents[doc_id]
        for kind, group in (("low_coverage", "yes" if lab["low_coverage"] else "no"),
                            ("imputed_factors", str(imputed)), ("period", doc["period"]),
                            ("programme", doc["programme"])):
            g = groups[(kind, group)]
            g[0] += 1
            g[1] += len(predicted)
            g[2] += wrong

    def order(key: tuple[str, str]) -> tuple:
        kind, group = key
        return (KIND_ORDER.index(kind), int(group) if kind == "imputed_factors" else
                PERIODS.index(group) if kind == "period" else 0, group)

    return [{"kind": k, "group": g, "documents": v[0], "predictions": v[1], "wrong": v[2],
             "error_rate": v[2] / v[1] if v[1] else None} for (k, g), v in sorted(groups.items(), key=lambda kv: order(kv[0]))]


MISCLASSIFIED_COLUMNS = ["doc_id", "call_code", "programme", "period", "tercile_label", "predicted_label",
                         "error_size", "repeats_wrong", "repeats", "errors_per_repeat", "n_determined", "low_coverage",
                         "normalised_score", "imputed_factors", "regex_llm_disagree",
                         *[f"{f}_{x}" for f in FACTORS for x in ("points", "origin")]]


def error_analysis(data: Data, today: str) -> dict[str, str]:
    """SPEC-E3-04: the Markdown report and its CSV files."""
    stages = ["E2", "M2", "E1", "L3", "L2", "L1", "C1"]
    body: list[str] = []
    files: dict[str, str] = {}
    if data.best is None:
        body.append(NOT_RUN.format(what="an evaluated model run (E2)"))
    else:
        others = f" Tied with {', '.join(f'`{model_name(k)}`' for k in data.tied_best)}; the first in the table's order " \
                 "is used." if data.tied_best else ""
        n = data.n_repeats
        body += [
            f"Best model run: `{model_name(data.best)}` (the highest mean macro-F1 over the folds).{others} Its "
            f"out-of-fold predictions of the {n} repeats are the material of this analysis.",
            "",
            "## 1. Misclassified documents",
            "",
            f"The documents misclassified in at least {n // 2 + 1} of the {n} repeats. Predicted: the label predicted "
            "most often (a tie goes to the higher-risk class). Error: predicted − tercile label (low = 0, medium = 1, "
            "high = 2), for the most frequent label and per repeat. Imputed: the factors filled with the factor mean "
            "(SPEC-L3-06). Regex ≠ LLM: the factors where both extractors found a value and their points differ "
            "(SPEC-L2-05). Every row is in `risk_dataset.csv` with the same values; all factor points are in "
            "`misclassified.csv`.",
            "",
        ]
        rows = misclassified(data)
        if rows:
            body += md_table(
                ["Call", "Period", "Tercile", "Predicted", "Error", "Per repeat", "Wrong", "Determined", "Low cov.",
                 "Imputed", "Regex ≠ LLM"],
                [[r["call_code"], r["period"], r["tercile_label"], r["predicted_label"], r["error_size"],
                  r["errors_per_repeat"], f"{r['repeats_wrong']}/{r['repeats']}", r["n_determined"],
                  "yes" if r["low_coverage"] == "1" else "no", r["imputed_factors"] or "–",
                  r["regex_llm_disagree"] or "–"] for r in rows])
        else:
            body.append("No document is misclassified in most repeats.")
        files["misclassified.csv"] = csv_text(MISCLASSIFIED_COLUMNS, ([r[c] for c in MISCLASSIFIED_COLUMNS] for r in rows),
                                              delimiter=";")
        rates = error_rates(data)
        body += ["", "## 2. Error rates by group", "",
                 "Error rate: wrong predictions / predictions, over all repeats (each document counts once per repeat). "
                 "Groups with few documents carry no interval and should not be over-read.", "",
                 *md_table(["Grouping", "Group", "Documents", "Predictions", "Wrong", "Error rate"],
                           [[r["kind"], r["group"], str(r["documents"]), str(r["predictions"]), str(r["wrong"]),
                             fmt(r["error_rate"])] for r in rates])]
        files["error_rates_by_group.csv"] = csv_text(
            ["kind", "group", "documents", "predictions", "wrong", "error_rate"],
            [[r["kind"], r["group"], r["documents"], r["predictions"], r["wrong"], rounded(r["error_rate"])] for r in rates])
    body += ["", "## 3. Propagation of extraction errors on the gold set", ""]
    if data.chain.e1 and data.e1_label_agreements:
        body += ["The label computed from each source's extracted values against the label computed from the expert's "
                 "points, on the gold documents (SPEC-E1-04). The share of disagreeing labels is the part of the "
                 "label error that comes from extraction; the ML model learns from these labels.", ""]
        body += md_table(["Source", "Label", "Agreement", "κw"],
                         [[f"`{r['source']}`", r["label_kind"],
                           f"{fmt(r['agreement'])} ({r['agree']}/{r['n_documents']})", fmt(r["kappa_quadratic"])]
                          for r in data.e1_label_agreements])
    else:
        body.append(NOT_RUN.format(what="E1"))
    body += ["", "## 4. Words that drive the model", ""]
    if data.top_terms:
        body += ["The terms with the largest mean weight per class in TF-IDF with logistic regression, averaged over "
                 "the 25 folds (a fold whose vocabulary lacks the term counts as 0; SPEC-M2-06).", ""]
        depth = max(len(v) for v in data.top_terms.values())
        body += md_table(["Rank", *LABELS], [
            [str(i + 1), *[f"{t[i]['term']} ({fmt(t[i]['mean_weight'])})" if i < len(t) else ""
                           for t in (data.top_terms.get(c, []) for c in LABELS)]] for i in range(depth)])
    else:
        body.append("Not available: the M2 run has no TF-IDF with logistic regression.")
    files["error_analysis.md"] = md_document("Error analysis", body, data, stages, today)
    return files
