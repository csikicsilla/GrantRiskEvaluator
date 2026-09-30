"""The metric definitions and the E2 computation (SPEC-E2-01 … -09), without a database."""

import math
import random

import numpy as np
import pytest
from e2_helpers import LABELS, corpus, folds_of, majority, noisy
from sklearn import metrics as skm

from grantrisk.evaluation import evaluate, metrics
from grantrisk.evaluation.evaluate import Prediction

# A hand-computed example (SPEC-E2-01 acceptance). Rows of the confusion matrix: the true label.
#   low:    low 1, medium 1, high 0
#   medium: low 0, medium 2, high 0
#   high:   low 1, medium 0, high 1
TRUE = ["low", "low", "medium", "medium", "high", "high"]
PRED = ["low", "medium", "medium", "medium", "high", "low"]
PROBS = [
    [0.7, 0.2, 0.1],
    [0.4, 0.5, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.2, 0.7],
    [0.5, 0.1, 0.4],
]
SK_LABELS = sorted(LABELS)  # scikit-learn wants the multi-class labels in sorted order


def sk_probs(probs):
    return [[p[LABELS.index(c)] for c in SK_LABELS] for p in probs]


def test_hand_computed_example():
    v = metrics.classification(TRUE, PRED, PROBS).values
    assert [[v[f"confusion:{t}:{p}"] for p in LABELS] for t in LABELS] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert v["accuracy"] == pytest.approx(4 / 6)
    assert v["precision:low"] == pytest.approx(1 / 2)
    assert v["precision:medium"] == pytest.approx(2 / 3)
    assert v["precision:high"] == pytest.approx(1)
    assert v["recall:low"] == pytest.approx(1 / 2)
    assert v["recall:medium"] == pytest.approx(1)
    assert v["recall:high"] == pytest.approx(1 / 2)
    assert v["f1:low"] == pytest.approx(1 / 2)
    assert v["f1:medium"] == pytest.approx(4 / 5)
    assert v["f1:high"] == pytest.approx(2 / 3)
    assert v["f1_macro"] == pytest.approx((1 / 2 + 4 / 5 + 2 / 3) / 3)
    assert v["sensitivity:high"] == v["recall:high"]
    assert v["specificity:low"] == pytest.approx(3 / 4)  # 4 documents outside low, 1 of them predicted low
    assert v["specificity:medium"] == pytest.approx(3 / 4)
    assert v["specificity:high"] == pytest.approx(1)
    # SPEC-E2-04: predicted minus true, low = 0, medium = 1, high = 2.
    assert [v[f"error:{metrics.signed(e)}"] for e in (-2, -1, 0, 1, 2)] == [1, 0, 4, 1, 0]
    assert v["error2_share"] == pytest.approx(1 / 6)
    # AUC of low: positives score 0.7, 0.4; negatives 0.1, 0.2, 0.1, 0.5 → 7 of 8 pairs ordered.
    assert v["roc_auc_ovr:low"] == pytest.approx(7 / 8)


def test_matches_scikit_learn_on_the_example():
    v = metrics.classification(TRUE, PRED, PROBS).values
    p, r, f, _ = skm.precision_recall_fscore_support(TRUE, PRED, labels=list(LABELS), zero_division=0)
    for i, c in enumerate(LABELS):
        assert v[f"precision:{c}"] == pytest.approx(p[i])
        assert v[f"recall:{c}"] == pytest.approx(r[i])
        assert v[f"f1:{c}"] == pytest.approx(f[i])
    assert v["kappa_quadratic"] == pytest.approx(
        skm.cohen_kappa_score(TRUE, PRED, labels=list(LABELS), weights="quadratic"))
    assert v["roc_auc_ovr_macro"] == pytest.approx(
        skm.roc_auc_score(TRUE, sk_probs(PROBS), multi_class="ovr", average="macro", labels=SK_LABELS))
    assert v["roc_auc_ovo_macro"] == pytest.approx(
        skm.roc_auc_score(TRUE, sk_probs(PROBS), multi_class="ovo", average="macro", labels=SK_LABELS))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_scikit_learn_on_random_predictions(seed):
    """Includes tied probabilities, which the rank-based AUC counts as one half."""
    rng = random.Random(seed)
    n = 300
    true = [rng.choice(LABELS) for _ in range(n)]
    probs = []
    for _ in range(n):
        raw = [rng.choice([1, 2, 3, 4]) for _ in LABELS]
        probs.append([x / sum(raw) for x in raw])
    pred = [rng.choice(LABELS) for _ in range(n)]
    v = metrics.classification(true, pred, probs).values
    assert v["accuracy"] == pytest.approx(skm.accuracy_score(true, pred))
    assert v["f1_macro"] == pytest.approx(skm.f1_score(true, pred, labels=list(LABELS), average="macro"))
    assert v["precision_macro"] == pytest.approx(
        skm.precision_score(true, pred, labels=list(LABELS), average="macro", zero_division=0))
    assert v["recall_macro"] == pytest.approx(skm.recall_score(true, pred, labels=list(LABELS), average="macro"))
    matrix = skm.confusion_matrix(true, pred, labels=list(LABELS))
    assert [[v[f"confusion:{t}:{p}"] for p in LABELS] for t in LABELS] == matrix.tolist()
    for j, c in enumerate(LABELS):
        assert v[f"roc_auc_ovr:{c}"] == pytest.approx(skm.roc_auc_score([t == c for t in true], [p[j] for p in probs]))
    assert v["roc_auc_ovo_macro"] == pytest.approx(
        skm.roc_auc_score(true, sk_probs(probs), multi_class="ovo", labels=SK_LABELS))
    assert v["kappa_quadratic"] == pytest.approx(
        skm.cohen_kappa_score(true, pred, labels=list(LABELS), weights="quadratic"))


def test_class_never_predicted_has_precision_zero():
    """§4: precision 0 without a division error; recall and F1 as usual."""
    true = ["low", "medium", "high", "high"]
    pred = ["low", "low", "high", "high"]
    s = metrics.classification(true, pred)
    assert s.values["precision:medium"] == 0.0
    assert s.values["recall:medium"] == 0.0
    assert s.values["f1:medium"] == 0.0
    assert "never predicted" in s.flags["precision:medium"]
    assert "roc_auc_ovr_macro" not in s.values  # no probabilities, no AUC


def test_auc_is_undefined_when_a_class_is_missing():
    """§4: left empty and flagged."""
    s = metrics.classification(["low", "high"], ["low", "high"], [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    assert s.values["roc_auc_ovr:medium"] is None
    assert s.values["roc_auc_ovr_macro"] is None
    assert s.values["roc_auc_ovo_macro"] is None
    assert "roc_auc_ovr:medium" in s.flags and "roc_auc_ovr_macro" in s.flags
    assert s.values["roc_auc_ovr:low"] == 1.0


def test_kappa_edge_cases():
    assert metrics.classification(["low"] * 3, ["low"] * 3).values["kappa_quadratic"] is None
    perfect = metrics.classification(list(LABELS), list(LABELS)).values["kappa_quadratic"]
    assert perfect == pytest.approx(1.0)
    reversed_ = metrics.classification(["low", "high"], ["high", "low"]).values["kappa_quadratic"]
    assert reversed_ == pytest.approx(
        skm.cohen_kappa_score(["low", "high"], ["high", "low"], labels=list(LABELS), weights="quadratic"))


def test_roc_vertices_match_scikit_learn():
    rng = random.Random(5)
    positive = [rng.random() < 0.4 for _ in range(200)]
    scores = [round(rng.random(), 2) for _ in range(200)]
    fpr, tpr, _ = skm.roc_curve(positive, scores, drop_intermediate=False)
    ours = metrics.roc_vertices(positive, scores)
    assert np.allclose([p[0] for p in ours], fpr) and np.allclose([p[1] for p in ours], tpr)


def test_roc_grid_keeps_the_area():
    """SPEC-E2-03: the stored curve starts at 0 or above, ends at 1, never falls, and keeps the AUC."""
    rng = random.Random(6)
    positive = [rng.random() < 0.3 for _ in range(500)]
    scores = [rng.random() + 0.3 * p for p in positive]
    grid = metrics.on_grid(metrics.roc_vertices(positive, scores))
    assert len(grid) == metrics.ROC_GRID and grid[-1] == 1.0
    assert all(b >= a for a, b in zip(grid, grid[1:]))
    area = np.trapezoid(grid, np.linspace(0, 1, metrics.ROC_GRID))
    assert area == pytest.approx(metrics.auc(positive, scores), abs=0.01)


def test_roc_grid_takes_the_top_of_a_vertical_rise():
    vertices = [(0.0, 0.0), (0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (1.0, 1.0)]
    grid = metrics.on_grid(vertices, n=5)  # FPR 0, .25, .5, .75, 1
    assert grid == [0.5, 0.5, 1.0, 1.0, 1.0]


def test_corrected_t_test():
    """SPEC-E2-09 acceptance: a run compared with itself has difference 0 and p = 1."""
    same = metrics.corrected_t_test([0.0] * 25, 0.25)
    assert same["mean_diff"] == 0 and same["p_value"] == 1.0
    from scipy import stats

    diffs = [0.02, 0.05, -0.01, 0.03, 0.04, 0.0, 0.02, 0.01, 0.03, 0.02]
    uncorrected = metrics.corrected_t_test(diffs, 0.0)  # without the correction: the ordinary paired t-test
    assert uncorrected["p_value"] == pytest.approx(stats.ttest_1samp(diffs, 0).pvalue)
    corrected = metrics.corrected_t_test(diffs, 0.25)
    j, var = len(diffs), np.var(diffs, ddof=1)
    assert corrected["t"] == pytest.approx(np.mean(diffs) / math.sqrt((1 / j + 0.25) * var))
    assert corrected["p_value"] > uncorrected["p_value"]  # the correction widens the variance


def test_mean_sd_uses_n_minus_1_and_skips_undefined():
    assert metrics.mean_sd([1.0, 2.0, 3.0, None]) == (2.0, 1.0, 3)
    assert metrics.mean_sd([None]) == (None, None, 0)
    assert metrics.mean_sd([4.0]) == (4.0, None, 1)


# --- The computation over model runs ---------------------------------------------------------


DOCS = corpus(10)
PERIODS = {d: p for d, (_, _, p) in DOCS.items()}
LABEL_ROWS = [(t, f, p) for t, f, p in DOCS.values()]


def predictions_of(predict, scheme="stratified"):
    labels = {d: t for d, (t, _, _) in DOCS.items()}
    folds = folds_of(sorted(DOCS), labels)
    out = []
    for d in sorted(DOCS):
        for r in range(5):
            label, probs = predict(d, labels[d], r)
            out.append(Prediction(d, r, folds[(d, r)], labels[d], label, tuple(probs)))
    return out


def test_every_scope_of_a_model_run():
    """SPEC-E2-02, -03, -08 acceptance."""
    e = evaluate.evaluate_model(("stratified", "tfidf", "logreg"), predictions_of(noisy(0.8, 1)), PERIODS)
    folds = [s for s in e.scores if s.startswith("fold:")]
    repeats = [s for s in e.scores if s.startswith("repeat:")]
    assert len(folds) == 25 and len(repeats) == 5
    for name in ("f1_macro", "accuracy", "roc_auc_ovr_macro", "kappa_quadratic"):
        assert all(e.scores[s].values[name] is not None for s in folds + repeats)
        assert e.summary(f"{name}:fold_mean") == pytest.approx(np.mean([e.scores[s].values[name] for s in folds]))
        assert e.summary(f"{name}:fold_sd") == pytest.approx(np.std([e.scores[s].values[name] for s in folds], ddof=1))
        assert e.summary(f"{name}:repeat_sd") == pytest.approx(
            np.std([e.scores[s].values[name] for s in repeats], ddof=1))
    # SPEC-E2-03: the confusion matrix, counts divided by 5, sums to the number of documents.
    total = sum(e.scores[f"class:{c}"].values[f"confusion:{p}"] for c in LABELS for p in LABELS)
    assert total == pytest.approx(len(DOCS))
    for c in LABELS:
        shares = [e.scores[f"class:{c}"].values[f"confusion_share:{p}"] for p in LABELS]
        assert sum(shares) == pytest.approx(1)
    # SPEC-E2-08: the periods' document counts add up to the total.
    assert sum(e.scores[f"period:{p}"].values["documents"] for p in evaluate.PERIODS) == len(DOCS)
    assert set(e.roc) == {*LABELS, "macro"} and all(len(v) == metrics.ROC_GRID for v in e.roc.values())
    assert e.test_train_ratio == pytest.approx(0.25)


def test_error_table_known_predictions():
    """SPEC-E2-04 acceptance: every high document predicted low in every repeat, the rest right."""
    def predict(d, true, r):
        label = "low" if true == "high" else true
        probs = [0.0, 0.0, 0.0]
        probs[LABELS.index(label)] = 1.0
        return label, probs

    e = evaluate.evaluate_model(("stratified", "tfidf", "logreg"), predictions_of(predict), PERIODS)
    assert e.errors["-2"] == {"mean_count": 10.0, "sd_count": 0.0, "share": pytest.approx(1 / 3)}
    assert e.errors["0"]["mean_count"] == 20.0
    assert e.errors["+1"]["mean_count"] == 0.0
    assert e.summary("error2_share:repeat_mean") == pytest.approx(1 / 3)
    kappa = skm.cohen_kappa_score([t for t, _, _ in DOCS.values()],
                                  ["low" if t == "high" else t for t, _, _ in DOCS.values()],
                                  labels=list(LABELS), weights="quadratic")
    assert e.scores["pooled"].values["kappa_quadratic"] == pytest.approx(kappa)


def compute(models, **kw):
    return evaluate.compute({("stratified", r, c): predictions_of(p) for (r, c), p in models.items()},
                            PERIODS, LABEL_ROWS, **kw)


def test_comparison_marks_the_best_and_the_baseline():
    """SPEC-E2-05, -06 acceptance: one row per model run, the baseline in the table, one best."""
    result = compute({("tfidf", "logreg"): noisy(0.9, 1), ("tfidf", "svm"): noisy(0.02, 2),
                      ("tfidf", "majority"): majority(), ("e5", "logreg"): noisy(0.6, 3),
                      ("e5", "majority"): majority()})
    rows = result.comparison["stratified"]
    assert len(rows) == 5
    assert [r["f1_macro_mean"] for r in rows] == sorted((r["f1_macro_mean"] for r in rows), reverse=True)
    assert [(r["representation"], r["classifier"]) for r in rows if r["is_best"]] == [("tfidf", "logreg")]
    assert {(r["representation"], r["classifier"]) for r in rows if r["is_baseline"]} == {
        ("tfidf", "majority"), ("e5", "majority")}
    svm = next(r for r in rows if r["classifier"] == "svm")
    assert svm["not_above_baseline"] is True and svm["baseline"] == "tfidf/majority"
    assert next(r for r in rows if r["is_best"])["not_above_baseline"] is False
    assert all(r["not_above_baseline"] is None for r in rows if r["is_baseline"])
    assert rows[0]["rank"] == 1


def test_equal_means_are_both_best():
    """§4: two model runs with the same mean macro-F1 are both marked best."""
    result = compute({("tfidf", "logreg"): noisy(0.9, 1), ("e5", "logreg"): noisy(0.9, 1),
                      ("tfidf", "majority"): majority()})
    rows = result.comparison["stratified"]
    assert sum(r["is_best"] for r in rows) == 2
    assert [r["rank"] for r in rows] == [1, 1, 3]


def test_grid_means():
    """SPEC-E2-06: representation means leave out the baseline; classifier means run over representations."""
    result = compute({("tfidf", "logreg"): noisy(0.9, 1), ("tfidf", "svm"): noisy(0.7, 2),
                      ("tfidf", "majority"): majority(), ("e5", "logreg"): noisy(0.6, 3),
                      ("e5", "svm"): noisy(0.5, 4), ("e5", "majority"): majority()})
    rows = {(r["representation"], r["classifier"]): r["f1_macro_mean"] for r in result.comparison["stratified"]}
    grid = {(g["axis"], g["key"]): g for g in result.grid["stratified"]}
    assert grid[("representation", "tfidf")]["f1_macro_mean"] == pytest.approx(
        (rows[("tfidf", "logreg")] + rows[("tfidf", "svm")]) / 2)
    assert grid[("representation", "tfidf")]["n_models"] == 2
    assert grid[("classifier", "majority")]["f1_macro_mean"] == pytest.approx(rows[("tfidf", "majority")])
    assert grid[("classifier", "logreg")]["f1_macro_mean"] == pytest.approx(
        (rows[("tfidf", "logreg")] + rows[("e5", "logreg")]) / 2)


def test_significance_against_the_best():
    result = compute({("tfidf", "logreg"): noisy(0.9, 1), ("e5", "logreg"): noisy(0.5, 3),
                      ("tfidf", "majority"): majority()})
    tests = {(t["representation"], t["classifier"]): t for t in result.significance["stratified"]}
    assert set(tests) == {("e5", "logreg"), ("tfidf", "majority")}
    assert all(t["reference_representation"] == "tfidf" and t["n_folds"] == 25 for t in tests.values())
    assert tests[("e5", "logreg")]["mean_diff"] > 0 and tests[("e5", "logreg")]["p_value"] < 0.05
    assert all(t["p_value"] <= t["p_holm"] <= 1 for t in tests.values())
    assert compute({("tfidf", "logreg"): noisy(0.9, 1)}, with_significance=False).significance == {}


def test_holm_adjustment():
    """DEC-52: Holm's step-down method, by hand: 0.005·4, 0.01·3, 0.03·2, then kept monotone."""
    assert metrics.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert metrics.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])  # capped at 1
    assert metrics.holm([]) == []


def test_label_distributions_add_up():
    """SPEC-E2-07 acceptance."""
    dist = evaluate.label_distributions(LABEL_ROWS)
    assert sum(dist["all"].values()) == len(DOCS)
    assert sum(sum(dist[p].values()) for p in evaluate.PERIODS) == len(DOCS)
    tercile = {c: sum(n for (t, _), n in dist["all"].items() if t == c) for c in LABELS}
    assert tercile == {"low": 10, "medium": 10, "high": 10}
    assert len(dist["all"]) == 9  # the whole cross-table, zeros included


def test_grouped_scheme_is_kept_apart():
    """SPEC-E2-10: the schemes are never mixed in one row; the best is chosen in the standard scheme."""
    predictions = {("stratified", "tfidf", "logreg"): predictions_of(noisy(0.8, 1)),
                   ("stratified", "tfidf", "majority"): predictions_of(majority()),
                   ("grouped", "tfidf", "logreg"): predictions_of(noisy(0.95, 5))}
    result = evaluate.compute(predictions, PERIODS, LABEL_ROWS)
    assert set(result.comparison) == {"stratified", "grouped"}
    assert list(result.comparison)[0] == "stratified"
    assert [r["scheme"] for r in result.comparison["grouped"]] == ["grouped"]
    assert not any(r["is_best"] for r in result.comparison["grouped"])
    assert result.comparison["grouped"][0]["not_above_baseline"] is None  # no baseline in the grouped scheme
