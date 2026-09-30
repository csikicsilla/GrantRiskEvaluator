"""The metric definitions of E2 (Spec_E2_EvaluateModels.md SPEC-E2-01, -04, -09).

Plain functions over (true label, predicted label, class probabilities), written out so
that every number can be checked by hand; the tests compare them with scikit-learn.
Change METRICS_VERSION whenever a definition changes (SPEC-E2-01).
"""

from __future__ import annotations

import bisect
import collections
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from grantrisk.evaluation.validate import quadratic_kappa
from grantrisk.labelling.scoring import HIGH, LABELS, LOW, MEDIUM

METRICS_VERSION = "1 (Spec_E2_EvaluateModels.md, frozen 2026-09-29)"
RANK = {LOW: 0, MEDIUM: 1, HIGH: 2}
ERROR_SIZES = (-2, -1, 0, 1, 2)
ROC_GRID = 201  # the ROC curves are stored at FPR = 0, 0.005, …, 1


def signed(e: int) -> str:
    """An error size as written in metric names and tables: -2, -1, 0, +1, +2."""
    return f"{e:+d}" if e else "0"

DEFINITIONS = {
    "class_order": list(LABELS),
    "accuracy": "correct predictions / predictions",
    "precision:<c>": "TP / (TP + FP); 0 if the class is never predicted",
    "recall:<c>": "TP / (TP + FN); 0 if the class never occurs (scikit-learn's zero_division=0)",
    "f1:<c>": "2 TP / (2 TP + FP + FN); 0 if the class neither occurs nor is predicted",
    "sensitivity:<c>": "the same as recall:<c> (one-vs-rest)",
    "specificity:<c>": "TN / (TN + FP), one-vs-rest; empty if every document of the scope belongs to the class",
    "<m>_macro": "the unweighted mean of <m>:<c> over the classes",
    "roc_auc_ovr:<c>": "the area under the ROC curve of the class's probability, the class against the rest "
                       "(the Mann-Whitney statistic; ties count one half)",
    "roc_auc_ovr_macro": "the unweighted mean of roc_auc_ovr:<c>",
    "roc_auc_ovo_macro": "Hand and Till (2001): the mean over the class pairs of the two one-against-one AUCs",
    "kappa_quadratic": "Cohen's kappa with quadratic weights between the predicted and the tercile label",
    "error:<e>": "the number of predictions whose error (predicted minus tercile label, low = 0, medium = 1, "
                 "high = 2) is <e>, from -2 to +2 (INT-EVAL-15)",
    "error2_share": "the share of predictions with an error of size 2 (|error| = 2)",
    "confusion:<t>:<p>": "the number of predictions of true label <t> predicted as <p>",
    "support:<c>": "the number of predictions whose true label is <c>",
    "n_predictions": "the number of predictions in the scope",
}


@dataclass
class Scores:
    """The metrics of one scope: name → value (None where undefined), and name → flag."""

    values: dict[str, float | None] = field(default_factory=dict)
    flags: dict[str, str] = field(default_factory=dict)


def confusion(true: Sequence[str], pred: Sequence[str], labels: Sequence[str] = LABELS) -> list[list[int]]:
    """Rows: the true label; columns: the predicted label, both in ``labels`` order."""
    index = {c: i for i, c in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(true, pred, strict=True):
        matrix[index[t]][index[p]] += 1
    return matrix


def _ranks(scores: Sequence[float]) -> list[float]:
    """1-based ranks; tied scores share the mean of their ranks."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auc(positive: Sequence[bool], scores: Sequence[float]) -> float | None:
    """The ROC area of ``scores`` for the positives; None without positives or negatives."""
    n_pos = sum(positive)
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = sum(r for r, is_pos in zip(_ranks(scores), positive) if is_pos)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def roc_vertices(positive: Sequence[bool], scores: Sequence[float]) -> list[tuple[float, float]] | None:
    """The (FPR, TPR) points of the ROC curve, one per distinct score from the highest; None if undefined."""
    n_pos = sum(positive)
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    points = [(0.0, 0.0)]
    tp = fp = 0
    for n, i in enumerate(order):
        tp += positive[i]
        fp += not positive[i]
        if n + 1 == len(order) or scores[order[n + 1]] != scores[i]:
            points.append((fp / n_neg, tp / n_pos))
    return points


def on_grid(vertices: Sequence[tuple[float, float]], n: int = ROC_GRID) -> list[float]:
    """The TPR of the curve at FPR = 0, 1/(n-1), …, 1, linear between the vertices.

    Where the curve rises vertically at an FPR, the grid point takes the top of the rise.
    """
    xs = [v[0] for v in vertices]  # nondecreasing, and so is the TPR
    out = []
    for g in range(n):
        x = g / (n - 1)
        lo, hi = bisect.bisect_left(xs, x), bisect.bisect_right(xs, x)
        if hi > lo:
            out.append(vertices[hi - 1][1])
            continue
        # Between the top of the last rise before x and the bottom of the first rise after it.
        (x0, y0), (x1, y1) = vertices[lo - 1], vertices[lo]
        out.append(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return out


def roc_curves(true: Sequence[str], probs: Sequence[Sequence[float]]) -> dict[str, list[float] | None]:
    """SPEC-E2-03: the one-vs-rest curve of each class on the grid, and their pointwise mean ('macro')."""
    curves: dict[str, list[float] | None] = {}
    for j, c in enumerate(LABELS):
        vertices = roc_vertices([t == c for t in true], [p[j] for p in probs])
        curves[c] = None if vertices is None else on_grid(vertices)
    defined = [v for v in curves.values() if v is not None]
    curves["macro"] = [sum(col) / len(defined) for col in zip(*defined)] if len(defined) == len(LABELS) else None
    return curves


def _div(a: float, b: float) -> float | None:
    return None if b == 0 else a / b


def classification(
    true: Sequence[str],
    pred: Sequence[str],
    probs: Sequence[Sequence[float]] | None = None,
    labels: Sequence[str] = LABELS,
) -> Scores:
    """SPEC-E2-01 and -04 on one scope. ``labels`` are the classes the macro averages run over.

    ``probs`` holds the probabilities of (low, medium, high) for each prediction; without
    them, no ROC-AUC is computed.
    """
    s = Scores()
    v = s.values
    n = len(true)
    v["n_predictions"] = n
    matrix = confusion(true, pred)
    for i, t in enumerate(LABELS):
        for j, p in enumerate(LABELS):
            v[f"confusion:{t}:{p}"] = matrix[i][j]
    v["accuracy"] = _div(sum(matrix[i][i] for i in range(len(LABELS))), n)
    for i, c in enumerate(LABELS):
        if c not in labels:
            continue
        tp = matrix[i][i]
        fp = sum(matrix[k][i] for k in range(len(LABELS))) - tp
        fn = sum(matrix[i]) - tp
        tn = n - tp - fp - fn
        v[f"support:{c}"] = tp + fn
        v[f"precision:{c}"] = tp / (tp + fp) if tp + fp else 0.0
        v[f"recall:{c}"] = tp / (tp + fn) if tp + fn else 0.0
        v[f"f1:{c}"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        v[f"sensitivity:{c}"] = v[f"recall:{c}"]
        v[f"specificity:{c}"] = _div(tn, tn + fp)
        if tp + fp == 0:
            s.flags[f"precision:{c}"] = "the class is never predicted: precision 0"
        if tp + fn == 0:
            s.flags[f"recall:{c}"] = "the class does not occur: recall 0"
        if v[f"specificity:{c}"] is None:
            s.flags[f"specificity:{c}"] = "no document outside the class: undefined"
    for m in ("precision", "recall", "f1"):
        v[f"{m}_macro"] = statistics.fmean(v[f"{m}:{c}"] for c in labels) if labels else None

    if probs is not None:
        per_class = []
        for j, c in enumerate(LABELS):
            a = auc([t == c for t in true], [p[j] for p in probs])
            v[f"roc_auc_ovr:{c}"] = a
            if a is None:
                s.flags[f"roc_auc_ovr:{c}"] = "the class, or every other class, is missing from the scope: undefined"
            per_class.append(a)
        v["roc_auc_ovr_macro"] = None if None in per_class else statistics.fmean(per_class)
        pairs = []
        for a_i in range(len(LABELS)):
            for b_i in range(a_i + 1, len(LABELS)):
                a, b = LABELS[a_i], LABELS[b_i]
                idx = [k for k, t in enumerate(true) if t in (a, b)]
                auc_a = auc([true[k] == a for k in idx], [probs[k][a_i] for k in idx])
                auc_b = auc([true[k] == b for k in idx], [probs[k][b_i] for k in idx])
                pairs.append(None if auc_a is None or auc_b is None else (auc_a + auc_b) / 2)
        v["roc_auc_ovo_macro"] = None if None in pairs else statistics.fmean(pairs)
        for name in ("roc_auc_ovr_macro", "roc_auc_ovo_macro"):
            if v[name] is None:
                s.flags[name] = "a class is missing from the scope: undefined"

    kappa = quadratic_kappa([(RANK[t], RANK[p]) for t, p in zip(true, pred)])
    v["kappa_quadratic"] = None if kappa is None else float(kappa)
    if kappa is None:
        s.flags["kappa_quadratic"] = "no disagreement is expected by chance: undefined"
    errors = collections.Counter(RANK[p] - RANK[t] for t, p in zip(true, pred))
    for e in ERROR_SIZES:
        v[f"error:{signed(e)}"] = errors.get(e, 0)
    v["error2_share"] = _div(errors.get(-2, 0) + errors.get(2, 0), n)
    return s


def summary_names(labels: Sequence[str] = LABELS) -> list[str]:
    """The metrics whose mean and spread over folds and repeats are reported (SPEC-E2-02)."""
    names = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro", "roc_auc_ovo_macro",
             "kappa_quadratic", "error2_share"]
    for c in labels:
        names += [f"precision:{c}", f"recall:{c}", f"f1:{c}", f"sensitivity:{c}", f"specificity:{c}",
                  f"roc_auc_ovr:{c}"]
    return names


def mean_sd(values: Sequence[float | None]) -> tuple[float | None, float | None, int]:
    """The mean and the standard deviation (n - 1) of the defined values, and their number."""
    defined = [x for x in values if x is not None]
    if not defined:
        return None, None, 0
    mean = statistics.fmean(defined)
    return mean, statistics.stdev(defined) if len(defined) > 1 else None, len(defined)


def corrected_t_test(differences: Sequence[float], test_train_ratio: float) -> dict[str, float | int]:
    """SPEC-E2-09: the corrected resampled t-test (Nadeau and Bengio, 2003).

    ``differences`` are the paired per-fold differences of a metric over the r × k folds;
    ``test_train_ratio`` is n_test / n_train. The variance of the mean is corrected from
    1/J to 1/J + n_test/n_train, because the training sets overlap. Two-sided p-value.
    """
    from scipy import stats

    j = len(differences)
    mean = statistics.fmean(differences)
    var = statistics.variance(differences) if j > 1 else 0.0
    df = j - 1
    if var == 0:
        return {"mean_diff": mean, "sd_diff": 0.0, "t": 0.0 if mean == 0 else math.copysign(math.inf, mean),
                "df": df, "p_value": 1.0 if mean == 0 else 0.0}
    t = mean / math.sqrt((1 / j + test_train_ratio) * var)
    return {"mean_diff": mean, "sd_diff": math.sqrt(var), "t": t, "df": df, "p_value": float(2 * stats.t.sf(abs(t), df))}


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjustment of ``p_values`` for multiple comparisons, in the input order (DEC-52).

    The i-th smallest of m p-values is multiplied by m - i + 1; the adjusted values are then
    made monotone in that order and capped at 1.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def error_table(repeat_scores: Mapping[int, Scores]) -> dict[str, dict[str, float | None]]:
    """SPEC-E2-04: per error size, the number of predictions averaged over the repeats, and its spread."""
    out = {}
    for e in ERROR_SIZES:
        counts = [s.values[f"error:{signed(e)}"] for _, s in sorted(repeat_scores.items())]
        mean, sd, _ = mean_sd(counts)
        n = statistics.fmean(s.values["n_predictions"] for s in repeat_scores.values())
        out[signed(e)] = {"mean_count": mean, "sd_count": sd, "share": _div(mean, n)}
    return out
