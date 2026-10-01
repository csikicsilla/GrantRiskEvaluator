"""Stage E4: explain (DEC-64, DEC-69): exploratory analyses of the representations.

Why do the representations reach the macro-F1 they do (INT-RQ-C)? Five analyses, specified before
the corpus results and reported in full; none of them decides H1 of DEC-63:

1. ``factor_probe``: which factor points, and the normalised score, each representation predicts;
2. ``combination``: TF-IDF beside each embedding, against TF-IDF alone on the same folds;
3. ``learning_curve``: logreg on shares of each training fold;
4. ``error_overlap``: whether TF-IDF and an embedding misclassify the same documents (McNemar);
5. ``context_length``: bge-m3 with 512-token chunks against its 8,192-token chunks.

Where folds are paired, a difference comes with the corrected resampled t-test and its 95%
interval; Holm's method runs within each analysis. ``compute`` is the computation on loaded
inputs; ``run`` reads an E2 run and, through it, the M2, M1 and L3 runs.
"""

from __future__ import annotations

import collections
import csv
import io
import json
import math
import sqlite3
import statistics
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from grantrisk.evaluation import metrics as m
from grantrisk.evaluation.evaluate import BASELINE, STRATIFIED, TFIDF, Prediction, input_of, load_predictions
from grantrisk.labelling.scoring import FACTORS, LABELS
from grantrisk.modelling import classifiers as clf_mod
from grantrisk.modelling import represent
from grantrisk.modelling.tfidf import settings as tfidf_settings_of
from grantrisk.modelling.train import Split, load_labels, probabilities, predicted_label
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

ANALYSES = ("factor_probe", "combination", "learning_curve", "error_overlap", "context_length")
SCORE = "normalised_score"
TERCILE = "tercile_label"
CONTEXT_SHORT, CONTEXT_LONG = "bge_m3_512", "bge_m3"  # DEC-64 no. 5: the same model, 512 against 8,192 tokens
RANK = {label: i for i, label in enumerate(LABELS)}
NONE = ""  # an empty setting


class E4InputError(ValueError):
    """The input runs violate the E4 contract; nothing is written."""


@dataclass(frozen=True)
class Settings:
    analyses: tuple[str, ...] = ("factor_probe", "combination", "error_overlap", "context_length")
    probe_splits: int = 5
    probe_repeats: int = 5
    random_state: int = 42
    probe_origins: tuple[str, ...] = ("band", "manual")
    ridge_alpha: float = 1.0
    combination_classifiers: tuple[str, ...] = ("logreg", "svm")
    learning_curve_shares: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)

    @classmethod
    def of(cls, values: Mapping[str, Any] | None, analyses: Sequence[str] | None = None) -> Settings:
        v = dict(values or {})
        cv = v.get("probe_cv") or {}
        s = cls(
            analyses=tuple(analyses or v.get("analyses") or cls.analyses),
            probe_splits=int(cv.get("n_splits", cls.probe_splits)),
            probe_repeats=int(cv.get("n_repeats", cls.probe_repeats)),
            random_state=int(cv.get("random_state", cls.random_state)),
            probe_origins=tuple(v.get("probe_origins", cls.probe_origins)),
            ridge_alpha=float(v.get("ridge_alpha", cls.ridge_alpha)),
            combination_classifiers=tuple(v.get("combination_classifiers", cls.combination_classifiers)),
            learning_curve_shares=tuple(float(x) for x in v.get("learning_curve_shares", cls.learning_curve_shares)),
        )
        unknown = sorted(set(s.analyses) - set(ANALYSES))
        if unknown:
            raise ValueError(f"unknown E4 analyses {unknown}; known: {list(ANALYSES)}")
        if not all(0 < x <= 1 for x in s.learning_curve_shares):
            raise ValueError("explain.learning_curve_shares must lie in (0, 1]")
        return s


@dataclass
class Inputs:
    """What E4 reads, in the order of ``doc_ids``."""

    doc_ids: list[str]
    y: np.ndarray  # the tercile labels of L3
    splits: list[Split]  # the stratified fold assignment of M2
    features: dict[str, Any]  # representation → plain texts (tfidf) or a matrix
    points: dict[str, dict[str, float]]  # factor → doc_id → points whose origin is probed
    scores: np.ndarray  # the normalised score of L3
    predictions: dict[tuple[str, str, str], list[Prediction]] = field(default_factory=dict)  # M2, stratified
    fold_f1: dict[tuple[str, str], dict[tuple[int, int], float]] = field(default_factory=dict)  # E2: (rep, clf) → fold


@dataclass(frozen=True)
class FoldValue:
    analysis: str
    representation: str
    classifier: str
    target: str
    setting: str
    repeat: int
    fold: int
    metric: str
    value: float | None
    ratio: float  # n_test / n_train of the fold, for the corrected t-test


@dataclass
class E4Result:
    folds: list[FoldValue]
    summaries: list[dict[str, Any]]
    tests: list[dict[str, Any]]
    notes: dict[str, list[str]]


def _subset(X: Any, idx: Sequence[int] | np.ndarray) -> Any:
    return [X[i] for i in idx] if isinstance(X, list) else X[np.asarray(idx)]


def _quiet(fn: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence and small-class warnings; the run reports its numbers
        return fn()


def _class_scores(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    return _quiet(lambda: {"f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                           "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))})


def _labels_of(model: Any, X: Any) -> list[str]:
    """The label of SPEC-M2-04: the argmax of the probabilities, a tie going to the higher risk."""
    return [predicted_label(p) for p in probabilities(model, X)]


# --- 1. Factor probes ----------------------------------------------------------------------------


def point_class(points: float) -> str:
    return str(int(points)) if float(points).is_integer() else f"{points:g}"


def factor_probes(inputs: Inputs, s: Settings, params: Mapping[str, Any], tfidf: Mapping[str, Any],
                  progress: Callable[[str], None] | None = None) -> tuple[list[FoldValue], list[str]]:
    """DEC-64 no. 1: per representation and factor, logistic regression predicts the factor's points.

    5 × 5 cross-validation stratified by the points, over the documents whose points have a probed
    origin (band or manual); the majority class is the baseline. Macro-F1 and balanced accuracy.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold

    out, notes = [], []
    for factor in FACTORS:
        given = inputs.points.get(factor, {})
        pos = [i for i, d in enumerate(inputs.doc_ids) if d in given]
        yf = np.array([point_class(given[inputs.doc_ids[i]]) for i in pos])
        counts = collections.Counter(yf.tolist())
        if len(counts) < 2:
            notes.append(f"{factor}: {len(pos)} documents with {len(counts)} point value(s); not probed")
            continue
        small = sorted(c for c, n in counts.items() if n < s.probe_splits)
        if small:
            notes.append(f"{factor}: point value(s) {small} occur in fewer than {s.probe_splits} documents")
        splitter = RepeatedStratifiedKFold(n_splits=s.probe_splits, n_repeats=s.probe_repeats,
                                           random_state=s.random_state)
        splits = _quiet(lambda: [(i // s.probe_splits, i % s.probe_splits, tr, te)
                                 for i, (tr, te) in enumerate(splitter.split(np.zeros(len(yf)), yf))])
        for r, k, tr, te in splits:
            base = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(tr), 1)), yf[tr])
            for metric, v in _class_scores(yf[te], base.predict(np.zeros((len(te), 1)))).items():
                out.append(FoldValue("factor_probe", BASELINE, BASELINE, factor, NONE, r, k, metric, v, len(te) / len(tr)))
        for rep, X in inputs.features.items():
            Xf = _subset(X, pos)
            for r, k, tr, te in splits:
                pipe = clf_mod.build("logreg", params["logreg"], text=rep == TFIDF, tfidf_settings=tfidf,
                                     seed=s.random_state)
                _quiet(lambda: pipe.fit(_subset(Xf, tr), yf[tr]))
                for metric, v in _class_scores(yf[te], pipe.predict(_subset(Xf, te))).items():
                    out.append(FoldValue("factor_probe", rep, "logreg", factor, NONE, r, k, metric, v, len(te) / len(tr)))
            if progress:
                progress(f"factor_probe {factor} {rep}")
    return out, notes


# --- Work on the folds of M2: the score probe, the combination and the learning curve ----------


def _spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    from scipy import stats

    rho = _quiet(lambda: stats.spearmanr(a, b).statistic)
    return None if rho is None or (isinstance(rho, float) and math.isnan(rho)) else float(rho)


def _block(E_train: np.ndarray, E_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The standardised embedding scaled by 1/sqrt(d), so that its expected squared norm is 1, as that of
    an L2-normalised TF-IDF row (DEC-69)."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(E_train)
    w = 1.0 / math.sqrt(E_train.shape[1])
    return scaler.transform(E_train) * w, scaler.transform(E_test) * w


def m2_fold_work(inputs: Inputs, split: Split, s: Settings, params: Mapping[str, Any],
                 tfidf: Mapping[str, Any]) -> list[FoldValue]:
    """The analyses on one fold of M2 that share its TF-IDF fit: the score probe (no. 1) and the combination (no. 2)."""
    import scipy.sparse as sp
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from grantrisk.modelling.tfidf import vectorizer

    out = []
    tr, te = split.train, split.test
    ratio = len(te) / len(tr)
    T_tr = T_te = None
    if TFIDF in inputs.features:
        vec = vectorizer(tfidf)
        texts = inputs.features[TFIDF]
        T_tr = vec.fit_transform(_subset(texts, tr))
        T_te = vec.transform(_subset(texts, te))

    def value(analysis, rep, clf, target, metric, v):
        out.append(FoldValue(analysis, rep, clf, target, NONE, split.repeat, split.fold, metric, v, ratio))

    if "factor_probe" in s.analyses:  # the normalised score: ridge regression, scored by Spearman's correlation
        for rep, X in inputs.features.items():
            if rep == TFIDF:
                A_tr, A_te = T_tr, T_te
            else:
                scaler = StandardScaler().fit(X[tr])
                A_tr, A_te = scaler.transform(X[tr]), scaler.transform(X[te])
            model = Ridge(alpha=s.ridge_alpha).fit(A_tr, inputs.scores[tr])
            value("factor_probe", rep, "ridge", SCORE, "spearman", _spearman(inputs.scores[te], model.predict(A_te)))

    if "combination" in s.analyses and T_tr is not None:
        y_tr, y_te = inputs.y[tr], inputs.y[te]
        for clf in s.combination_classifiers:
            model = _quiet(lambda: clf_mod.estimator(clf, params[clf], s.random_state).fit(T_tr, y_tr))
            value("combination", TFIDF, clf, TERCILE, "f1_macro", _class_scores(y_te, _labels_of(model, T_te))["f1_macro"])
        for rep, X in inputs.features.items():
            if rep == TFIDF:
                continue
            E_tr, E_te = _block(X[tr], X[te])
            C_tr = sp.hstack([T_tr, sp.csr_matrix(E_tr)], format="csr")
            C_te = sp.hstack([T_te, sp.csr_matrix(E_te)], format="csr")
            for clf in s.combination_classifiers:
                model = _quiet(lambda: clf_mod.estimator(clf, params[clf], s.random_state).fit(C_tr, y_tr))
                value("combination", f"{TFIDF}+{rep}", clf, TERCILE, "f1_macro",
                      _class_scores(y_te, _labels_of(model, C_te))["f1_macro"])
    return out


def learning_curve_fold(inputs: Inputs, split: Split, s: Settings, params: Mapping[str, Any],
                        tfidf: Mapping[str, Any]) -> list[FoldValue]:
    """DEC-64 no. 3: logreg on a stratified share of the training fold, tested on the whole test fold."""
    from sklearn.model_selection import train_test_split

    out = []
    te = split.test
    for share in s.learning_curve_shares:
        tr = split.train if share >= 1 else train_test_split(
            split.train, train_size=share, stratify=inputs.y[split.train],
            random_state=s.random_state + 100 * split.repeat + split.fold)[0]
        for rep, X in inputs.features.items():
            pipe = clf_mod.build("logreg", params["logreg"], text=rep == TFIDF, tfidf_settings=tfidf,
                                 seed=s.random_state)
            _quiet(lambda: pipe.fit(_subset(X, tr), inputs.y[tr]))
            f1 = _class_scores(inputs.y[te], _labels_of(pipe, _subset(X, te)))["f1_macro"]
            out.append(FoldValue("learning_curve", rep, "logreg", TERCILE, f"{share:g}", split.repeat, split.fold,
                                 "f1_macro", f1, len(te) / len(tr)))
    return out


# --- 4. Error overlap and 5. context length: from the stored predictions and fold metrics -------


def modal_label(predicted: Sequence[str]) -> str:
    """The label predicted most often over the repeats; a tie goes to the higher-risk class (as DEC-32)."""
    counts = collections.Counter(predicted)
    return max(LABELS, key=lambda c: (counts.get(c, 0), RANK[c]))


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar's exact test: two-sided binomial p of the discordant pairs."""
    from scipy import stats

    n = b + c
    return 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def error_overlap(predictions: Mapping[tuple[str, str, str], Sequence[Prediction]]) -> list[dict[str, Any]]:
    """DEC-64 no. 4: for TF-IDF and each embedding with the same classifier, which documents each gets wrong.

    Per repeat, the four counts and the overlap (both wrong / either wrong); the test is McNemar's exact
    test on the modal prediction of each document over the repeats, so that each document counts once.
    """
    out = []
    keys = sorted(k for k in predictions if k[0] == STRATIFIED and k[1] != TFIDF and k[2] != BASELINE)
    for scheme, rep, clf in keys:
        ref = predictions.get((scheme, TFIDF, clf))
        if ref is None:
            continue
        mine = {(p.doc_id, p.repeat): p for p in predictions[(scheme, rep, clf)]}
        theirs = {(p.doc_id, p.repeat): p for p in ref}
        per_repeat = []
        for r in sorted({k[1] for k in mine}):
            docs = sorted(d for d, rr in mine if rr == r)
            right_e = [mine[(d, r)].predicted_label == mine[(d, r)].true_label for d in docs]
            right_t = [theirs[(d, r)].predicted_label == theirs[(d, r)].true_label for d in docs]
            n = collections.Counter(zip(right_t, right_e))
            wrong_any = n[(False, False)] + n[(True, False)] + n[(False, True)]
            per_repeat.append({"repeat": r, "both_right": n[(True, True)], "both_wrong": n[(False, False)],
                               "only_embedding_wrong": n[(True, False)], "only_tfidf_wrong": n[(False, True)],
                               "overlap": n[(False, False)] / wrong_any if wrong_any else None, "p_value": mcnemar_exact(
                                   n[(True, False)], n[(False, True)])})
        docs = sorted({d for d, _ in mine})
        true = {d: mine[(d, 0)].true_label for d in docs}
        modal_e = {d: modal_label([p.predicted_label for (dd, _), p in mine.items() if dd == d]) for d in docs}
        modal_t = {d: modal_label([p.predicted_label for (dd, _), p in theirs.items() if dd == d]) for d in docs}
        b = sum(modal_t[d] == true[d] and modal_e[d] != true[d] for d in docs)  # only the embedding wrong
        c = sum(modal_t[d] != true[d] and modal_e[d] == true[d] for d in docs)  # only TF-IDF wrong
        both = sum(modal_t[d] != true[d] and modal_e[d] != true[d] for d in docs)
        out.append({"representation": rep, "classifier": clf, "per_repeat": per_repeat, "documents": len(docs),
                    "modal": {"both_wrong": both, "only_embedding_wrong": b, "only_tfidf_wrong": c,
                              "both_right": len(docs) - both - b - c},
                    "p_value": mcnemar_exact(b, c)})
    return out


def context_length(fold_f1: Mapping[tuple[str, str], Mapping[tuple[int, int], float]], ratio: float) -> list[FoldValue]:
    """DEC-64 no. 5: the fold macro-F1 of bge-m3 with 512-token and with 8,192-token chunks, per classifier (from E2)."""
    out = []
    for (rep, clf), folds in sorted(fold_f1.items()):
        if rep in (CONTEXT_SHORT, CONTEXT_LONG) and clf != BASELINE and (
                (CONTEXT_LONG if rep == CONTEXT_SHORT else CONTEXT_SHORT), clf) in fold_f1:
            out += [FoldValue("context_length", rep, clf, TERCILE, NONE, r, k, "f1_macro", v, ratio)
                    for (r, k), v in sorted(folds.items())]
    return out


# --- Summaries and tests -----------------------------------------------------------------------


def summarise(folds: Sequence[FoldValue]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[float | None]] = collections.defaultdict(list)
    for f in folds:
        groups[(f.analysis, f.representation, f.classifier, f.target, f.setting, f.metric)].append(f.value)
    out = []
    for (analysis, rep, clf, target, setting, metric), values in sorted(groups.items()):
        mean, sd, n = m.mean_sd(values)
        out.append({"analysis": analysis, "representation": rep, "classifier": clf, "target": target,
                    "setting": setting, "metric": metric, "mean": mean, "sd": sd, "n": n})
    return out


def _paired(folds: Sequence[FoldValue], pairs: Sequence[tuple[tuple, tuple]]) -> list[dict[str, Any]]:
    """Corrected resampled t-tests of (representation minus reference) on the folds both have."""
    index: dict[tuple, dict[tuple[int, int], FoldValue]] = collections.defaultdict(dict)
    for f in folds:
        index[(f.analysis, f.representation, f.classifier, f.target, f.setting, f.metric)][(f.repeat, f.fold)] = f
    out = []
    for mine, ref in pairs:
        a, b = index.get(mine, {}), index.get(ref, {})
        common = sorted(k for k in set(a) & set(b) if a[k].value is not None and b[k].value is not None)
        if len(common) < 2:
            continue
        ratio = statistics.fmean(b[k].ratio for k in common)
        t = m.corrected_t_test([a[k].value - b[k].value for k in common], ratio)
        analysis, rep, clf, target, setting, metric = mine
        out.append({"analysis": analysis, "representation": rep, "reference": ref[1], "classifier": clf,
                    "target": target, "setting": setting, "metric": metric, "test": "corrected_t", "n": len(common),
                    "mean_diff": t["mean_diff"], "ci_low": t["ci_low"], "ci_high": t["ci_high"], "statistic": t["t"],
                    "p_value": t["p_value"], "details": {"sd_diff": t["sd_diff"], "df": t["df"],
                                                         "test_train_ratio": ratio}})
    return out


def tests(folds: Sequence[FoldValue], overlap: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The paired comparisons of every analysis, with Holm's adjustment within each analysis (DEC-64)."""
    keys = {(f.analysis, f.representation, f.classifier, f.target, f.setting, f.metric) for f in folds}
    pairs = []
    for key in sorted(keys):
        analysis, rep, clf, target, setting, metric = key
        if analysis == "factor_probe" and rep not in (TFIDF, BASELINE) and metric in ("f1_macro", "spearman"):
            pairs.append((key, (analysis, TFIDF, clf, target, setting, metric)))  # what an embedding carries
        elif analysis == "combination" and rep != TFIDF:
            pairs.append((key, (analysis, TFIDF, clf, target, setting, metric)))  # what it adds to TF-IDF
        elif analysis == "learning_curve" and rep != TFIDF:
            pairs.append((key, (analysis, TFIDF, clf, target, setting, metric)))
        elif analysis == "context_length" and rep == CONTEXT_SHORT:
            pairs.append((key, (analysis, CONTEXT_LONG, clf, target, setting, metric)))
    out = _paired(folds, pairs)
    for o in overlap:
        out.append({"analysis": "error_overlap", "representation": o["representation"], "reference": TFIDF,
                    "classifier": o["classifier"], "target": TERCILE, "setting": "modal over the repeats",
                    "metric": "errors", "test": "mcnemar_exact", "n": o["documents"], "mean_diff": None,
                    "ci_low": None, "ci_high": None,
                    "statistic": float(min(o["modal"]["only_embedding_wrong"], o["modal"]["only_tfidf_wrong"])),
                    "p_value": o["p_value"], "details": {"modal": o["modal"], "per_repeat": o["per_repeat"]}})
    for analysis in ANALYSES:
        group = [t for t in out if t["analysis"] == analysis]
        for t, p in zip(group, m.holm([t["p_value"] for t in group])):
            t["p_holm"] = p
    return out


def compute(inputs: Inputs, s: Settings, params: Mapping[str, Any], tfidf: Mapping[str, Any],
            progress: Callable[[str], None] | None = None) -> E4Result:
    """Every requested analysis on the loaded inputs."""
    folds: list[FoldValue] = []
    notes: dict[str, list[str]] = collections.defaultdict(list)
    if "factor_probe" in s.analyses:
        values, probe_notes = factor_probes(inputs, s, params, tfidf, progress)
        folds += values
        notes["factor_probe"] += probe_notes
    if {"factor_probe", "combination"} & set(s.analyses):
        if "combination" in s.analyses and TFIDF not in inputs.features:
            notes["combination"].append("the M2 run has no TF-IDF representation; nothing to combine")
        for split in inputs.splits:
            folds += m2_fold_work(inputs, split, s, params, tfidf)
            if progress:
                progress(f"score probe / combination: repeat {split.repeat} fold {split.fold}")
    if "learning_curve" in s.analyses:
        for split in inputs.splits:
            folds += learning_curve_fold(inputs, split, s, params, tfidf)
            if progress:
                progress(f"learning curve: repeat {split.repeat} fold {split.fold}")
    overlap = error_overlap(inputs.predictions) if "error_overlap" in s.analyses else []
    if "context_length" in s.analyses:
        ratio = statistics.fmean(len(sp.test) / len(sp.train) for sp in inputs.splits)
        values = context_length(inputs.fold_f1, ratio)
        if not values:
            notes["context_length"].append(f"the M2 run lacks {CONTEXT_SHORT} or {CONTEXT_LONG}; not compared")
        folds += values
    return E4Result(folds, summarise(folds), tests(folds, overlap), dict(notes))


# --- Report ------------------------------------------------------------------------------------


def _f(x: float | None, digits: int = 3, signed: bool = False) -> str:
    return "–" if x is None else (f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *("| " + " | ".join(r) + " |" for r in rows)]


def _test_rows(ts: Sequence[Mapping[str, Any]], first: Sequence[str]) -> list[list[str]]:
    return [[*(str(t[k]) for k in first), f"{_f(t['mean_diff'], signed=True)} [{_f(t['ci_low'], signed=True)}, "
             f"{_f(t['ci_high'], signed=True)}]", _f(t["p_value"], 4), _f(t["p_holm"], 4)] for t in ts]


def markdown_report(result: E4Result, meta: Mapping[str, Any]) -> str:
    s = meta["settings"]
    by = {(x["analysis"], x["representation"], x["classifier"], x["target"], x["setting"], x["metric"]): x
          for x in result.summaries}
    lines = ["# E4: explanatory analyses of the representations", "",
             f"E2 run `{meta['input_runs']['E2']}`, M2 run `{meta['input_runs']['M2']}`, {meta['documents']} documents. "
             "Exploratory (DEC-64): specified before the corpus results and reported in full; they do not decide H1 "
             "of DEC-63. Differences: mean over the paired folds with the 95% interval of the corrected resampled "
             "t-test; p (Holm): adjusted within the analysis."]
    reps = meta["representations"]
    tests_of = collections.defaultdict(list)
    for t in result.tests:
        tests_of[t["analysis"]].append(t)
    if "factor_probe" in s["analyses"]:
        lines += ["", "## 1. Factor probes", "",
                  "Logistic regression predicts each factor's points (5 × 5 CV stratified by the points, over the "
                  f"documents whose points come from {', '.join(s['probe_origins'])}); macro-F1, mean over the folds. "
                  "The normalised score: ridge regression on the folds of M2, Spearman's correlation.", ""]
        rows = []
        for target in (*FACTORS, SCORE):
            metric, clf = ("spearman", "ridge") if target == SCORE else ("f1_macro", "logreg")
            base = by.get(("factor_probe", BASELINE, BASELINE, target, NONE, metric))
            if not any(("factor_probe", r, clf, target, NONE, metric) in by for r in reps):
                continue
            rows.append([target, _f(base["mean"]) if base else "–",
                         *[_f(by[("factor_probe", r, clf, target, NONE, metric)]["mean"])
                           if ("factor_probe", r, clf, target, NONE, metric) in by else "–" for r in reps]])
        lines += _table(["Target", "Majority", *reps], rows)
        lines += ["", "Each embedding minus TF-IDF:", "",
                  *_table(["Target", "Representation", "Difference [95% interval]", "p", "p (Holm)"],
                          _test_rows(tests_of["factor_probe"], ("target", "representation")))]
    if "combination" in s["analyses"]:
        lines += ["", "## 2. TF-IDF combined with each embedding", "",
                  "The TF-IDF matrix beside the standardised embedding (scaled by 1/√d, DEC-69), fitted in each training "
                  "fold of M2, with the default settings of SPEC-M2-04; against TF-IDF alone on the same folds.", "",
                  *_table(["Combination", "Classifier", "Difference [95% interval]", "p", "p (Holm)"],
                          _test_rows(tests_of["combination"], ("representation", "classifier")))]
    if "learning_curve" in s["analyses"]:
        shares = [f"{x:g}" for x in s["learning_curve_shares"]]
        lines += ["", "## 3. Learning curve", "", "logreg on a stratified share of each training fold of M2; macro-F1.", "",
                  *_table(["Representation", *shares],
                          [[r, *[_f((by.get(("learning_curve", r, "logreg", TERCILE, sh, "f1_macro")) or {}).get("mean"))
                                 for sh in shares]] for r in reps]),
                  "", "Each embedding minus TF-IDF, per share:", "",
                  *_table(["Share", "Representation", "Difference [95% interval]", "p", "p (Holm)"],
                          _test_rows(tests_of["learning_curve"], ("setting", "representation")))]
    if "error_overlap" in s["analyses"]:
        rows = []
        for t in tests_of["error_overlap"]:
            mo = t["details"]["modal"]
            overlaps = [x["overlap"] for x in t["details"]["per_repeat"] if x["overlap"] is not None]
            rows.append([t["representation"], t["classifier"], str(mo["both_wrong"]), str(mo["only_embedding_wrong"]),
                         str(mo["only_tfidf_wrong"]), _f(statistics.fmean(overlaps)) if overlaps else "–",
                         _f(t["p_value"], 4), _f(t["p_holm"], 4)])
        lines += ["", "## 4. Error overlap with TF-IDF", "",
                  "Counts on the modal prediction of each document over the repeats; overlap: both wrong / either "
                  "wrong, mean over the repeats; McNemar's exact test on the modal predictions.", "",
                  *_table(["Representation", "Classifier", "Both wrong", "Only embedding wrong", "Only TF-IDF wrong",
                           "Overlap", "p", "p (Holm)"], rows)]
    if "context_length" in s["analyses"]:
        lines += ["", "## 5. Context length within bge-m3", "",
                  f"`{CONTEXT_SHORT}` (512-token chunks) minus `{CONTEXT_LONG}` (8,192-token chunks), macro-F1 on the "
                  "folds of M2 (from E2).", "",
                  *_table(["Classifier", "Difference [95% interval]", "p", "p (Holm)"],
                          _test_rows(tests_of["context_length"], ("classifier",)))]
    for analysis, ns in result.notes.items():
        lines += ["", f"Notes on `{analysis}`: " + "; ".join(ns) + "."]
    return "\n".join(lines) + "\n"


def folds_csv(folds: Sequence[FoldValue]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["analysis", "representation", "classifier", "target", "setting", "repeat", "fold", "metric", "value",
                "test_train_ratio"])
    for f in folds:
        w.writerow([f.analysis, f.representation, f.classifier, f.target, f.setting, f.repeat, f.fold, f.metric,
                    "" if f.value is None else round(f.value, 6), round(f.ratio, 6)])
    return buf.getvalue()


# --- Database run ------------------------------------------------------------------------------


def load_inputs(conn: sqlite3.Connection, data_root: Path, e2_run_id: str, s: Settings) -> tuple[Inputs, dict[str, str]]:
    m2 = input_of(conn, e2_run_id, "M2")
    l3 = input_of(conn, m2, "L3")
    m1 = input_of(conn, m2, "M1")
    for run_id, stage in ((m2, "M2"), (l3, "L3"), (m1, "M1")):
        runs.require_complete(conn, run_id, stage)
    doc_ids, labels = load_labels(conn, l3)
    folds = {(r[0], r[1]): r[2] for r in conn.execute(
        "SELECT doc_id, repeat, fold FROM cv_folds WHERE run_id = ? AND scheme = ?", (m2, STRATIFIED))}
    if not folds:
        raise E4InputError(f"M2 run {m2} has no stratified fold assignment")
    position = {d: i for i, d in enumerate(doc_ids)}
    lacking = sorted({d for d, _ in folds} ^ set(doc_ids))
    if lacking:
        raise E4InputError(f"the fold assignment of M2 run {m2} and the documents of L3 run {l3} differ: {lacking[:10]}")
    splits = []
    for r in sorted({r for _, r in folds}):
        for k in sorted({f for (_, rr), f in folds.items() if rr == r}):
            test = np.array(sorted(position[d] for (d, rr), f in folds.items() if rr == r and f == k))
            train = np.array(sorted(set(range(len(doc_ids))) - set(test.tolist())))
            splits.append(Split(r, k, train, test))
    reps = [r[0] for r in conn.execute(
        "SELECT DISTINCT representation FROM model_runs WHERE run_id = ? AND scheme = ? AND status = 'complete'",
        (m2, STRATIFIED))]
    reps = [r for r in represent.REPRESENTATIONS if r in reps] + sorted(set(reps) - set(represent.REPRESENTATIONS))
    features = {rep: represent.load_features(conn, data_root, m1, rep, doc_ids) for rep in reps}
    points: dict[str, dict[str, float]] = collections.defaultdict(dict)
    marks = ",".join("?" * len(s.probe_origins))
    for d, factor, p in conn.execute(
            f"SELECT doc_id, factor, points FROM risk_label_factors WHERE run_id = ? AND origin IN ({marks})",
            (l3, *s.probe_origins)):
        points[factor][d] = p
    score_of = dict(conn.execute("SELECT doc_id, normalised_score FROM risk_labels WHERE run_id = ?", (l3,)).fetchall())
    predictions = {k: v for k, v in load_predictions(conn, m2).items() if k[0] == STRATIFIED}
    fold_f1: dict[tuple[str, str], dict[tuple[int, int], float]] = collections.defaultdict(dict)
    for rep, clf, scope, value in conn.execute(
            "SELECT representation, classifier, scope, value FROM metrics WHERE run_id = ? AND scheme = ?"
            " AND name = 'f1_macro' AND scope LIKE 'fold:%'", (e2_run_id, STRATIFIED)):
        r, k = scope.removeprefix("fold:").split(".")
        if value is not None:
            fold_f1[(rep, clf)][(int(r), int(k))] = value
    inputs = Inputs(doc_ids, np.asarray(labels), splits, features, dict(points),
                    np.array([score_of[d] for d in doc_ids], dtype=float), predictions, dict(fold_f1))
    return inputs, {"E2": e2_run_id, "M2": m2, "M1": m1, "L3": l3}


def _insert(conn: sqlite3.Connection, run_id: str, result: E4Result) -> None:
    conn.executemany(
        "INSERT INTO explain_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(run_id, x["analysis"], x["representation"], x["classifier"], x["target"], x["setting"], x["metric"],
          x["mean"], x["sd"], x["n"]) for x in result.summaries])
    conn.executemany(
        "INSERT INTO explain_tests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(run_id, t["analysis"], t["representation"], t["reference"], t["classifier"], t["target"], t["setting"],
          t["metric"], t["test"], t["n"], t["mean_diff"], t["ci_low"], t["ci_high"], t["statistic"], t["p_value"],
          t["p_holm"], json.dumps(t["details"], sort_keys=True)) for t in result.tests])


def run(
    conn: sqlite3.Connection,
    config_values: Mapping[str, Any],
    data_root: Path,
    *,
    e2_run_id: str,
    analyses: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Run the analyses of DEC-64 on the chain of an E2 run. Returns the new run id."""
    runs.require_complete(conn, e2_run_id, "E2")
    s = Settings.of(config_values.get("explain"), analyses)
    train_settings = config_values.get("train") or {}
    overrides = train_settings.get("classifier_params") or {}
    params = {c: clf_mod.params(c, overrides.get(c)) for c in ("logreg", "svm")}
    tfidf = tfidf_settings_of((config_values.get("represent") or {}).get("tfidf"))
    inputs, input_runs = load_inputs(conn, data_root, e2_run_id, s)
    snapshot = {**config_values, "e4_run": {**asdict(s), "classifier_params": params}}
    run_id = runs.start(conn, "E4", snapshot, inputs=sorted(set(input_runs.values())))
    written: list[str] = []
    try:
        result = compute(inputs, s, params, tfidf, progress)
        meta = {"input_runs": input_runs, "documents": len(inputs.doc_ids), "settings": asdict(s),
                "representations": list(inputs.features), "classifier_params": params}
        folder = f"reports/{run_id}"
        written.append(files.write_text(data_root, f"{folder}/e4_report.md", markdown_report(result, meta)))
        written.append(files.write_text(data_root, f"{folder}/e4_folds.csv", folds_csv(result.folds)))
        report = {**meta, "summaries": result.summaries, "tests": result.tests, "notes": result.notes,
                  "files": ["e4_report.md", "e4_folds.csv"],
                  "definitions": {
                      "difference": "representation minus reference, mean over the paired folds; 95% interval and p "
                                    "from the corrected resampled t-test (Nadeau and Bengio, 2003), two-sided",
                      "p_holm": "Holm's step-down adjustment within the analysis (DEC-64)",
                      "error_overlap": "McNemar's exact test on the modal prediction of each document over the repeats",
                      "combination_block": "standardised embedding scaled by 1/sqrt(d) beside the TF-IDF row (DEC-69)",
                  }}
        report_path = files.write_json(data_root, f"{folder}/e4_report.json", report)
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
