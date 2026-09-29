"""Stage M2: train & predict (Spec_M2_TrainPredict.md).

Every classifier is cross-validated on every representation with one stored fold
assignment, and every prediction comes from a model that did not see the document
(INT-EVAL-10). ``fit_predict`` is the pure computation; ``run`` reads an M1 and an
L3 run and writes the fold assignment, the model runs and the predictions.
"""

from __future__ import annotations

import collections
import json
import sqlite3
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from grantrisk.modelling import classifiers as clf_mod
from grantrisk.modelling import represent
from grantrisk.modelling.represent import REPRESENTATIONS, TFIDF
from grantrisk.modelling.tfidf import settings as tfidf_settings_of
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

LABELS = ("low", "medium", "high")
STRATIFIED, GROUPED = "stratified", "grouped"
TOP_TERMS = 30


@dataclass(frozen=True)
class CVSettings:
    n_splits: int = 5
    n_repeats: int = 5
    random_state: int = 42


@dataclass(frozen=True)
class Split:
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray


@dataclass
class ModelResult:
    probabilities: np.ndarray  # repeat × document × (low, medium, high)
    folds: np.ndarray  # repeat × document: the fold in which the document was tested
    warnings: dict[str, int] = field(default_factory=dict)
    top_terms: dict[str, list[tuple[str, float, int]]] | None = None


def check_classes(y: Sequence[str], n_splits: int) -> None:
    """Edge case: stratified k-fold CV needs at least k documents in every class."""
    counts = collections.Counter(y)
    unknown = sorted(set(counts) - set(LABELS))
    if unknown:
        raise ValueError(f"unknown labels {unknown}")
    small = {c: counts.get(c, 0) for c in LABELS if counts.get(c, 0) < n_splits}
    if small:
        raise ValueError(f"stratified {n_splits}-fold CV needs at least {n_splits} documents per class; the L3 run has {small}")


def stratified_splits(y: Sequence[str], cv: CVSettings) -> list[Split]:
    """SPEC-M2-02: RepeatedStratifiedKFold; in each repeat every document is tested once."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    splitter = RepeatedStratifiedKFold(n_splits=cv.n_splits, n_repeats=cv.n_repeats, random_state=cv.random_state)
    return [
        Split(i // cv.n_splits, i % cv.n_splits, train, test)
        for i, (train, test) in enumerate(splitter.split(np.zeros(len(y)), np.asarray(y)))
    ]


def grouped_splits(y: Sequence[str], groups: Sequence[str], cv: CVSettings) -> list[Split]:
    """SPEC-M2-08: StratifiedGroupKFold by call series, one seed per repeat."""
    from sklearn.model_selection import StratifiedGroupKFold

    out = []
    for r in range(cv.n_repeats):
        splitter = StratifiedGroupKFold(n_splits=cv.n_splits, shuffle=True, random_state=cv.random_state + r)
        for f, (train, test) in enumerate(splitter.split(np.zeros(len(y)), np.asarray(y), np.asarray(groups))):
            out.append(Split(r, f, train, test))
    return out


def predicted_label(probabilities: Sequence[float]) -> str:
    """The class with the highest probability; a tie goes to the higher-risk class (as DEC-32)."""
    return LABELS[max(range(len(LABELS)), key=lambda i: (probabilities[i], i))]


def _subset(X: Any, idx: np.ndarray) -> Any:
    return [X[i] for i in idx] if isinstance(X, list) else X[idx]


def fit_fold(X: Any, y: np.ndarray, split: Split, *, text: bool, classifier: str, params: Mapping[str, Any],
             tfidf_settings: Mapping[str, Any] | None, seed: int):
    """Fit a new pipeline on the training part of one fold only (SPEC-M2-03)."""
    pipe = clf_mod.build(classifier, params, text=text, tfidf_settings=tfidf_settings, seed=seed)
    pipe.fit(_subset(X, split.train), y[split.train])
    return pipe


def _fold(X, y, split, text, classifier, params, tfidf_settings, seed, want_terms):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe = fit_fold(X, y, split, text=text, classifier=classifier, params=params,
                        tfidf_settings=tfidf_settings, seed=seed)
        raw = pipe.predict_proba(_subset(X, split.test))
    classes = list(pipe.classes_)
    probs = np.zeros((len(split.test), len(LABELS)))
    for j, label in enumerate(LABELS):
        if label in classes:  # always, with stratified folds (SPEC-M2-05)
            probs[:, j] = raw[:, classes.index(label)]
    terms = None
    if want_terms:
        names = pipe.named_steps["tfidf"].get_feature_names_out()
        coef = pipe.named_steps["clf"].coef_
        clf_classes = list(pipe.named_steps["clf"].classes_)
        terms = (names, np.stack([coef[clf_classes.index(label)] for label in LABELS]))
    messages = [f"{w.category.__name__}: {str(w.message)[:200]}" for w in caught]
    return split, probs, messages, terms


def _top_terms(fold_terms, n_folds: int, k: int = TOP_TERMS) -> dict[str, list[tuple[str, float, int]]]:
    """SPEC-M2-06: per class, the terms with the largest weight averaged over all folds (absent = 0)."""
    index: dict[str, int] = {}
    sums = np.zeros((len(LABELS), 0))
    counts = np.zeros(0, dtype=int)
    for names, coef in fold_terms:
        ids = np.fromiter((index.setdefault(t, len(index)) for t in names), dtype=int, count=len(names))
        if len(index) > sums.shape[1]:
            sums = np.pad(sums, ((0, 0), (0, len(index) - sums.shape[1])))
            counts = np.pad(counts, (0, len(index) - counts.shape[0]))
        sums[:, ids] += coef
        counts[ids] += 1
    terms = np.array(list(index), dtype=object)
    out = {}
    for j, label in enumerate(LABELS):
        mean = sums[j] / n_folds
        order = sorted(range(len(terms)), key=lambda i: (-mean[i], terms[i]))[:k]
        out[label] = [(str(terms[i]), float(mean[i]), int(counts[i])) for i in order]
    return out


def fit_predict(
    X: Any,
    y: Sequence[str],
    splits: Sequence[Split],
    *,
    text: bool,
    classifier: str,
    params: Mapping[str, Any],
    tfidf_settings: Mapping[str, Any] | None = None,
    seed: int = 42,
    n_jobs: int = 1,
    top_terms: bool = False,
) -> ModelResult:
    """Cross-validate one classifier on one representation (SPEC-M2-03 … -06).

    ``X`` holds plain texts (a list) when ``text`` is true, and a matrix otherwise, in
    the order of ``y``. Each document gets one prediction per repeat.
    """
    y = np.asarray(y)
    n_repeats = max(s.repeat for s in splits) + 1
    want_terms = top_terms and text and classifier == "logreg"
    args = (text, classifier, params, tfidf_settings, seed, want_terms)
    if n_jobs == 1:
        results = [_fold(X, y, s, *args) for s in splits]
    else:  # the folds are independent; each has its own fixed seed, so the result does not change
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(delayed(_fold)(X, y, s, *args) for s in splits)
    probabilities = np.full((n_repeats, len(y), len(LABELS)), np.nan)
    folds = np.full((n_repeats, len(y)), -1, dtype=int)
    messages = collections.Counter()
    fold_terms = []
    for split, probs, msgs, terms in results:
        probabilities[split.repeat, split.test] = probs
        folds[split.repeat, split.test] = split.fold
        messages.update(msgs)
        if terms is not None:
            fold_terms.append(terms)
    if (folds < 0).any():
        raise ValueError("a document was not tested in every repeat")
    return ModelResult(
        probabilities, folds, dict(sorted(messages.items())),
        _top_terms(fold_terms, len(splits)) if want_terms else None,
    )


# --- Database run (SPEC-M2-01, -07) ---------------------------------------------------------


def load_labels(conn: sqlite3.Connection, l3_run_id: str) -> tuple[list[str], list[str]]:
    """SPEC-M2-01: the documents of the L3 run sorted by doc_id, and their tercile labels."""
    rows = conn.execute(
        "SELECT doc_id, tercile_label FROM risk_labels WHERE run_id = ? ORDER BY doc_id", (l3_run_id,)
    ).fetchall()
    return [r[0] for r in rows], [r[1] for r in rows]


def _input_of(conn: sqlite3.Connection, run_id: str, stage: str) -> str:
    found = [r for r in runs.inputs(conn, run_id) if runs.get(conn, r)["stage"] == stage]
    if len(found) != 1:
        raise ValueError(f"run {run_id} has no single {stage} input")
    return found[0]


def _store_scheme(conn, run_id, scheme, splitter, params, doc_ids, splits) -> None:
    with transaction(conn):
        conn.execute("INSERT INTO cv_schemes VALUES (?, ?, ?, ?)", (run_id, scheme, splitter, json.dumps(params, sort_keys=True)))
        conn.executemany(
            "INSERT INTO cv_folds VALUES (?, ?, ?, ?, ?)",
            [(run_id, scheme, doc_ids[i], s.repeat, s.fold) for s in splits for i in s.test],
        )


def _fold_check(y: Sequence[str], splits: Sequence[Split]) -> dict[str, Any]:
    """SPEC-M2-02 acceptance: each test fold holds a fifth of each class, give or take one."""
    y = np.asarray(y)
    per_class = {label: [int((y[s.test] == label).sum()) for s in splits] for label in LABELS}
    return {label: {"min": min(v), "max": max(v)} for label, v in per_class.items()}


def run(
    conn: sqlite3.Connection,
    config_values: Mapping[str, Any],
    data_root: Path,
    *,
    m1_run_id: str,
    l3_run_id: str,
    representations: Sequence[str] | None = None,
    classifiers: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Cross-validate the grid of representations × classifiers. Returns the run id.

    M2 refuses to start if a class is too small or a representation does not cover
    every document of the L3 run. A model run that fails is recorded without
    predictions, and the others continue (SPEC-M2-07).
    """
    runs.require_complete(conn, m1_run_id, "M1")
    runs.require_complete(conn, l3_run_id, "L3")
    settings = config_values.get("train") or {}
    cv = CVSettings(**(settings.get("cv") or {}))
    doc_ids, y = load_labels(conn, l3_run_id)
    if not doc_ids:
        raise ValueError(f"L3 run {l3_run_id} has no documents")
    check_classes(y, cv.n_splits)

    in_m1 = {r[0] for r in conn.execute("SELECT DISTINCT representation FROM features WHERE run_id = ?", (m1_run_id,))}
    reps = list(representations or settings.get("representations") or [r for r in REPRESENTATIONS if r in in_m1])
    absent = [r for r in reps if r not in in_m1]
    if absent:
        raise ValueError(f"M1 run {m1_run_id} has no representation {absent}")
    clfs = list(dict.fromkeys(classifiers or settings.get("classifiers") or clf_mod.CLASSIFIERS))
    if "majority" not in clfs:
        clfs.append("majority")  # the baseline is in every comparison (DEC-25)
    overrides = settings.get("classifier_params") or {}
    params = {c: clf_mod.params(c, overrides.get(c)) for c in clfs}
    tfidf = tfidf_settings_of((config_values.get("represent") or {}).get("tfidf"))

    features, problems = {}, []
    for rep in reps:  # SPEC-M2-01: every representation must cover every document
        try:
            features[rep] = represent.load_features(conn, data_root, m1_run_id, rep, doc_ids)
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise ValueError("M2 cannot start:\n  " + "\n  ".join(problems))

    grouped = settings.get("grouped_check") or {}
    grouped_models = [tuple(m.split("/")) for m in grouped.get("models") or []] if grouped.get("enabled") else []
    for rep, clf in grouped_models:
        if rep not in features or clf not in params:
            raise ValueError(f"grouped_check model {rep}/{clf} is not in the grid")
    inputs = [m1_run_id, l3_run_id]
    c1_run_id = _input_of(conn, l3_run_id, "C1") if grouped_models else None
    if c1_run_id:
        inputs.append(c1_run_id)

    snapshot = {**config_values, "m2_run": {"representations": reps, "classifiers": clfs, "cv": asdict(cv),
                                            "classifier_params": params, "grouped_models": ["/".join(m) for m in grouped_models]}}
    run_id = runs.start(conn, "M2", snapshot, inputs=inputs)
    report_path = None
    try:
        schemes = [(STRATIFIED, stratified_splits(y, cv), "RepeatedStratifiedKFold", asdict(cv), [(r, c) for r in reps for c in clfs])]
        if grouped_models:
            series = dict(conn.execute("SELECT doc_id, call_series FROM documents WHERE run_id = ?", (c1_run_id,)).fetchall())
            lacking = [d for d in doc_ids if d not in series]
            if lacking:
                raise ValueError(f"C1 run {c1_run_id} has no call_series for {lacking[:20]}")
            groups = [series[d] for d in doc_ids]
            schemes.append((GROUPED, grouped_splits(y, groups, cv), "StratifiedGroupKFold",
                            {**asdict(cv), "group_by": "call_series", "seed_per_repeat": "random_state + repeat"},
                            grouped_models))
        grid, fold_checks = [], {}
        for scheme, splits, splitter, scheme_params, models in schemes:
            _store_scheme(conn, run_id, scheme, splitter, scheme_params, doc_ids, splits)
            fold_checks[scheme] = _fold_check(y, splits)
            for rep, clf in models:
                started = time.perf_counter()
                entry = {"scheme": scheme, "representation": rep, "classifier": clf}
                try:
                    result = fit_predict(
                        features[rep], y, splits, text=rep == TFIDF, classifier=clf, params=params[clf],
                        tfidf_settings=tfidf, seed=cv.random_state, n_jobs=settings.get("n_jobs", 1),
                        top_terms=scheme == STRATIFIED,
                    )
                except Exception as exc:  # reported; no predictions; the grid continues
                    duration = round(time.perf_counter() - started, 1)
                    with transaction(conn):
                        conn.execute(
                            "INSERT INTO model_runs VALUES (?, ?, ?, ?, 'failed', ?, NULL, ?, ?)",
                            (run_id, scheme, rep, clf, json.dumps(params[clf], sort_keys=True), f"{type(exc).__name__}: {exc}", duration),
                        )
                    grid.append({**entry, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "duration_s": duration})
                    if progress:
                        progress(f"{scheme} {rep}/{clf} FAILED {exc}")
                    continue
                duration = round(time.perf_counter() - started, 1)
                with transaction(conn):
                    conn.execute(
                        "INSERT INTO model_runs VALUES (?, ?, ?, ?, 'complete', ?, ?, NULL, ?)",
                        (run_id, scheme, rep, clf, json.dumps(params[clf], sort_keys=True),
                         json.dumps(result.warnings) if result.warnings else None, duration),
                    )
                    conn.executemany(
                        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            (run_id, scheme, rep, clf, d, r, int(result.folds[r, i]), y[i],
                             predicted_label(result.probabilities[r, i]), *map(float, result.probabilities[r, i]))
                            for r in range(result.folds.shape[0]) for i, d in enumerate(doc_ids)
                        ],
                    )
                    if result.top_terms:
                        conn.executemany(
                            "INSERT INTO top_terms VALUES (?, ?, ?, ?, ?, ?, ?)",
                            [(run_id, scheme, label, rank, term, weight, n)
                             for label, terms in result.top_terms.items()
                             for rank, (term, weight, n) in enumerate(terms, start=1)],
                        )
                grid.append({**entry, "status": "complete", "duration_s": duration, "warnings": result.warnings})
                if progress:
                    progress(f"{scheme} {rep}/{clf} {duration} s")
        if all(g["status"] == "failed" for g in grid):
            raise RuntimeError("every model run failed")
        report = {
            "input_runs": {"M1": m1_run_id, "L3": l3_run_id, **({"C1": c1_run_id} if c1_run_id else {})},
            "documents": len(doc_ids),
            "class_sizes": {label: y.count(label) for label in LABELS},
            "cv": asdict(cv),
            "test_fold_class_counts": fold_checks,
            "representations": reps,
            "classifiers": clfs,
            "classifier_params": params,
            "tfidf": tfidf,
            "grid": grid,
            "failed": [g for g in grid if g["status"] == "failed"],
        }
        report_path = files.write_json(data_root, f"reports/{run_id}/m2_report.json", report)
        with transaction(conn):
            runs.complete(conn, run_id, report_path)
    except BaseException as exc:  # the stored model runs stay, under a failed run that E2 does not read
        if report_path:
            files.remove(data_root, report_path)
        runs.fail(conn, run_id, f"{type(exc).__name__}: {exc}")
        raise
    return run_id
