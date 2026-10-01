"""The classifiers of the M2 grid (SPEC-M2-04), each built fresh for every fold.

Everything that learns from data sits in one pipeline, so it is fitted only on the
training part of a fold (SPEC-M2-03): the TF-IDF vectoriser, the scaler and the
classifier. With tuning (DEC-63 (b)), `C` of `logreg` and `svm` is chosen inside the
training part as well (train.choose_c).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from grantrisk.modelling import tfidf

CLASSIFIERS = ("logreg", "svm", "rf", "majority")
TUNABLE = ("logreg", "svm")  # the classifiers with a `C` (DEC-63 (b))

DEFAULTS: dict[str, dict[str, Any]] = {
    "logreg": {"C": 1.0, "class_weight": "balanced", "max_iter": 5000},
    # Platt scaling with an internal 5-fold CV: what SVC(probability=True) did, which
    # scikit-learn 1.9 deprecates in favour of this form.
    "svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced", "calibration_cv": 5},
    "rf": {"n_estimators": 500, "class_weight": "balanced", "n_jobs": 1},
    "majority": {},
}


def params(classifier: str, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The settings of a classifier: the defaults of SPEC-M2-04, overridden by the configuration."""
    if classifier not in DEFAULTS:
        raise ValueError(f"unknown classifier {classifier!r}; known: {list(CLASSIFIERS)}")
    return {**DEFAULTS[classifier], **(overrides or {})}


def estimator(classifier: str, p: Mapping[str, Any], seed: int):
    """A new, unfitted classifier, without the steps before it."""
    if classifier == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=p["max_iter"], random_state=seed)
    if classifier == "svm":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC

        svc = SVC(C=p["C"], kernel=p["kernel"], gamma=p["gamma"], class_weight=p["class_weight"], random_state=seed)
        return CalibratedClassifierCV(svc, method="sigmoid", cv=p["calibration_cv"], ensemble=False)
    if classifier == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=p["n_estimators"], class_weight=p["class_weight"], n_jobs=p.get("n_jobs", 1), random_state=seed
        )
    if classifier == "majority":
        from sklearn.dummy import DummyClassifier

        # predict_proba gives the class shares of the training fold (DEC-25)
        return DummyClassifier(strategy="prior")
    raise ValueError(f"unknown classifier {classifier!r}")


def preprocessing(classifier: str, *, text: bool, tfidf_settings: Mapping[str, Any] | None) -> list[tuple[str, Any]]:
    """The new, unfitted steps before the classifier.

    ``text``: the features are plain texts, vectorised by TF-IDF. Otherwise they are
    dense embeddings, standardised before ``logreg`` and ``svm``.
    """
    from sklearn.preprocessing import StandardScaler

    if text:
        return [("tfidf", tfidf.vectorizer(tfidf_settings))]  # already L2-normalised; not standardised
    if classifier in ("logreg", "svm"):
        return [("scale", StandardScaler())]
    return []


def build(classifier: str, p: Mapping[str, Any], *, text: bool, tfidf_settings: Mapping[str, Any] | None, seed: int):
    """A new, unfitted pipeline for one fold: the preprocessing steps and the classifier."""
    from sklearn.pipeline import Pipeline

    steps = preprocessing(classifier, text=text, tfidf_settings=tfidf_settings)
    return Pipeline([*steps, ("clf", estimator(classifier, p, seed))])
