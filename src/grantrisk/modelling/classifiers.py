"""The classifiers of the M2 grid (SPEC-M2-04), each built fresh for every fold.

Everything that learns from data sits in one pipeline, so it is fitted only on the
training part of a fold (SPEC-M2-03): the TF-IDF vectoriser, the scaler and the
classifier. With tuning (DEC-63 (b)), `C` of `logreg` and `svm` is chosen inside the
training part as well (train.choose_c).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.base import BaseEstimator, ClassifierMixin

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


class RBFSVC(ClassifierMixin, BaseEstimator):
    """An RBF-kernel SVM: the model of ``SVC(kernel="rbf")``, with the kernel computed from dot products.

    libsvm evaluates the kernel pair by pair, which is slow on sparse TF-IDF rows of 200,000
    features. Here ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y comes from one (sparse) matrix product,
    and libsvm solves the same problem on the precomputed kernel. ``gamma="scale"`` is computed
    exactly as scikit-learn does: 1 / (n_features * Var(X)), over every entry of X.
    """

    def __init__(self, C: float = 1.0, gamma: str | float = "scale", class_weight: Any = None, random_state: Any = None):
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state

    @staticmethod
    def _sq_norms(X) -> Any:
        import numpy as np
        import scipy.sparse as sp

        return np.asarray(X.multiply(X).sum(axis=1)).ravel() if sp.issparse(X) else np.einsum("ij,ij->i", X, X)

    def _kernel(self, X) -> Any:
        import numpy as np
        import scipy.sparse as sp

        dots = X @ self.X_fit_.T
        dots = dots.toarray() if sp.issparse(dots) else np.asarray(dots)
        d2 = self._sq_norms(X)[:, None] + self.sq_fit_[None, :] - 2.0 * dots
        return np.exp(-self.gamma_ * np.maximum(d2, 0.0))

    def fit(self, X, y) -> RBFSVC:
        import numpy as np
        import scipy.sparse as sp
        from sklearn.svm import SVC

        X = X.tocsr().astype(np.float64) if sp.issparse(X) else np.asarray(X, dtype=np.float64)
        if self.gamma == "scale":
            var = (X.multiply(X)).mean() - X.mean() ** 2 if sp.issparse(X) else X.var()
            self.gamma_ = 1.0 / (X.shape[1] * var) if var != 0 else 1.0
        else:
            self.gamma_ = float(self.gamma)
        self.X_fit_, self.sq_fit_ = X, self._sq_norms(X)
        self.svc_ = SVC(kernel="precomputed", C=self.C, class_weight=self.class_weight,
                        random_state=self.random_state).fit(self._kernel(X), y)
        self.classes_ = self.svc_.classes_
        return self

    def _as_fitted(self, X):
        import numpy as np
        import scipy.sparse as sp

        return X.tocsr().astype(np.float64) if sp.issparse(X) else np.asarray(X, dtype=np.float64)

    def decision_function(self, X):
        return self.svc_.decision_function(self._kernel(self._as_fitted(X)))

    def predict(self, X):
        return self.svc_.predict(self._kernel(self._as_fitted(X)))


def estimator(classifier: str, p: Mapping[str, Any], seed: int):
    """A new, unfitted classifier, without the steps before it."""
    if classifier == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=p["max_iter"], random_state=seed)
    if classifier == "svm":
        from sklearn.calibration import CalibratedClassifierCV

        if p["kernel"] != "rbf":
            raise ValueError(f"svm: only the RBF kernel is implemented (got {p['kernel']!r})")
        svc = RBFSVC(C=p["C"], gamma=p["gamma"], class_weight=p["class_weight"], random_state=seed)  # DEC-69
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
