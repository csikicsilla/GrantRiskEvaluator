"""M2 cross-validation on synthetic data (SPEC-M2-02 … -06, -08)."""

import collections

import numpy as np
import pytest
from modelling_helpers import CLASS_WORDS, synthetic_corpus

from grantrisk.modelling import classifiers, train
from grantrisk.modelling.text import plain_text
from grantrisk.modelling.train import LABELS, CVSettings, Split

TEXTS, LABEL_OF = synthetic_corpus(n_per_class=10)
DOCS = sorted(TEXTS)
Y = [LABEL_OF[d] for d in DOCS]
X_TEXT = [plain_text(TEXTS[d]) for d in DOCS]
RNG = np.random.default_rng(0)
X_DENSE = np.array([[3.0 * (y == label) for label in LABELS] for y in Y]) + RNG.normal(0, 1, (len(Y), 3))
CV = CVSettings()
FAST = {"rf": {"n_estimators": 10}}


def test_stratified_splits_cover_every_document_once_per_repeat():
    """SPEC-M2-02: 25 cycles; each document tested once per repeat; a fifth of each class per test fold."""
    splits = train.stratified_splits(Y, CV)
    assert len(splits) == 25
    for r in range(5):
        tested = np.concatenate([s.test for s in splits if s.repeat == r])
        assert sorted(tested) == list(range(len(Y)))
    for s in splits:
        assert not set(s.train) & set(s.test)
        counts = collections.Counter(np.asarray(Y)[s.test])
        assert all(abs(counts[label] - Y.count(label) / 5) <= 1 for label in LABELS)
    again = train.stratified_splits(Y, CV)
    assert all((a.test == b.test).all() for a, b in zip(splits, again))


def test_too_small_class_is_refused():
    with pytest.raises(ValueError, match="at least 5 documents per class"):
        train.check_classes(["low"] * 10 + ["medium"] * 10 + ["high"] * 4, 5)


def test_predicted_label_is_the_argmax_and_a_tie_goes_up():
    assert train.predicted_label([0.2, 0.5, 0.3]) == "medium"
    assert train.predicted_label([0.4, 0.4, 0.2]) == "medium"
    assert train.predicted_label([1 / 3, 1 / 3, 1 / 3]) == "high"


def test_tfidf_is_fitted_on_the_training_part_only():
    """SPEC-M1-05, SPEC-M2-03: a word that occurs only in test documents is not in the vocabulary."""
    split = train.stratified_splits(Y, CV)[0]
    x = list(X_TEXT)
    for i in split.test:
        x[i] += " kizárólagtesztszó"
    pipe = train.fit_fold(x, np.asarray(Y), split, text=True, classifier="logreg",
                          params=classifiers.params("logreg"), tfidf_settings={"min_df": 1}, seed=42)
    vocabulary = pipe.named_steps["tfidf"].vocabulary_
    assert "kizárólagtesztszó" not in vocabulary
    assert CLASS_WORDS["high"] in vocabulary


def test_dense_features_are_standardised_for_logreg_and_svm_only():
    steps = lambda c: [n for n, _ in classifiers.build(c, classifiers.params(c), text=False, tfidf_settings=None, seed=1).steps]
    assert steps("logreg") == ["scale", "clf"] and steps("svm") == ["scale", "clf"]
    assert steps("rf") == ["clf"] and steps("majority") == ["clf"]
    assert [n for n, _ in classifiers.build("svm", classifiers.params("svm"), text=True, tfidf_settings=None, seed=1).steps] == ["tfidf", "clf"]


@pytest.mark.parametrize("classifier", classifiers.CLASSIFIERS)
@pytest.mark.parametrize("text", [False, True])
def test_every_classifier_predicts_every_document_once_per_repeat(classifier, text):
    """SPEC-M2-04, -05: probabilities always present; the label is their argmax."""
    splits = train.stratified_splits(Y, CVSettings(n_repeats=2))
    result = train.fit_predict(X_TEXT if text else X_DENSE, Y, splits, text=text, classifier=classifier,
                               params=classifiers.params(classifier, FAST.get(classifier)))
    assert result.probabilities.shape == (2, len(Y), 3)
    assert not np.isnan(result.probabilities).any()
    assert np.allclose(result.probabilities.sum(axis=2), 1.0)
    assert (result.folds >= 0).all()


def test_learnable_signal_is_learnt():
    splits = train.stratified_splits(Y, CVSettings(n_repeats=1))
    result = train.fit_predict(X_TEXT, Y, splits, text=True, classifier="logreg", params=classifiers.params("logreg"))
    predicted = [train.predicted_label(p) for p in result.probabilities[0]]
    assert np.mean([p == t for p, t in zip(predicted, Y)]) > 0.9


def test_majority_baseline_predicts_the_class_shares_of_the_training_fold():
    """DEC-25."""
    y = ["low"] * 10 + ["medium"] * 15 + ["high"] * 5
    x = np.zeros((len(y), 2))
    splits = train.stratified_splits(y, CVSettings(n_repeats=1))
    result = train.fit_predict(x, y, splits, text=False, classifier="majority", params={})
    for s in splits:
        train_labels = collections.Counter(np.asarray(y)[s.train])
        shares = [train_labels[label] / len(s.train) for label in LABELS]
        assert np.allclose(result.probabilities[0, s.test], shares)
        assert {train.predicted_label(p) for p in result.probabilities[0, s.test]} == {"medium"}


def test_same_input_gives_same_predictions():
    """SPEC-M2-07."""
    splits = train.stratified_splits(Y, CVSettings(n_repeats=2))
    a = train.fit_predict(X_DENSE, Y, splits, text=False, classifier="rf", params=classifiers.params("rf", FAST["rf"]))
    b = train.fit_predict(X_DENSE, Y, splits, text=False, classifier="rf", params=classifiers.params("rf", FAST["rf"]))
    assert np.array_equal(a.probabilities, b.probabilities)


def test_parallel_folds_give_the_same_result():
    splits = train.stratified_splits(Y, CVSettings(n_repeats=1))
    a = train.fit_predict(X_DENSE, Y, splits, text=False, classifier="logreg", params=classifiers.params("logreg"))
    b = train.fit_predict(X_DENSE, Y, splits, text=False, classifier="logreg", params=classifiers.params("logreg"), n_jobs=2)
    assert np.allclose(a.probabilities, b.probabilities)


def test_top_terms_of_tfidf_logreg():
    """SPEC-M2-06: per class, the terms with the largest average weight."""
    splits = train.stratified_splits(Y, CV)
    result = train.fit_predict(X_TEXT, Y, splits, text=True, classifier="logreg",
                               params=classifiers.params("logreg"), top_terms=True)
    assert set(result.top_terms) == set(LABELS)
    for label, word in CLASS_WORDS.items():
        terms = result.top_terms[label]
        assert 0 < len(terms) <= 30
        assert terms[0][0].startswith(word) or word in [t for t, _, _ in terms[:3]]
        assert all(0 < n <= 25 for _, _, n in terms)
    none = train.fit_predict(X_DENSE, Y, splits[:5], text=False, classifier="logreg", params=classifiers.params("logreg"), top_terms=True)
    assert none.top_terms is None


def test_top_terms_average_counts_absent_folds_as_zero():
    names = [np.array(["a", "b"]), np.array(["a"])]
    coef = [np.array([[2.0, 4.0], [0, 0], [0, 0]]), np.array([[4.0], [0], [0]])]
    top = train._top_terms(list(zip(names, coef)), n_folds=2)
    assert top["low"] == [("a", 3.0, 2), ("b", 2.0, 1)]


def test_grouped_splits_keep_each_series_on_one_side():
    """SPEC-M2-08."""
    groups = [f"S{i // 2}" for i in range(len(Y))]  # pairs of documents share a series
    splits = train.grouped_splits(Y, groups, CV)
    assert len(splits) == 25
    g = np.asarray(groups)
    for s in splits:
        assert not set(g[s.train]) & set(g[s.test])
    for r in range(5):
        tested = np.concatenate([s.test for s in splits if s.repeat == r])
        assert sorted(tested) == list(range(len(Y)))


# --- Tuning of C inside the training folds (DEC-63 (b)) ---------------------------------------

TUNING = train.Tuning(C=(0.01, 1.0, 100.0))


def test_tuning_reads_the_configuration():
    assert train.Tuning.of(None) is None and train.Tuning.of({"enabled": False}) is None
    t = train.Tuning.of({"enabled": True, "C": [0.1, 10], "inner_splits": 3})
    assert t.C == (0.1, 10.0) and t.classifiers == ("logreg", "svm") and t.applies("svm") and not t.applies("rf")
    with pytest.raises(ValueError, match="can be tuned"):
        train.Tuning.of({"enabled": True, "classifiers": ["rf"]})


@pytest.mark.parametrize("text", [False, True])
def test_c_is_chosen_from_the_training_part_only(text):
    """The documents of the test part can change without changing the choice."""
    split = train.stratified_splits(Y, CV)[0]
    X = list(X_TEXT) if text else X_DENSE.copy()
    kwargs = dict(text=text, classifier="logreg", params=classifiers.params("logreg"),
                  tfidf_settings={"min_df": 1}, seed=42, tuning=TUNING)
    before = train.choose_c(X, np.asarray(Y), split.train, **kwargs)
    for i in split.test:
        if text:
            X[i] = "egészen más szöveg " * 50
        else:
            X[i] = 100.0
    assert train.choose_c(X, np.asarray(Y), split.train, **kwargs) == before
    best, scores = before
    assert set(scores) == set(TUNING.C) and all(0.0 <= s <= 1.0 for s in scores.values())


def test_a_tie_goes_to_the_smaller_c():
    """Separable classes: every C scores 1.0, and the strongest regularisation is chosen."""
    X = np.array([[10.0 * (y == label) for label in LABELS] for y in Y])
    split = train.stratified_splits(Y, CV)[0]
    best, scores = train.choose_c(X, np.asarray(Y), split.train, text=False, classifier="logreg",
                                  params=classifiers.params("logreg"), tfidf_settings=None,
                                  seed=42, tuning=train.Tuning(C=(100.0, 1.0, 0.01)))
    assert set(scores.values()) == {1.0} and best == 0.01


def test_the_chosen_c_is_the_one_fitted_on_the_training_fold():
    """With a single candidate, the tuned run equals an untuned run with that C."""
    splits = train.stratified_splits(Y, CVSettings(n_repeats=1))
    tuned = train.fit_predict(X_DENSE, Y, splits, text=False, classifier="logreg", params=classifiers.params("logreg"),
                              tuning=train.Tuning(C=(0.05,)))
    fixed = train.fit_predict(X_DENSE, Y, splits, text=False, classifier="logreg",
                              params=classifiers.params("logreg", {"C": 0.05}))
    assert np.allclose(tuned.probabilities, fixed.probabilities)
    assert [c["C"] for c in tuned.tuning] == [0.05] * 5


@pytest.mark.parametrize("classifier", classifiers.CLASSIFIERS)
def test_every_fold_records_its_choice_for_the_tuned_classifiers_only(classifier):
    splits = train.stratified_splits(Y, CVSettings(n_repeats=2))
    result = train.fit_predict(X_TEXT, Y, splits, text=True, classifier=classifier,
                               params=classifiers.params(classifier, FAST.get(classifier)), tuning=TUNING)
    if classifier not in ("logreg", "svm"):
        assert result.tuning is None
        return
    assert [(c["repeat"], c["fold"]) for c in result.tuning] == [(r, f) for r in range(2) for f in range(5)]
    assert all(c["C"] in TUNING.C and set(c["inner_macro_f1"]) == {"0.01", "1.0", "100.0"} for c in result.tuning)
    summary = train.tuning_summary(result.tuning)
    assert sum(summary["chosen_C"].values()) == 10


def test_tuned_folds_in_parallel_give_the_same_result():
    splits = train.stratified_splits(Y, CVSettings(n_repeats=1))
    kwargs = dict(text=False, classifier="svm", params=classifiers.params("svm"), tuning=TUNING)
    a = train.fit_predict(X_DENSE, Y, splits, **kwargs)
    b = train.fit_predict(X_DENSE, Y, splits, n_jobs=2, **kwargs)
    assert np.allclose(a.probabilities, b.probabilities) and a.tuning == b.tuning
