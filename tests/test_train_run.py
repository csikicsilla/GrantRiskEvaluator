"""The M2 run in the database (SPEC-M2-01, -02, -05 … -08), over an M1 run with a fake embedder."""

import json
import random

import numpy as np
import pytest
from modelling_helpers import FakeEmbedder, seed_c2, seed_l3, synthetic_corpus

from grantrisk.modelling import represent, train
from grantrisk.store import db, runs

TEXTS, LABELS = synthetic_corpus(n_per_class=10)
N = len(TEXTS)
CONFIG = {
    "represent": {"representations": ["tfidf", "hubert"]},
    "train": {"classifier_params": {"rf": {"n_estimators": 10}}},
}


@pytest.fixture
def data_root(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def conn(data_root):
    c = db.connect(data_root)
    yield c
    c.close()


def seed(conn, data_root, texts=TEXTS, labels=LABELS, series=None):
    c2 = seed_c2(conn, texts)
    m1 = represent.run(conn, CONFIG, data_root, c2_run_id=c2, embedders={"hubert": FakeEmbedder()})
    return m1, seed_l3(conn, labels, series)


def predictions(conn, run_id, scheme="stratified"):
    return conn.execute(
        "SELECT representation, classifier, doc_id, repeat, fold, true_label, predicted_label, p_low, p_medium, p_high"
        " FROM predictions WHERE run_id = ? AND scheme = ? ORDER BY 1, 2, 3, 4", (run_id, scheme)).fetchall()


def test_run_writes_the_grid(conn, data_root):
    """SPEC-M2-04, -05: every model run has 5 × n predictions; the label is the argmax."""
    m1, l3 = seed(conn, data_root)
    run_id = train.run(conn, CONFIG, data_root, m1_run_id=m1, l3_run_id=l3)
    assert runs.get(conn, run_id)["status"] == "complete"
    assert runs.inputs(conn, run_id) == sorted([m1, l3])
    models = conn.execute("SELECT representation, classifier, status FROM model_runs WHERE run_id = ?", (run_id,)).fetchall()
    assert len(models) == 2 * 4 and {m["status"] for m in models} == {"complete"}
    for m in models:
        n = conn.execute("SELECT COUNT(*) FROM predictions WHERE run_id = ? AND representation = ? AND classifier = ?",
                         (run_id, m["representation"], m["classifier"])).fetchone()[0]
        assert n == 5 * N
    for p in predictions(conn, run_id):
        assert p["predicted_label"] == train.predicted_label([p["p_low"], p["p_medium"], p["p_high"]])
        assert p["true_label"] == LABELS[p["doc_id"]]
    report = json.loads((data_root / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))
    assert report["input_runs"] == {"M1": m1, "L3": l3}
    assert report["class_sizes"] == {"low": 10, "medium": 10, "high": 10}
    assert report["test_fold_class_counts"]["stratified"]["low"] == {"min": 2, "max": 2}
    assert report["failed"] == []


def test_one_fold_assignment_for_every_model_run(conn, data_root):
    """SPEC-M2-02: stored once; the predictions carry the same folds."""
    m1, l3 = seed(conn, data_root)
    run_id = train.run(conn, CONFIG, data_root, m1_run_id=m1, l3_run_id=l3, classifiers=["logreg"])
    folds = {(r["doc_id"], r["repeat"]): r["fold"] for r in conn.execute(
        "SELECT doc_id, repeat, fold FROM cv_folds WHERE run_id = ? AND scheme = 'stratified'", (run_id,))}
    assert len(folds) == 5 * N
    for p in predictions(conn, run_id):
        assert folds[(p["doc_id"], p["repeat"])] == p["fold"]


def test_majority_baseline_is_always_in_the_grid(conn, data_root):
    m1, l3 = seed(conn, data_root)
    run_id = train.run(conn, CONFIG, data_root, m1_run_id=m1, l3_run_id=l3, classifiers=["logreg"])
    clfs = {r[0] for r in conn.execute("SELECT classifier FROM model_runs WHERE run_id = ?", (run_id,))}
    assert clfs == {"logreg", "majority"}


def test_same_input_gives_same_predictions_whatever_the_row_order(conn, data_root):
    """SPEC-M2-01, -07: sorted by doc_id; identical predictions."""
    m1, l3 = seed(conn, data_root)
    shuffled = dict(random.Random(3).sample(sorted(LABELS.items()), N))
    l3_shuffled = seed_l3(conn, shuffled)
    a = train.run(conn, CONFIG, data_root, m1_run_id=m1, l3_run_id=l3, classifiers=["rf", "logreg"])
    b = train.run(conn, CONFIG, data_root, m1_run_id=m1, l3_run_id=l3_shuffled, classifiers=["rf", "logreg"])
    assert [tuple(r) for r in predictions(conn, a)] == [tuple(r) for r in predictions(conn, b)]


def test_top_terms_are_stored_for_every_class(conn, data_root):
    """SPEC-M2-06."""
    m1, l3 = seed(conn, data_root)
    run_id = train.run(conn, CONFIG, data_root, m1_run_id=m1, l3_run_id=l3, classifiers=["logreg"])
    rows = conn.execute("SELECT class_label, COUNT(*) FROM top_terms WHERE run_id = ? GROUP BY 1", (run_id,)).fetchall()
    assert {r[0] for r in rows} == {"low", "medium", "high"}
    assert all(0 < r[1] <= 30 for r in rows)


def test_incomplete_representation_is_refused(conn, data_root):
    """SPEC-M2-01: M2 does not start, and lists what is missing."""
    missing = sorted(TEXTS)[0]
    c2 = seed_c2(conn, TEXTS, failed={missing})
    m1 = represent.run(conn, CONFIG, data_root, c2_run_id=c2, embedders={"hubert": FakeEmbedder()})
    l3 = seed_l3(conn, LABELS)
    with pytest.raises(ValueError, match=missing):
        train.run(conn, CONFIG, data_root, m1_run_id=m1, l3_run_id=l3)
    assert conn.execute("SELECT COUNT(*) FROM runs WHERE stage = 'M2'").fetchone()[0] == 0


def test_a_representation_not_in_m1_is_refused(conn, data_root):
    m1, l3 = seed(conn, data_root)
    with pytest.raises(ValueError, match="e5"):
        train.run(conn, CONFIG, data_root, m1_run_id=m1, l3_run_id=l3, representations=["e5"])


def test_too_small_class_is_refused(conn, data_root):
    labels = dict(LABELS)
    for d in [d for d, label in labels.items() if label == "high"][:7]:
        labels[d] = "medium"
    m1, l3 = seed(conn, data_root, labels=labels)
    with pytest.raises(ValueError, match="at least 5"):
        train.run(conn, CONFIG, data_root, m1_run_id=m1, l3_run_id=l3)


def test_a_failed_model_run_leaves_no_predictions_and_the_others_continue(conn, data_root):
    """SPEC-M2-07."""
    m1, l3 = seed(conn, data_root)
    config = {**CONFIG, "train": {"classifier_params": {"svm": {"C": -1.0}}}}
    run_id = train.run(conn, config, data_root, m1_run_id=m1, l3_run_id=l3, representations=["hubert"],
                       classifiers=["svm", "logreg"])
    assert runs.get(conn, run_id)["status"] == "complete"
    status = dict(conn.execute("SELECT classifier, status FROM model_runs WHERE run_id = ?", (run_id,)).fetchall())
    assert status == {"svm": "failed", "logreg": "complete", "majority": "complete"}
    assert conn.execute("SELECT COUNT(*) FROM predictions WHERE run_id = ? AND classifier = 'svm'", (run_id,)).fetchone()[0] == 0
    report = json.loads((data_root / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))
    assert [f["classifier"] for f in report["failed"]] == ["svm"]


def test_grouped_check_keeps_each_series_on_one_side(conn, data_root):
    """SPEC-M2-08: a separate CV scheme for the chosen model runs."""
    series = {d: f"S{i // 2}" for i, d in enumerate(sorted(TEXTS))}
    m1, l3 = seed(conn, data_root, series=series)
    config = {**CONFIG, "train": {**CONFIG["train"], "grouped_check": {"enabled": True, "models": ["hubert/logreg"]}}}
    run_id = train.run(conn, config, data_root, m1_run_id=m1, l3_run_id=l3, classifiers=["logreg"])
    schemes = {r[0] for r in conn.execute("SELECT scheme FROM cv_schemes WHERE run_id = ?", (run_id,))}
    assert schemes == {"stratified", "grouped"}
    grouped = conn.execute("SELECT DISTINCT representation, classifier FROM model_runs WHERE run_id = ? AND scheme = 'grouped'",
                           (run_id,)).fetchall()
    assert [tuple(r) for r in grouped] == [("hubert", "logreg")]
    folds = conn.execute("SELECT doc_id, repeat, fold FROM cv_folds WHERE run_id = ? AND scheme = 'grouped'", (run_id,)).fetchall()
    for r in range(5):
        for f in range(5):
            test = {series[x["doc_id"]] for x in folds if x["repeat"] == r and x["fold"] == f}
            train_ = {series[x["doc_id"]] for x in folds if x["repeat"] == r and x["fold"] != f}
            assert not test & train_
    assert len(predictions(conn, run_id, "grouped")) == 5 * N
    assert len(runs.inputs(conn, run_id)) == 3  # M1, L3 and the C1 run with the call series


def test_dense_features_come_from_the_cache(conn, data_root):
    m1, _ = seed(conn, data_root)
    docs = sorted(TEXTS)
    matrix = represent.load_features(conn, data_root, m1, "hubert", docs)
    assert matrix.shape[0] == N and np.isfinite(matrix).all()
