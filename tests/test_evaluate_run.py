"""The E2 run in the database (SPEC-E2-02, -05, -06, -10, -11)."""

import json

import pytest
from e2_helpers import LABELS, corpus, default_models, majority, noisy, rows, seed_l3, seed_m2
from modelling_helpers import FakeEmbedder, seed_c2, synthetic_corpus
from modelling_helpers import seed_l3 as seed_l3_labels

from grantrisk.evaluation import evaluate, metrics
from grantrisk.evaluation.evaluate import E2InputError
from grantrisk.modelling import represent, train
from grantrisk.store import db, runs

DOCS = corpus(10)
TABLES = ("metrics", "error_sizes", "roc_curves", "model_comparison", "model_grid", "significance",
          "label_distributions")


@pytest.fixture
def data_root(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def conn(data_root):
    c = db.connect(data_root)
    yield c
    c.close()


@pytest.fixture
def chain(conn):
    c1, l3 = seed_l3(conn, DOCS)
    return {"c1": c1, "l3": l3, "m2": seed_m2(conn, l3, DOCS, default_models())}


def count(conn, sql, *args):
    return conn.execute(sql, args).fetchone()[0]


def test_run_writes_every_scope(conn, data_root, chain):
    """SPEC-E2-02 acceptance: 25 fold rows and 5 repeat rows per metric, plus the summary."""
    run_id = evaluate.run(conn, {}, data_root, m2_run_id=chain["m2"])
    assert runs.get(conn, run_id)["status"] == "complete"
    assert runs.inputs(conn, run_id) == sorted([chain["m2"], chain["l3"], chain["c1"]])
    for rep, clf in default_models():
        key = (run_id, "stratified", rep, clf)
        base = "SELECT COUNT(*) FROM metrics WHERE run_id = ? AND scheme = ? AND representation = ? AND classifier = ?"
        assert count(conn, base + " AND name = 'f1_macro' AND scope LIKE 'fold:%'", *key) == 25
        assert count(conn, base + " AND name = 'f1_macro' AND scope LIKE 'repeat:%'", *key) == 5
        for name in ("f1_macro:fold_mean", "f1_macro:fold_sd", "f1_macro:repeat_mean", "f1_macro:repeat_sd"):
            assert count(conn, base + " AND scope = 'summary' AND name = ?", *key, name) == 1
        # SPEC-E2-03 acceptance: the confusion matrix sums to the number of documents.
        total = count(conn, "SELECT SUM(value) FROM metrics WHERE run_id = ? AND scheme = ? AND representation = ?"
                            " AND classifier = ? AND scope LIKE 'class:%' AND name LIKE 'confusion:%'", *key)
        assert total == pytest.approx(len(DOCS))
        assert count(conn, "SELECT COUNT(*) FROM error_sizes WHERE run_id = ? AND scheme = ? AND representation = ?"
                           " AND classifier = ?", *key) == 5
        assert count(conn, "SELECT COUNT(*) FROM roc_curves WHERE run_id = ? AND scheme = ? AND representation = ?"
                           " AND classifier = ?", *key) == 4 * metrics.ROC_GRID
        periods = count(conn, "SELECT SUM(value) FROM metrics WHERE run_id = ? AND scheme = ? AND representation = ?"
                              " AND classifier = ? AND scope LIKE 'period:%' AND name = 'documents'", *key)
        assert periods == len(DOCS)  # SPEC-E2-08 acceptance


def test_comparison_table_and_best(conn, data_root, chain):
    """SPEC-E2-05, -06 acceptance: one row per model run with the baseline, and a best run."""
    run_id = evaluate.run(conn, {}, data_root, m2_run_id=chain["m2"])
    table = conn.execute("SELECT * FROM model_comparison WHERE run_id = ? ORDER BY rank", (run_id,)).fetchall()
    assert len(table) == len(default_models())
    assert {(r["representation"], r["classifier"]) for r in table if r["is_baseline"]} == {
        ("tfidf", "majority"), ("hubert", "majority")}
    best = [r for r in table if r["is_best"]]
    assert [(r["representation"], r["classifier"]) for r in best] == [("tfidf", "logreg")]
    assert best[0]["f1_macro_mean"] == pytest.approx(count(
        conn, "SELECT value FROM metrics WHERE run_id = ? AND representation = 'tfidf' AND classifier = 'logreg'"
              " AND scope = 'summary' AND name = 'f1_macro:fold_mean'", run_id))
    grid = conn.execute("SELECT axis, key, n_models FROM model_grid WHERE run_id = ? ORDER BY 1, 2", (run_id,)).fetchall()
    assert [tuple(g) for g in grid] == [("classifier", "logreg", 2), ("classifier", "majority", 2),
                                        ("classifier", "svm", 2), ("representation", "hubert", 2),
                                        ("representation", "tfidf", 2)]
    assert count(conn, "SELECT COUNT(*) FROM significance WHERE run_id = ?", run_id) == len(default_models()) - 1
    report = json.loads((data_root / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))
    assert report["metrics_version"] == metrics.METRICS_VERSION
    assert report["best"] == {"stratified": ["tfidf/logreg"]}
    assert report["input_runs"] == {"M2": chain["m2"], "L3": chain["l3"], "C1": chain["c1"]}
    assert (data_root / f"reports/{run_id}/e2_report.md").exists()
    snapshot = json.loads(runs.get(conn, run_id)["config_json"])
    assert snapshot["e2_run"]["metrics_version"] == metrics.METRICS_VERSION  # SPEC-E2-11


def test_label_distributions_are_stored(conn, data_root, chain):
    """SPEC-E2-07 acceptance: the counts add up to the number of documents, overall and over the periods."""
    run_id = evaluate.run(conn, {}, data_root, m2_run_id=chain["m2"])
    assert count(conn, "SELECT SUM(n) FROM label_distributions WHERE run_id = ? AND subset = 'all'", run_id) == len(DOCS)
    assert count(conn, "SELECT SUM(n) FROM label_distributions WHERE run_id = ? AND subset != 'all'", run_id) == len(DOCS)
    assert count(conn, "SELECT COUNT(*) FROM label_distributions WHERE run_id = ?", run_id) == 5 * 9


def test_deterministic(conn, data_root, chain):
    """SPEC-E2-11 acceptance: two runs give identical rows, apart from the run id."""
    a = evaluate.run(conn, {}, data_root, m2_run_id=chain["m2"])
    b = evaluate.run(conn, {}, data_root, m2_run_id=chain["m2"])
    for table in TABLES:
        assert rows(conn, table, a) == rows(conn, table, b), table


def test_failed_model_run_is_skipped(conn, data_root):
    c1, l3 = seed_l3(conn, DOCS)
    m2 = seed_m2(conn, l3, DOCS, {("tfidf", "logreg"): noisy(0.8, 1), ("tfidf", "svm"): noisy(0.7, 2),
                                  ("tfidf", "majority"): majority()}, failed={("tfidf", "svm")})
    run_id = evaluate.run(conn, {}, data_root, m2_run_id=m2)
    assert count(conn, "SELECT COUNT(*) FROM model_comparison WHERE run_id = ?", run_id) == 2
    report = json.loads((data_root / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))
    assert report["failed_in_m2"] == [["stratified", "tfidf", "svm"]]


def test_grouped_scheme_is_reported_separately(conn, data_root):
    """SPEC-E2-10 acceptance: the two schemes are never mixed in one row."""
    c1, l3 = seed_l3(conn, DOCS)
    m2 = runs.start(conn, "M2", {"test": True}, inputs=[l3])
    seed_m2(conn, l3, DOCS, {("tfidf", "logreg"): noisy(0.8, 1), ("tfidf", "majority"): majority()}, m2=m2)
    seed_m2(conn, l3, DOCS, {("tfidf", "logreg"): noisy(0.6, 7)}, scheme="grouped", m2=m2)
    runs.complete(conn, m2)
    run_id = evaluate.run(conn, {}, data_root, m2_run_id=m2)
    table = conn.execute("SELECT scheme, representation, classifier, is_best FROM model_comparison WHERE run_id = ?"
                         " ORDER BY scheme, rank", (run_id,)).fetchall()
    assert [tuple(r) for r in table] == [("grouped", "tfidf", "logreg", 0), ("stratified", "tfidf", "logreg", 1),
                                         ("stratified", "tfidf", "majority", 0)]
    grouped = count(conn, "SELECT value FROM metrics WHERE run_id = ? AND scheme = 'grouped' AND scope = 'summary'"
                          " AND name = 'f1_macro:fold_mean'", run_id)
    standard = count(conn, "SELECT value FROM metrics WHERE run_id = ? AND scheme = 'stratified' AND scope = 'summary'"
                           " AND classifier = 'logreg' AND name = 'f1_macro:fold_mean'", run_id)
    assert grouped != standard


def test_labels_must_match_the_l3_run(conn, data_root):
    """The true labels of M2 must be the tercile labels of its L3 run; otherwise nothing is written."""
    c1, l3 = seed_l3(conn, DOCS)
    m2 = seed_m2(conn, l3, DOCS, {("tfidf", "logreg"): noisy(0.8, 1)})
    conn.execute("UPDATE risk_labels SET tercile_label = 'low' WHERE run_id = ? AND doc_id = 'doc029'", (l3,))
    with pytest.raises(E2InputError, match="true labels differ"):
        evaluate.run(conn, {}, data_root, m2_run_id=m2)
    failed = conn.execute("SELECT run_id, status FROM runs WHERE stage = 'E2'").fetchone()
    assert failed["status"] == "failed"
    for table in TABLES:
        assert count(conn, f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", failed["run_id"]) == 0
    assert not (data_root / "reports" / failed["run_id"]).exists() or not any(
        (data_root / "reports" / failed["run_id"]).iterdir())


def test_needs_a_complete_m2_run(conn, data_root):
    c1, l3 = seed_l3(conn, DOCS)
    m2 = runs.start(conn, "M2", {"test": True}, inputs=[l3])
    with pytest.raises(ValueError, match="not complete"):
        evaluate.run(conn, {}, data_root, m2_run_id=m2)
    with pytest.raises(ValueError, match="does not exist"):
        evaluate.run(conn, {}, data_root, m2_run_id="M2-nothing")


def test_on_a_real_m2_run(conn, data_root):
    """E2 over an M2 run of the real code (TF-IDF with logistic regression, and the baseline)."""
    texts, labels = synthetic_corpus(n_per_class=10)
    c2 = seed_c2(conn, texts)
    config = {"represent": {"representations": ["tfidf"]}}
    m1 = represent.run(conn, config, data_root, c2_run_id=c2, embedders={"hubert": FakeEmbedder()})
    l3 = seed_l3_labels(conn, labels)
    m2 = train.run(conn, config, data_root, m1_run_id=m1, l3_run_id=l3, classifiers=["logreg"])
    run_id = evaluate.run(conn, config, data_root, m2_run_id=m2)
    table = {(r["classifier"]): r for r in conn.execute(
        "SELECT * FROM model_comparison WHERE run_id = ?", (run_id,))}
    assert set(table) == {"logreg", "majority"}
    assert table["logreg"]["is_best"] == 1
    assert table["logreg"]["f1_macro_mean"] > table["majority"]["f1_macro_mean"]
    assert table["logreg"]["not_above_baseline"] == 0
    # Each class's predictions of one repeat equal the M2 predictions it was computed from.
    n = count(conn, "SELECT COUNT(*) FROM predictions WHERE run_id = ? AND classifier = 'logreg' AND repeat = 0"
                    " AND predicted_label = true_label", m2)
    acc = count(conn, "SELECT value FROM metrics WHERE run_id = ? AND classifier = 'logreg' AND scope = 'repeat:0'"
                      " AND name = 'accuracy'", run_id)
    assert acc == pytest.approx(n / len(labels))
    assert set(LABELS) == {r[0] for r in conn.execute(
        "SELECT DISTINCT curve FROM roc_curves WHERE run_id = ? AND curve != 'macro'", (run_id,))}


def test_embeddings_against_tfidf_are_stored_and_reported(conn, data_root, chain):
    """DEC-63: one row per transformer model run; the report states the hyperparameters and the verdict."""
    run_id = evaluate.run(conn, {}, data_root, m2_run_id=chain["m2"])
    stored = conn.execute("SELECT * FROM tfidf_comparisons WHERE run_id = ? ORDER BY representation, classifier",
                          (run_id,)).fetchall()
    assert [(r["representation"], r["classifier"]) for r in stored] == [("hubert", "logreg"), ("hubert", "svm")]
    assert all(r["in_family"] == 1 and r["outcome"] and r["ci_low"] <= r["mean_diff"] <= r["ci_high"] for r in stored)
    report = (data_root / f"reports/{run_id}/e2_report.md").read_text(encoding="utf-8")
    assert "## Embeddings against TF-IDF (DEC-63)" in report
    assert "the fixed defaults of SPEC-M2-04 (DEC-63 (a))" in report  # the seeded M2 run records no tuning
    assert "H1 (INT-RQ-B) is" in report
    snapshot = json.loads(runs.get(conn, run_id)["config_json"])["e2_run"]["tfidf_comparison"]
    assert snapshot["margin"] == 0.02 and snapshot["m2_tuning"] is None
