"""E4: the explanatory analyses of the representations (DEC-64, DEC-69), on synthetic data."""

import json

import numpy as np
import pytest
from modelling_helpers import FakeEmbedder, seed_c2, seed_l3, synthetic_corpus

from grantrisk.evaluation import evaluate, explain
from grantrisk.evaluation.evaluate import Prediction
from grantrisk.evaluation.explain import FoldValue, Inputs, Settings
from grantrisk.modelling import classifiers, represent, train
from grantrisk.modelling.text import plain_text
from grantrisk.modelling.train import CVSettings
from grantrisk.store import db, runs
from grantrisk.store.db import transaction

TEXTS, LABEL_OF = synthetic_corpus(n_per_class=10)
DOCS = sorted(TEXTS)
Y = np.array([LABEL_OF[d] for d in DOCS])
X_TEXT = [plain_text(TEXTS[d]) for d in DOCS]
RNG = np.random.default_rng(0)
NOISE = RNG.normal(0, 1, (len(DOCS), 4))  # carries nothing
POINTS = {"low": 0, "medium": 1, "high": 3}
PARAMS = {c: classifiers.params(c) for c in ("logreg", "svm")}
TFIDF = {"min_df": 1}


def inputs(**kw):
    splits = train.stratified_splits(list(Y), CVSettings(n_repeats=2))
    base = dict(doc_ids=DOCS, y=Y, splits=splits, features={"tfidf": X_TEXT, "noise": NOISE},
                points={"fin_form": {d: POINTS[LABEL_OF[d]] for d in DOCS},
                        "konzorcium": {d: 0 for d in DOCS}},  # one point value only: not probed
                scores=np.array([POINTS[LABEL_OF[d]] + RNG.normal(0, 0.1) for d in DOCS]))
    return Inputs(**{**base, **kw})


def settings(**kw):
    return Settings(**{"probe_repeats": 2, **kw})


def test_settings_of_the_configuration():
    s = Settings.of({"analyses": ["error_overlap"], "probe_cv": {"n_repeats": 3}, "learning_curve_shares": [0.5, 1]})
    assert s.analyses == ("error_overlap",) and s.probe_repeats == 3 and s.learning_curve_shares == (0.5, 1.0)
    assert Settings.of(None, ["combination"]).analyses == ("combination",)
    with pytest.raises(ValueError, match="unknown E4 analyses"):
        Settings.of({"analyses": ["probes"]})
    with pytest.raises(ValueError, match="shares"):
        Settings.of({"learning_curve_shares": [0, 1]})


def test_factor_probes_find_what_a_representation_carries():
    """DEC-64 no. 1: TF-IDF carries the class words that set fin_form here; the noise does not."""
    values, notes = explain.factor_probes(inputs(), settings(), PARAMS, TFIDF)
    summary = {(x["representation"], x["target"], x["metric"]): x["mean"] for x in explain.summarise(values)}
    assert summary[("tfidf", "fin_form", "f1_macro")] > 0.9
    assert summary[("noise", "fin_form", "f1_macro")] < 0.6
    assert summary[("majority", "fin_form", "balanced_accuracy")] == pytest.approx(1 / 3)
    assert any(n.startswith("konzorcium: 30 documents with 1 point value") for n in notes)
    assert {v.target for v in values} == {"fin_form"}
    assert len([v for v in values if v.representation == "tfidf" and v.metric == "f1_macro"]) == 10  # 5 × 2 folds


def test_only_probed_origins_are_read(tmp_path):
    """Imputed (mean) and rule-set points are left out of the probes."""
    data_root = tmp_path / "data"
    conn = db.connect(data_root)
    chain = build_chain(conn, data_root)
    s = Settings.of(None)
    loaded, _ = explain.load_inputs(conn, data_root, chain["e2"], s)
    assert set(loaded.points["fin_form"]) == set(DOCS)
    assert "biztositek" not in loaded.points  # origin 'mean'
    conn.close()


def test_score_probe_and_combination_on_the_folds_of_m2():
    s = settings(analyses=("factor_probe", "combination"))
    values = explain.m2_fold_work(inputs(), inputs().splits[0], s, PARAMS, TFIDF)
    keys = {(v.analysis, v.representation, v.classifier, v.metric) for v in values}
    assert keys == {("factor_probe", "tfidf", "ridge", "spearman"), ("factor_probe", "noise", "ridge", "spearman"),
                    ("combination", "tfidf", "logreg", "f1_macro"), ("combination", "tfidf", "svm", "f1_macro"),
                    ("combination", "tfidf+noise", "logreg", "f1_macro"),
                    ("combination", "tfidf+noise", "svm", "f1_macro")}
    spearman = {v.representation: v.value for v in values if v.metric == "spearman"}
    assert spearman["tfidf"] > 0.6
    assert all(v.ratio == pytest.approx(6 / 24) for v in values)


def test_embedding_block_has_unit_expected_squared_norm():
    """DEC-69: the standardised embedding is scaled by 1/sqrt(d), like an L2-normalised TF-IDF row."""
    E = RNG.normal(5, 3, (200, 64))
    a, _ = explain._block(E, E[:10])
    assert np.mean(np.sum(a * a, axis=1)) == pytest.approx(1.0)


def test_learning_curve_trains_on_stratified_shares():
    s = settings(learning_curve_shares=(0.5, 1.0))
    split = inputs().splits[0]
    values = explain.learning_curve_fold(inputs(), split, s, PARAMS, TFIDF)
    assert {(v.representation, v.setting) for v in values} == {("tfidf", "0.5"), ("tfidf", "1"),
                                                               ("noise", "0.5"), ("noise", "1")}
    half = next(v for v in values if v.setting == "0.5")
    assert half.ratio == pytest.approx(len(split.test) / 12)


def test_mcnemar_exact():
    assert explain.mcnemar_exact(0, 0) == 1.0
    assert explain.mcnemar_exact(0, 5) == pytest.approx(2 * 0.5 ** 5)
    assert explain.mcnemar_exact(1, 9) == pytest.approx(2 * 11 / 1024)
    assert explain.mcnemar_exact(4, 4) == 1.0


def test_modal_label_tie_goes_up():
    assert explain.modal_label(["low", "low", "high"]) == "low"
    assert explain.modal_label(["low", "medium"]) == "medium"


def prediction(doc, repeat, true, predicted):
    return Prediction(doc, repeat, 0, true, predicted, (1 / 3, 1 / 3, 1 / 3))


def test_error_overlap_counts_and_mcnemar():
    """DEC-64 no. 4: TF-IDF is always right; the embedding gets four documents wrong in every repeat."""
    docs = [f"d{i}" for i in range(10)]
    wrong = {"d0", "d1", "d2", "d3"}
    preds = {("stratified", "tfidf", "logreg"): [prediction(d, r, "low", "low") for d in docs for r in range(3)],
             ("stratified", "e5", "logreg"): [prediction(d, r, "low", "high" if d in wrong else "low")
                                              for d in docs for r in range(3)],
             ("stratified", "e5", "majority"): []}
    (o,) = explain.error_overlap(preds)
    assert (o["representation"], o["classifier"], o["documents"]) == ("e5", "logreg", 10)
    assert o["modal"] == {"both_wrong": 0, "only_embedding_wrong": 4, "only_tfidf_wrong": 0, "both_right": 6}
    assert o["p_value"] == pytest.approx(2 * 0.5 ** 4)
    assert [r["only_embedding_wrong"] for r in o["per_repeat"]] == [4, 4, 4]
    assert all(r["overlap"] == 0.0 for r in o["per_repeat"])


def test_context_length_and_holm_within_each_analysis():
    """DEC-64 no. 5: bge_m3_512 minus bge_m3 per classifier; Holm runs within one analysis only."""
    fold_f1 = {("bge_m3_512", c): {(r, k): 0.6 + 0.01 * k for r in range(2) for k in range(5)} for c in ("logreg", "svm")}
    fold_f1.update({("bge_m3", c): {(r, k): 0.5 + 0.02 * k for r in range(2) for k in range(5)} for c in ("logreg", "svm")})
    fold_f1[("bge_m3", "majority")] = {(0, 0): 0.2}
    folds = explain.context_length(fold_f1, 0.25)
    assert {(f.representation, f.classifier) for f in folds} == {
        ("bge_m3_512", "logreg"), ("bge_m3_512", "svm"), ("bge_m3", "logreg"), ("bge_m3", "svm")}
    folds.append(FoldValue("combination", "tfidf+e5", "logreg", "tercile_label", "", 0, 0, "f1_macro", 0.5, 0.25))
    tests = explain.tests(folds, [])
    context = [t for t in tests if t["analysis"] == "context_length"]
    assert [(t["representation"], t["reference"], t["classifier"]) for t in context] == [
        ("bge_m3_512", "bge_m3", "logreg"), ("bge_m3_512", "bge_m3", "svm")]
    assert context[0]["mean_diff"] == pytest.approx(0.1 - 0.01 * 2)
    assert context[0]["p_holm"] == pytest.approx(min(1.0, 2 * min(t["p_value"] for t in context)))


# --- The run over a real chain ---------------------------------------------------------------


def build_chain(conn, data_root):
    """C2 → M1 (TF-IDF and a fake huBERT) → L3 with factor points → M2 → E2."""
    c2 = seed_c2(conn, TEXTS)
    config = {"represent": {"representations": ["tfidf", "hubert"]}, "train": {"classifier_params": {"rf": {"n_estimators": 10}}}}
    m1 = represent.run(conn, config, data_root, c2_run_id=c2, embedders={"hubert": FakeEmbedder()})
    l3 = seed_l3(conn, LABEL_OF)
    with transaction(conn):
        for d in DOCS:
            p = POINTS[LABEL_OF[d]]
            conn.execute("UPDATE risk_labels SET normalised_score = ? WHERE run_id = ? AND doc_id = ?", (p, l3, d))
            conn.execute("INSERT INTO risk_label_factors VALUES (?, ?, 'fin_form', ?, ?, 'band')", (l3, d, p, str(p)))
            conn.execute("INSERT INTO risk_label_factors VALUES (?, ?, 'biztositek', 1.5, '3/2', 'mean')", (l3, d))
    m2 = train.run(conn, config, data_root, m1_run_id=m1, l3_run_id=l3, classifiers=["logreg", "svm"])
    e2 = evaluate.run(conn, config, data_root, m2_run_id=m2)
    return {"c2": c2, "m1": m1, "l3": l3, "m2": m2, "e2": e2, "config": config}


def test_run_writes_results_tests_and_report(tmp_path):
    data_root = tmp_path / "data"
    conn = db.connect(data_root)
    chain = build_chain(conn, data_root)
    config = {**chain["config"], "explain": {"probe_cv": {"n_repeats": 1}, "learning_curve_shares": [0.5, 1.0],
                                             "analyses": ["factor_probe", "combination", "learning_curve",
                                                          "error_overlap", "context_length"]}}
    run_id = explain.run(conn, config, data_root, e2_run_id=chain["e2"])
    run = runs.get(conn, run_id)
    assert run["status"] == "complete" and run["stage"] == "E4"
    assert runs.inputs(conn, run_id) == sorted([chain["e2"], chain["m2"], chain["m1"], chain["l3"]])
    analyses = {r[0] for r in conn.execute("SELECT DISTINCT analysis FROM explain_results WHERE run_id = ?", (run_id,))}
    assert analyses == {"factor_probe", "combination", "learning_curve"}
    tests = conn.execute("SELECT * FROM explain_tests WHERE run_id = ?", (run_id,)).fetchall()
    kinds = {(t["analysis"], t["representation"], t["reference"], t["classifier"]) for t in tests}
    assert ("combination", "tfidf+hubert", "tfidf", "logreg") in kinds
    assert ("error_overlap", "hubert", "tfidf", "svm") in kinds
    assert ("factor_probe", "hubert", "tfidf", "logreg") in kinds  # fin_form
    assert all(t["p_value"] <= t["p_holm"] <= 1 for t in tests)
    report = (data_root / f"reports/{run_id}/e4_report.md").read_text(encoding="utf-8")
    for heading in ("## 1. Factor probes", "## 2. TF-IDF combined", "## 3. Learning curve", "## 4. Error overlap",
                    "## 5. Context length"):
        assert heading in report
    assert "lacks bge_m3_512 or bge_m3" in report
    folds = (data_root / f"reports/{run_id}/e4_folds.csv").read_text(encoding="utf-8").splitlines()
    assert folds[0].startswith("analysis,representation,classifier")
    snapshot = json.loads(run["config_json"])["e4_run"]
    assert snapshot["probe_repeats"] == 1 and snapshot["classifier_params"]["svm"]["C"] == 1.0
    conn.close()


def test_run_needs_a_complete_e2_run(tmp_path):
    conn = db.connect(tmp_path / "data")
    with pytest.raises(ValueError):
        explain.run(conn, {}, tmp_path / "data", e2_run_id="E2-missing")
    conn.close()
