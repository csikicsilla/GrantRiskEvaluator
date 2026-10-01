"""The E3 run (SPEC-E3-01 … -06) on a whole chain built with the real stages: C1 … L3, M1, M2, E1, E2."""

import csv
import io
import json
import random
import re

import pytest
from e1_helpers import GOLD, VALUE_FOR, doc_id, identical_observations, seed_c1, seed_l1, sha, write_gold_csv
from modelling_helpers import FakeEmbedder, seed_c2

from grantrisk.evaluation import evaluate, validate
from grantrisk.extraction.manual import gold
from grantrisk.labelling import consolidate, label
from grantrisk.labelling.scoring import FACTORS
from grantrisk.modelling import represent, train
from grantrisk.reporting import appendix, report
from grantrisk.reporting.chain import E3InputError
from grantrisk.store import db, runs

# Nine more calls with totals 2 … 11 (not 6), so that the 15 totals are distinct: terciles 5 / 5 / 5.
EXTRA_TOTALS = [2, 3, 4, 5, 7, 8, 9, 10, 11]
EXTRA_PROGRAMMES = [("GINOP", "2014-2020"), ("RRF", "RRF"), ("VP", "VP"), ("GINOP_PLUSZ", "2021-2027"),
                    ("EFOP", "2014-2020")]
EXTRA = [(f"Á{i}-{t}.1.1-20", *EXTRA_PROGRAMMES[i % 5], None) for i, t in enumerate(EXTRA_TOTALS)]
SPREAD = ("tam_osszeg", "bead_napok", "max_tam_int", "eloleg", "idotartam")
BASE_POINTS = {"fin_form": 1, "konzorcium": 0, "tam_tevekenyseg": 0, "egysz_elszam": 0, "biztositek": 0}
FILLER = "pályázat támogatás összeg határidő kedvezményezett projekt előleg biztosíték konzorcium felhívás".split()
CONFIG = {
    "extract": {"llm": {"model": "test-model"}},
    "consolidate": {"preferred_source": {f: "regex" for f in FACTORS}},
    "represent": {"representations": ["tfidf", "hubert"]},
    "train": {"classifier_params": {"rf": {"n_estimators": 10}}},
}
TODAY = "2030-01-01"  # a creation date that cannot collide with the runs' own timestamps


def extra_points(total):
    points = dict(BASE_POINTS)
    rest = total - 1
    for f in SPREAD:
        points[f] = min(3, rest)
        rest -= points[f]
    return points


def extra_observations():
    out = {}
    for (name, _, _, _), total in zip(EXTRA, EXTRA_TOTALS):
        for f, p in extra_points(total).items():
            out[(doc_id(name), f)] = VALUE_FOR[f][p]
    return out


def text_for(i):
    """A plain text with no class signal: the models are near chance, so some documents are misclassified."""
    rng = random.Random(i)
    words = [rng.choice(FILLER) for _ in range(40)]
    return f"<!-- page 1 -->\n# Felhívás {i}\n\n{' '.join(words)}\n"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e3")
    data_root = tmp / "data"
    conn = db.connect(data_root)
    rows = GOLD + EXTRA
    c1 = seed_c1(conn, rows)
    pins = {name: sha(name) for name, _, _, _ in GOLD}
    manual = gold.run(conn, {}, data_root, c1_run_id=c1, gold_csv=write_gold_csv(tmp / "gold.csv"), gold_pins=pins)
    c2 = seed_c2(conn, {doc_id(name): text_for(i) for i, (name, _, _, _) in enumerate(rows)})
    regex_obs = {**identical_observations(), **extra_observations()}
    llm_obs = dict(regex_obs)
    first = doc_id(EXTRA[0][0])
    llm_obs[(first, "eloleg")] = VALUE_FOR["eloleg"][3]  # regex and the LLM disagree here (SPEC-L2-05)
    regex = seed_l1(conn, "regex", regex_obs)
    llm = seed_l1(conn, "llm:test-model", llm_obs)
    l2 = consolidate.run(conn, CONFIG, data_root, manual_run_id=manual, regex_run_id=regex, llm_run_id=llm, c2_run_id=c2)
    l3 = label.run(conn, CONFIG, data_root, l2_run_id=l2, c1_run_id=c1)
    m1 = represent.run(conn, CONFIG, data_root, c2_run_id=c2, embedders={"hubert": FakeEmbedder()})
    m2 = train.run(conn, CONFIG, data_root, m1_run_id=m1, l3_run_id=l3, classifiers=["logreg"])
    e1 = validate.run(conn, CONFIG, data_root, manual_run_id=manual, extraction_run_ids=[regex, llm], l3_run_id=l3,
                      c1_run_id=c1)
    e2 = evaluate.run(conn, CONFIG, data_root, m2_run_id=m2)
    e3 = report.run(conn, CONFIG, data_root, e2_run_id=e2, e1_run_id=e1, today=TODAY)
    ids = {"c1": c1, "c2": c2, "manual": manual, "regex": regex, "llm": llm, "l2": l2, "l3": l3, "m1": m1, "m2": m2,
           "e1": e1, "e2": e2, "e3": e3, "first_extra": first}
    yield conn, data_root, ids
    conn.close()


def folder(data_root, run_id):
    return data_root / "reports" / run_id


def read(data_root, run_id, name, encoding="utf-8"):
    return (folder(data_root, run_id) / name).read_text(encoding=encoding)


def test_run_and_files(built):
    conn, data_root, ids = built
    run = runs.get(conn, ids["e3"])
    assert run["status"] == "complete"
    chain = [ids[k] for k in ("c1", "c2", "manual", "regex", "llm", "l2", "l3", "m1", "m2", "e1", "e2")]
    assert runs.inputs(conn, ids["e3"]) == sorted(chain)
    report_json = json.loads((data_root / run["report_path"]).read_text(encoding="utf-8"))
    for name in ("dashboard.html", "risk_dataset.csv", "tables/model_comparison.csv", "tables/model_comparison.md",
                 "tables/model_grid.md", "tables/label_distributions.md", "tables/error_sizes.md",
                 "tables/gold_validation.md", "tables/gold_label_agreement.md", "tables/period_breakdown.md",
                 "tables/significance.md", "tables/tfidf_comparison.md", "error_analysis/error_analysis.md", "error_analysis/misclassified.csv",
                 "error_analysis/error_rates_by_group.csv", "appendix/8_1_scoring_table.md",
                 "appendix/8_2_er_diagram.mmd", "appendix/8_3_reproduction.md", "appendix/8_4_generative_ai.md",
                 "appendix/data_flow.mmd", "PROVENANCE.md"):
        assert name in report_json["files"]
        assert (folder(data_root, ids["e3"]) / name).exists(), name
    assert report_json["best_model_run"].startswith("stratified/")
    assert report_json["documents"] == 15


def risk_rows(data_root, run_id):
    raw = (folder(data_root, run_id) / "risk_dataset.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 with BOM, for Excel
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter=";"))


def test_risk_dataset(built):
    """SPEC-E3-02 acceptance: one row per document of the L3 run, accented letters intact."""
    conn, data_root, ids = built
    rows = risk_rows(data_root, ids["e3"])
    l3 = {r["doc_id"]: r for r in conn.execute("SELECT * FROM risk_labels WHERE run_id = ?", (ids["l3"],))}
    assert sorted(r["doc_id"] for r in rows) == sorted(l3)
    assert any("Á" in r["call_code"] for r in rows)
    for r in rows:
        assert r["tercile_label"] == l3[r["doc_id"]]["tercile_label"]
        assert float(r["normalised_score"].replace(",", ".")) == pytest.approx(l3[r["doc_id"]]["normalised_score"], abs=1e-4)
        assert r["predicted_label"] in ("low", "medium", "high") and r["repeats"] == "5"
        for f in FACTORS:
            assert r[f"{f}_origin"] in ("band", "manual", "mean", "top_rule", "loan_rule")
    scores = [float(r["normalised_score"].replace(",", ".")) for r in rows]
    assert scores == sorted(scores, reverse=True)  # a ranked list (INT-GOAL-03)
    gold_doc = next(r for r in rows if r["doc_id"] == doc_id(GOLD[0][0]))
    assert gold_doc["fin_form_source"] == "manual" and gold_doc["fin_form_origin"] == "manual"


def test_thesis_tables_trace_to_stored_metrics(built):
    """SPEC-E3-03 acceptance: every number in an export can be traced to a stored metric of the named run."""
    conn, data_root, ids = built
    table = list(csv.DictReader(io.StringIO(read(data_root, ids["e3"], "tables/model_comparison.csv"))))
    assert len(table) == conn.execute("SELECT COUNT(*) FROM model_comparison WHERE run_id = ?", (ids["e2"],)).fetchone()[0]
    for row in table:
        stored = conn.execute(
            "SELECT value FROM metrics WHERE run_id = ? AND scheme = ? AND representation = ? AND classifier = ?"
            " AND scope = 'summary' AND name = 'f1_macro:fold_mean'",
            (ids["e2"], row["scheme"], row["representation"], row["classifier"])).fetchone()[0]
        assert float(row["f1_macro_mean"]) == pytest.approx(round(stored, 4))
    md = read(data_root, ids["e3"], "tables/model_comparison.md")
    assert "majority" in md and "**best**" in md and ids["e2"] in md
    gold_md = read(data_root, ids["e3"], "tables/gold_validation.md")
    assert "`regex`" in gold_md and "`llm:test-model`" in gold_md and "`old_regex`" not in gold_md


def test_dashboard_is_self_contained(built):
    """SPEC-E3-01: one file, no network, every view has its data; the model runs differ in their details."""
    _, data_root, ids = built
    page = read(data_root, ids["e3"], "dashboard.html")
    assert not re.search(r"https?://", page)
    assert "<script src" not in page and "<link" not in page and "@import" not in page
    payload = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S).group(1))
    models = payload["models"]
    assert {r["key"] for r in models["rows"]} == set(models["details"])
    assert any(r["baseline"] for r in models["rows"]) and sum(r["best"] for r in models["rows"]) >= 1
    a, b = models["details"]["stratified/tfidf/logreg"], models["details"]["stratified/tfidf/majority"]
    assert a["roc"] != b["roc"] and a["fold_f1"] != b["fold_f1"]
    assert payload["extraction"]["sources"] == ["regex", "llm:test-model"]
    assert set(payload["coverage"]) == set(FACTORS)
    assert payload["label_distributions"]["all"]
    assert {r["label"] for r in payload["runs"]} >= {"C1", "L2", "L3", "M2", "E1", "E2"}
    for view in ("models", "extraction", "labels", "coverage", "provenance"):
        assert f'<section id="{view}"' in page


def test_error_analysis_matches_the_risk_dataset(built):
    """SPEC-E3-04 acceptance: every misclassified document is in the risk dataset with the same values."""
    conn, data_root, ids = built
    risk = {r["doc_id"]: r for r in risk_rows(data_root, ids["e3"])}
    raw = read(data_root, ids["e3"], "error_analysis/misclassified.csv", "utf-8-sig")
    wrong = list(csv.DictReader(io.StringIO(raw), delimiter=";"))
    assert wrong, "the near-chance models misclassify some documents"
    for w in wrong:
        r = risk[w["doc_id"]]
        for column in ("call_code", "tercile_label", "predicted_label", "n_determined", "low_coverage",
                       "normalised_score", *[f"{f}_{x}" for f in FACTORS for x in ("points", "origin")]):
            assert w[column] == r[column], column
        assert int(w["repeats_wrong"]) >= 3
        assert int(w["repeats_wrong"]) == 5 - int(r["repeats_correct"])
    md = read(data_root, ids["e3"], "error_analysis/error_analysis.md")
    for heading in ("## 1. Misclassified documents", "## 2. Error rates by group", "## 3. Propagation",
                    "## 4. Words that drive the model"):
        assert heading in md
    rates = list(csv.DictReader(io.StringIO(read(data_root, ids["e3"], "error_analysis/error_rates_by_group.csv"))))
    by_kind = {}
    for r in rates:
        by_kind.setdefault(r["kind"], 0)
        by_kind[r["kind"]] += int(r["documents"])
    assert by_kind == {"low_coverage": 15, "imputed_factors": 15, "period": 15, "programme": 15}


def test_regex_llm_disagreement_is_shown(built):
    conn, data_root, ids = built
    data_mod = __import__("grantrisk.reporting.chain", fromlist=["chain"])
    chain = data_mod.resolve(conn, e2_run_id=ids["e2"], e1_run_id=ids["e1"])
    data = data_mod.load(conn, data_root, chain)
    from grantrisk.reporting import tables

    assert tables.disagreeing_factors(data, ids["first_extra"]) == ["eloleg"]
    assert tables.disagreeing_factors(data, doc_id(GOLD[0][0])) == []


def test_appendix(built):
    """SPEC-E3-05."""
    conn, data_root, ids = built
    scoring_md = read(data_root, ids["e3"], "appendix/8_1_scoring_table.md")
    assert "Finanszírozási forma" in scoring_md and "Form of financing" in scoring_md
    er = read(data_root, ids["e3"], "appendix/8_2_er_diagram.mmd")
    assert er.startswith("erDiagram")
    for table in ("runs", "documents", "predictions", "metrics", "extraction_agreements"):
        assert f"    {table} {{" in er
    assert 'runs ||--o{ documents : "run_id"' in er
    assert "%% Provenance" in er
    genai = read(data_root, ids["e3"], "appendix/8_4_generative_ai.md")
    assert ids["llm"] in genai and "test-model" in genai
    assert "No embedding of this chain was computed by a hosted service." in genai
    repro = read(data_root, ids["e3"], "appendix/8_3_reproduction.md")
    assert f"--m2-run {ids['m2']}" in repro and f"--l2-run {ids['l2']}" in repro
    flow = read(data_root, ids["e3"], "appendix/data_flow.mmd")
    assert flow.startswith("flowchart TD") and ids["m2"] in flow
    assert "L3 -- factor means, tercile cuts --> E1" in flow and "E1 -. preferred source per factor .-> L2" in flow


# Spec_L3_ScoreAndLabel.md Appendix A.1: (factor, value, expected points).
A1 = [
    ("fin_form", "grant", 3), ("fin_form", "conditional_grant", 2), ("fin_form", "loan", 1),
    *[("tam_osszeg", v, p) for v, p in [(0, 0), (100_000_000, 0), (100_000_001, 1), (300_000_000, 1),
                                         (300_000_001, 2), (600_000_000, 2), (600_000_001, 3)]],
    ("konzorcium", 0, 0), ("konzorcium", 1, 3),
    *[("bead_napok", v, p) for v, p in [(0, 3), (7, 3), (8, 2), (9, 2), (15, 2), (16, 1), (30, 1), (31, 0),
                                         ("keret_kimerulesig", 0)]],
    *[("max_tam_int", v, p) for v, p in [(0, 0), (30, 0), (30.01, 1), (50, 1), (50.01, 2), (69, 2), (69.5, 2),
                                          (70, 3), (100, 3)]],
    *[("eloleg", v, p) for v, p in [(0, 0), (30, 0), (30.01, 1), (50, 1), (50.01, 2), (70, 2), (70.01, 3), (100, 3)]],
    *[("idotartam", v, p) for v, p in [(1, 0), (12, 0), (12.1, 1), (18, 1), (18.1, 2), (24, 2), (24.1, 3)]],
    ("tam_tevekenyseg", ["egyeb"], 0), ("tam_tevekenyseg", ["kutatas_fejlesztes"], 2),
    ("tam_tevekenyseg", ["infrastruktura_ingatlan"], 3),
    ("tam_tevekenyseg", ["kutatas_fejlesztes", "infrastruktura_ingatlan"], 3),
    ("tam_tevekenyseg", ["egyeb", "kutatas_fejlesztes"], 2),
    ("egysz_elszam", 1, 0), ("egysz_elszam", 0, 3), ("biztositek", 1, 0), ("biztositek", 0, 3),
]


@pytest.mark.parametrize("factor,value,expected", A1)
def test_scoring_table_matches_appendix_a1(factor, value, expected):
    """SPEC-E3-05 acceptance: the scoring table of 8.1 matches the test vectors of SPEC-L3 Appendix A.1."""
    band = appendix.band_of(factor, value)
    assert band is not None and band.points == expected


def test_scoring_table_is_checked_against_the_scoring_module(monkeypatch):
    appendix.check_bands()
    wrong = dict(appendix.BANDS)
    wrong["eloleg"] = [appendix.Band(1, "x", "x", appendix.F(0), True, appendix.F(30), True), *wrong["eloleg"][1:]]
    monkeypatch.setattr(appendix, "BANDS", wrong)
    with pytest.raises(appendix.AppendixError, match="eloleg"):
        appendix.check_bands()


def test_two_builds_differ_only_in_the_date(built):
    """SPEC-E3-06 acceptance."""
    conn, data_root, ids = built
    other = report.run(conn, CONFIG, data_root, e2_run_id=ids["e2"], e1_run_id=ids["e1"], today="2030-12-31")
    a, b = folder(data_root, ids["e3"]), folder(data_root, other)
    names = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file() and p.name != "e3_report.json")
    assert names == sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file() and p.name != "e3_report.json")
    for name in names:
        x = (a / name).read_bytes().replace(TODAY.encode(), b"DATE")
        y = (b / name).read_bytes().replace(b"2030-12-31", b"DATE")
        assert x == y, name
    assert ids["e3"] not in (b / "dashboard.html").read_text(encoding="utf-8")


def test_every_artefact_carries_its_provenance(built):
    """SPEC-E3-06: the runs and the code version, at the end of each Markdown, Mermaid and HTML file."""
    _, data_root, ids = built
    base = folder(data_root, ids["e3"])
    code_version = runs.get(built[0], ids["e3"])["code_version"]
    for path in base.rglob("*"):
        if path.suffix in (".md", ".mmd", ".html"):
            tail = path.read_text(encoding="utf-8")[-1500:]
            assert "rovenance" in tail and code_version in tail and TODAY in tail, path.name
    index = (base / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "`risk_dataset.csv`" in index and ids["l3"] in index


def test_report_without_models(built):
    """§4: a chain without M1, M2, E1 and E2 gives the artefacts without those parts, and says so."""
    conn, data_root, ids = built
    run_id = report.run(conn, CONFIG, data_root, l3_run_id=ids["l3"], today=TODAY)
    rep = json.loads((data_root / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))
    assert "E2 (evaluate models)" in rep["missing"] and "E1 (validate extraction)" in rep["missing"]
    assert rep["best_model_run"] is None
    assert "tables/model_comparison.csv" not in rep["files"]
    assert "Not available" in read(data_root, run_id, "tables/model_comparison.md")
    assert "Not run." in read(data_root, run_id, "tables/significance.md")
    rows = risk_rows(data_root, run_id)
    assert len(rows) == 15 and all(r["predicted_label"] == "" for r in rows)
    page = read(data_root, run_id, "dashboard.html")
    payload = json.loads(re.search(r'id="data">(.*?)</script>', page, re.S).group(1))
    assert payload["models"] is None and payload["extraction"] is None


def test_chain_comes_from_the_configuration(built):
    conn, data_root, ids = built
    config = {**CONFIG, "report": {"chain": {"e2_run": ids["e2"], "e1_run": None, "l3_run": None}}}
    run_id = report.run(conn, config, data_root, today=TODAY)
    assert ids["e2"] in runs.inputs(conn, run_id) and ids["e1"] not in runs.inputs(conn, run_id)


def test_chain_errors(built):
    conn, data_root, ids = built
    with pytest.raises(E3InputError, match="name an E2 run"):
        report.run(conn, {}, data_root)
    with pytest.raises(E3InputError, match="evaluated L3 run"):
        report.run(conn, {}, data_root, e2_run_id=ids["e2"], l3_run_id=ids["l2"])
    with pytest.raises(ValueError, match="is a L2 run"):
        report.run(conn, {}, data_root, l3_run_id=ids["l2"])
    assert conn.execute("SELECT COUNT(*) FROM runs WHERE stage = 'E3' AND status = 'running'").fetchone()[0] == 0


# --- DEC-70: the E2 run beside and the E4 run --------------------------------------------------


@pytest.fixture(scope="module")
def beside(built):
    """A tuned M2 run on the chain's M1 and L3 runs, its E2 run and an E4 run on it; the chain's E2 run is untuned."""
    from grantrisk.evaluation import explain

    conn, data_root, ids = built
    config = {**CONFIG, "train": {**CONFIG["train"], "tuning": {"enabled": True, "C": [0.1, 10.0]}},
              "explain": {"analyses": ["factor_probe", "combination", "error_overlap", "context_length"],
                          "probe_cv": {"n_repeats": 1}, "combination_classifiers": ["logreg"]}}
    m2 = train.run(conn, config, data_root, m1_run_id=ids["m1"], l3_run_id=ids["l3"], classifiers=["logreg"])
    e2 = evaluate.run(conn, config, data_root, m2_run_id=m2)
    e4 = explain.run(conn, config, data_root, e2_run_id=e2)
    e3 = report.run(conn, config, data_root, e2_run_id=e2, e1_run_id=ids["e1"], e2_beside_run_id=ids["e2"],
                    e4_run_id=e4, today=TODAY)
    return {"m2": m2, "e2": e2, "e4": e4, "e3": e3}


def test_report_with_the_defaults_beside_and_e4(built, beside):
    conn, data_root, ids = built
    run = runs.get(conn, beside["e3"])
    rep = json.loads((data_root / run["report_path"]).read_text(encoding="utf-8"))
    assert rep["input_runs"]["E2"] == beside["e2"] and rep["input_runs"]["M2"] == beside["m2"]
    assert rep["input_runs"]["E2 beside"] == ids["e2"] and rep["input_runs"]["M2 beside"] == ids["m2"]
    assert rep["input_runs"]["E4"] == beside["e4"] and "E4 (explanatory analyses)" not in rep["missing"]
    for name in ("tables/tuning.md", "tables/tuning.csv", "tables/e4_factor_probes.md", "tables/e4_factor_probes.csv",
                 "tables/e4_combination.md", "tables/e4_error_overlap.md", "tables/e4_tests.csv"):
        assert name in rep["files"], name
    comparison = read(data_root, beside["e3"], "tables/tfidf_comparison.md")
    assert f"## Beside it: E2 `{ids['e2']}`" in comparison
    assert "tuned inside each training fold" in comparison and "the fixed defaults of SPEC-M2-04" in comparison
    rows = list(csv.DictReader(io.StringIO(read(data_root, beside["e3"], "tables/tfidf_comparison.csv"))))
    assert {r["hyperparameters"] for r in rows} == {"tuned", "defaults"}
    tuning = read(data_root, beside["e3"], "tables/tuning.md")
    assert "| tfidf | logreg |" in tuning and "×" in tuning  # the chosen C per fold
    assert "Not run: the E4 run did not run `learning_curve`." in read(data_root, beside["e3"],
                                                                        "tables/e4_learning_curve.md")
    assert "Not compared" in read(data_root, beside["e3"], "tables/e4_context_length.md")
    probes = read(data_root, beside["e3"], "tables/e4_factor_probes.md")
    assert "## Macro-F1" in probes and "Exploratory (DEC-64)" in probes and f"E4 `{beside['e4']}`" in probes
    dashboard = read(data_root, beside["e3"], "dashboard.html")
    assert 'id="representations"' in dashboard and '"probes":{' in dashboard
    reproduction = read(data_root, beside["e3"], "appendix/8_3_reproduction.md")
    assert f"--l3-run {ids['l3']} --no-tuning" in reproduction  # the M2 run beside used the defaults
    assert f"explain --e2-run {beside['e2']}" in reproduction and f"--e4-run {beside['e4']}" in reproduction
    assert 'E4["E4 Explain' in read(data_root, beside["e3"], "appendix/data_flow.mmd")


def test_the_runs_beside_must_belong_to_the_chain(built, beside):
    conn, data_root, ids = built
    with pytest.raises(E3InputError, match="not in this chain"):
        report.run(conn, CONFIG, data_root, e2_run_id=ids["e2"], e4_run_id=beside["e4"], today=TODAY)
    with pytest.raises(E3InputError, match="needs the E2 run"):
        report.run(conn, CONFIG, data_root, l3_run_id=ids["l3"], e4_run_id=beside["e4"], today=TODAY)
