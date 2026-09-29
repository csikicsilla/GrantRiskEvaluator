"""L2 against Spec_L2_Consolidate.md (SPEC-L2-01 … -06) and the gold-only mode of DEC-35."""

import json

import pytest
from test_gold_import import fake_sha, gold_names, seed_c1

from grantrisk.extraction.manual import gold
from grantrisk.labelling import consolidate, label
from grantrisk.labelling.consolidate import L2InputError, Observation
from grantrisk.labelling.scoring import FACTORS
from grantrisk.store import db, runs
from grantrisk.store.db import transaction

PREFS = {f: "llm" for f in FACTORS} | {"bead_napok": "regex"}


def found(source, value=None, points=None):
    return Observation(source=source, status="found", value=value, points=points, evidence=f"{source} says so", evidence_page=3)


def missing(source):
    return Observation(source=source, status="not_found")


def one_doc(factor, manual=None, regex=None, llm=None):
    m = {("D", factor): manual} if manual else {}
    a = {"regex": {("D", factor): regex} if regex else {}, "llm": {("D", factor): llm} if llm else {}}
    rows = consolidate.consolidate(["D"], m, a, PREFS)
    return {r.factor: r for r in rows}[factor]


# --- SPEC-L2-02: one test per step -----------------------------------------------------


def test_step_manual():
    r = one_doc("eloleg", manual=found("manual", points=2), regex=found("regex", 40), llm=found("llm:m", 60))
    assert (r.rule, r.chosen_source, r.points, r.value) == ("manual", "manual", 2, None)


def test_step_preferred():
    r = one_doc("eloleg", manual=missing("manual"), regex=found("regex", 40), llm=found("llm:m", 60))
    assert (r.rule, r.chosen_source, r.value) == ("preferred", "llm:m", 60)
    assert (r.evidence, r.evidence_page) == ("llm:m says so", 3)


def test_step_fallback():
    r = one_doc("eloleg", regex=found("regex", 40), llm=missing("llm:m"))
    assert (r.rule, r.chosen_source, r.value) == ("fallback", "regex", 40)


def test_step_none():
    r = one_doc("eloleg", regex=missing("regex"))
    assert (r.rule, r.chosen_source, r.value, r.points) == ("none", None, None, None)


def test_preference_per_factor():
    r = one_doc("bead_napok", regex=found("regex", 20), llm=found("llm:m", 10))
    assert (r.rule, r.chosen_source) == ("preferred", "regex")


def test_empty_activity_list_falls_back():
    r = one_doc("tam_tevekenyseg", llm=found("llm:m", []), regex=found("regex", ["egyeb"]))
    assert (r.rule, r.chosen_source, r.value) == ("fallback", "regex", ["egyeb"])


def test_exactly_ten_rows_per_document():
    rows = consolidate.consolidate(["A", "B"], {}, {}, PREFS)
    assert len(rows) == 20
    assert {r.rule for r in rows} == {"none"}


def test_disagreements():
    automated = {
        "regex": {("A", "eloleg"): found("regex", 40), ("B", "eloleg"): found("regex", 40), ("C", "eloleg"): found("regex", 40)},
        "llm": {("A", "eloleg"): found("llm:m", 45), ("B", "eloleg"): found("llm:m", 60), ("C", "eloleg"): missing("llm:m")},
    }
    result = consolidate.disagreements(["A", "B", "C"], automated)
    assert result["eloleg"] == {"both_found": 2, "differ": 1}  # 40 and 45 are both 1 point


def test_preferences_must_cover_every_factor():
    with pytest.raises(L2InputError, match="idotartam"):
        consolidate.check_preferences({f: "llm" for f in FACTORS if f != "idotartam"})
    with pytest.raises(L2InputError, match="sonnet"):
        consolidate.check_preferences({**PREFS, "eloleg": "sonnet"})


# --- Database runs -----------------------------------------------------------------------


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "data")
    yield c
    c.close()


@pytest.fixture
def chain(conn, tmp_path, gold_csv):
    """A C1 run with the 42 gold documents and their manual import."""
    pins = {n: fake_sha(n) for n in gold_names(gold_csv)}
    c1 = seed_c1(conn, pins.values())
    manual = gold.run(conn, {}, tmp_path / "data", c1_run_id=c1, gold_csv=gold_csv, gold_pins=pins)
    return c1, manual


def seed_automated(conn, source, observations):
    run_id = runs.start(conn, "L1", {"test": True})
    with transaction(conn):
        for (doc_id, factor), value in observations.items():
            conn.execute(
                "INSERT INTO factor_observations (run_id, doc_id, factor, source, value_json, status)"
                " VALUES (?, ?, ?, ?, ?, 'found')",
                (run_id, doc_id, factor, source, json.dumps(value)),
            )
        runs.complete(conn, run_id)
    return run_id


def test_gold_only_run_and_l3_on_top(conn, tmp_path, chain):
    c1, manual = chain
    l2 = consolidate.run(conn, {}, tmp_path / "data", manual_run_id=manual, gold_only=True)
    n = conn.execute("SELECT COUNT(*) FROM consolidated_factors WHERE run_id = ?", (l2,)).fetchone()[0]
    assert n == 42 * 10
    assert runs.inputs(conn, l2) == [manual]
    rules = {r[0] for r in conn.execute("SELECT DISTINCT rule FROM consolidated_factors WHERE run_id = ?", (l2,))}
    assert rules == {"manual"}
    report = json.loads((tmp_path / "data" / runs.get(conn, l2)["report_path"]).read_text(encoding="utf-8"))
    assert report["document_set"] == {"mode": "gold_only", "from_run": manual, "documents": 42}
    # The first end-to-end chain: L3 labels the gold documents from the expert's points.
    l3 = label.run(conn, {"label": {"low_coverage_threshold": 5}}, tmp_path / "data", l2_run_id=l2, c1_run_id=c1)
    origins = {r[0] for r in conn.execute("SELECT DISTINCT origin FROM risk_label_factors WHERE run_id = ?", (l3,))}
    assert origins == {"manual"}
    assert conn.execute("SELECT COUNT(*) FROM risk_labels WHERE run_id = ?", (l3,)).fetchone()[0] == 42


def test_automated_runs_need_preferences(conn, tmp_path, chain):
    _, manual = chain
    regex = seed_automated(conn, "regex", {("D", "eloleg"): 40})
    with pytest.raises(L2InputError, match="preferred_source"):
        consolidate.run(conn, {}, tmp_path / "data", manual_run_id=manual, regex_run_id=regex, gold_only=True)


def test_values_are_copied_unchanged(conn, tmp_path, chain):
    _, manual = chain
    doc = conn.execute("SELECT doc_id FROM factor_observations WHERE run_id = ? LIMIT 1", (manual,)).fetchone()[0]
    conn.execute("UPDATE factor_observations SET status = 'not_found', points = NULL WHERE run_id = ? AND doc_id = ? AND factor = 'eloleg'", (manual, doc))
    regex = seed_automated(conn, "regex", {(doc, "eloleg"): 30.01})
    llm = seed_automated(conn, "llm:test-model", {(doc, "eloleg"): 60})  # 2 points against 1
    cfg = {"consolidate": {"preferred_source": {f: "regex" for f in FACTORS}}}
    l2 = consolidate.run(conn, cfg, tmp_path / "data", manual_run_id=manual, regex_run_id=regex, llm_run_id=llm, gold_only=True)
    row = conn.execute(
        "SELECT value_json, chosen_source, rule FROM consolidated_factors WHERE run_id = ? AND doc_id = ? AND factor = 'eloleg'",
        (l2, doc),
    ).fetchone()
    assert tuple(row) == ("30.01", "regex", "preferred")  # a number, not the string "30.01"
    report = json.loads((tmp_path / "data" / runs.get(conn, l2)["report_path"]).read_text(encoding="utf-8"))
    assert report["disagreement_regex_llm"]["eloleg"] == {"both_found": 1, "differ": 1}
    assert len(report["documents_missing_from"]["llm"]) == 41


def test_document_set_must_be_named(conn, tmp_path, chain):
    _, manual = chain
    with pytest.raises(L2InputError, match="either a C2 run or the gold-only mode"):
        consolidate.run(conn, {}, tmp_path / "data", manual_run_id=manual)
    with pytest.raises(L2InputError, match="C2 is not implemented"):
        consolidate.run(conn, {}, tmp_path / "data", manual_run_id=manual, c2_run_id="C2-x")


def test_manual_run_must_hold_manual_observations(conn, tmp_path, chain):
    regex = seed_automated(conn, "regex", {("D", "eloleg"): 40})
    with pytest.raises(L2InputError, match="not manual"):
        consolidate.run(conn, {}, tmp_path / "data", manual_run_id=regex, gold_only=True)


def test_same_input_same_rows(conn, tmp_path, chain):
    _, manual = chain
    a = consolidate.run(conn, {}, tmp_path / "data", manual_run_id=manual, gold_only=True)
    b = consolidate.run(conn, {}, tmp_path / "data", manual_run_id=manual, gold_only=True)

    def rows(run_id):
        return [tuple(r)[1:] for r in conn.execute("SELECT * FROM consolidated_factors WHERE run_id = ? ORDER BY doc_id, factor", (run_id,))]

    assert rows(a) == rows(b)


def test_failure_leaves_no_rows(conn, tmp_path, chain, monkeypatch):
    _, manual = chain

    def boom(*args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(consolidate.runs, "complete", boom)
    with pytest.raises(RuntimeError, match="injected"):
        consolidate.run(conn, {}, tmp_path / "data", manual_run_id=manual, gold_only=True)
    assert conn.execute("SELECT COUNT(*) FROM consolidated_factors").fetchone()[0] == 0
    assert conn.execute("SELECT status FROM runs WHERE stage = 'L2'").fetchone()[0] == "failed"
