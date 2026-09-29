"""The E1 run in the database (SPEC-E1-02, -03, -06, -07, -08) on a synthetic gold chain."""

import hashlib
import json

import pytest
import yaml
from e1_helpers import GOLD, VALUE_FOR, doc_id, identical_observations, seed_c1, seed_l1, sha, write_gold_csv

from grantrisk.evaluation import validate
from grantrisk.evaluation.validate import E1InputError
from grantrisk.extraction.manual import gold
from grantrisk.labelling import consolidate, label
from grantrisk.labelling.scoring import FACTORS
from grantrisk.store import db, runs

CONFIG = {"extract": {"llm": {"model": "test-model"}}, "validate": {"old_baseline_renamed": {"Old name": "D-2.1.1-16"}}}
LLM = "llm:test-model"
A, D = doc_id("A Plusz-1.1.1-21"), doc_id("D-2.1.1-16")


@pytest.fixture
def data_root(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def conn(data_root):
    c = db.connect(data_root)
    yield c
    c.close()


@pytest.fixture
def chain(conn, data_root, tmp_path):
    """C1 → gold import → L2 (gold-only) → L3: the main L3 run over the six gold documents."""
    c1 = seed_c1(conn)
    pins = {name: sha(name) for name, _, _, _ in GOLD}
    manual = gold.run(conn, {}, data_root, c1_run_id=c1, gold_csv=write_gold_csv(tmp_path / "gold.csv"), gold_pins=pins)
    l2 = consolidate.run(conn, {}, data_root, manual_run_id=manual, gold_only=True)
    l3 = label.run(conn, {}, data_root, l2_run_id=l2, c1_run_id=c1)
    return {"c1": c1, "manual": manual, "l3": l3}


@pytest.fixture
def sources(conn):
    regex = identical_observations()
    regex[(A, "eloleg")] = VALUE_FOR["eloleg"][1]  # disagree
    regex[(D, "idotartam")] = {"status": "ambiguous"}  # not found
    llm = identical_observations()
    llm[(A, "fin_form")] = {"value": "grant", "evidence": "vissza nem térítendő", "page": 2,
                            "warnings": ["evidence_not_in_text"]}
    llm = {k: v for k, v in llm.items() if k[0] != D}  # the LLM failed on one document
    return {"regex": seed_l1(conn, "regex", regex), "llm": seed_l1(conn, LLM, llm)}


@pytest.fixture
def baseline(tmp_path):
    lines = ["document_id,felhivas,factor,gold,regex,e5,haiku"]
    for name, _, _, points in GOLD:
        written = "Old name" if name == "D-2.1.1-16" else name  # renamed since, like GINOP Plusz-2.2.1-24
        for f, p in zip(FACTORS, points):
            cell = "" if f == "tam_osszeg" else f"{float(p)}"  # the old regex never found an amount
            lines.append(f"1,{written},{f},{p}.0,{cell},,")
    path = tmp_path / "Gold_second_test_results_final.csv"
    path.write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_e1(conn, data_root, chain, sources, baseline=None, config=CONFIG):
    return validate.run(
        conn, config, data_root, manual_run_id=chain["manual"], extraction_run_ids=list(sources.values()),
        l3_run_id=chain["l3"], c1_run_id=chain["c1"], old_baseline_csv=baseline,
    )


def count(conn, table, run_id):
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)).fetchone()[0]


def test_run_writes_tables_report_and_lineage(conn, data_root, chain, sources, baseline):
    run_id = run_e1(conn, data_root, chain, sources, baseline)
    row = runs.get(conn, run_id)
    assert (row["stage"], row["status"]) == ("E1", "complete")
    assert runs.inputs(conn, run_id) == sorted([chain["manual"], chain["l3"], chain["c1"], *sources.values()])
    n_sources, n_docs = 3, len(GOLD)
    assert count(conn, "extraction_agreements", run_id) == n_sources * (len(FACTORS) + 1) * 3
    assert count(conn, "extraction_details", run_id) == n_sources * n_docs * len(FACTORS)
    assert count(conn, "extraction_labels", run_id) == (n_sources + 1) * n_docs
    assert count(conn, "extraction_label_agreements", run_id) == n_sources * 2

    def agreement(source, factor="all"):
        return conn.execute(
            "SELECT agree, disagree, not_found, not_comparable FROM extraction_agreements"
            " WHERE run_id = ? AND source = ? AND factor = ? AND subset = 'all'", (run_id, source, factor)
        ).fetchone()

    assert tuple(agreement("regex")) == (58, 1, 1, 0)
    assert tuple(agreement(LLM)) == (50, 0, 10, 0)  # the missing document: ten cells not found
    # The baseline found no amount, but the two TOP calls get the TOP rule's points (DEC-31).
    assert tuple(agreement("old_regex", "tam_osszeg")) == (2, 0, 4, 0)

    detail = conn.execute(
        "SELECT value_json, points, category, evidence, evidence_page, warnings_json FROM extraction_details"
        " WHERE run_id = ? AND source = ? AND doc_id = ? AND factor = 'fin_form'", (run_id, LLM, A)
    ).fetchone()
    assert tuple(detail) == ('"grant"', 3, "agree", "vissza nem térítendő", 2, '["evidence_not_in_text"]')

    report = json.loads((data_root / row["report_path"]).read_text(encoding="utf-8"))
    assert report["missing_documents"] == {LLM: [D]}
    assert report["evidence_audit"][LLM]["fin_form"]["evidence_not_in_text"] == 1
    assert report["sources"]["old_regex"]["sha256"] == hashlib.sha256(baseline.read_bytes()).hexdigest()
    folder = data_root / "reports" / run_id
    assert sorted(p.name for p in folder.iterdir()) == [
        "e1_agreement.csv", "e1_details.csv", "e1_labels.csv", "e1_report.json", "e1_report.md", "l2_preferences.yaml"
    ]


def test_markdown_report_has_every_source_breakdown_and_limitation(conn, data_root, chain, sources, baseline):
    run_id = run_e1(conn, data_root, chain, sources, baseline)
    text = (data_root / "reports" / run_id / "e1_report.md").read_text(encoding="utf-8")
    header = next(line for line in text.splitlines() if line.startswith("| Factor | `regex`"))
    assert header == f"| Factor | `regex` | `{LLM}` | `old_regex` |"  # SPEC-E1-02: one column per source
    for f in FACTORS:
        assert f"| {f} |" in text
    assert "**all factors**" in text
    assert "## 3. By period" in text and "| Source | all | 2021-2027 | other |" in text
    assert "## 4. Evidence audit" in text  # SPEC-E1-03
    assert "## 5. Label-level agreement" in text  # SPEC-E1-04
    assert "[0." in text  # SPEC-E1-05: Wilson intervals
    for heading in ("**In-sample:**", "**One expert:**", "**Small sample:**", "**L2 preferences:**",
                    "**What gold measures:**"):  # SPEC-E1-07
        assert heading in text
    assert "6 documents" in text


def test_suggested_preferences_run_l2_unchanged(conn, data_root, chain, sources):
    run_id = run_e1(conn, data_root, chain, sources)
    text = (data_root / "reports" / run_id / "l2_preferences.yaml").read_text(encoding="utf-8")
    pasted = yaml.safe_load(text)
    prefs = pasted["consolidate"]["preferred_source"]
    # regex: 1 disagree on eloleg, 1 not found on idotartam; LLM: the missing document on every factor.
    assert prefs == {f: "regex" for f in FACTORS}
    assert pasted["consolidate"]["preferences_from_e1_run"] == run_id
    l2 = consolidate.run(
        conn, pasted, data_root, manual_run_id=chain["manual"], regex_run_id=sources["regex"],
        llm_run_id=sources["llm"], gold_only=True,
    )
    l2_report = json.loads((data_root / runs.get(conn, l2)["report_path"]).read_text(encoding="utf-8"))
    assert l2_report["preferences_from_e1_run"] == run_id


def test_no_preferences_file_without_the_corpus_model(conn, data_root, chain, sources):
    run_id = run_e1(conn, data_root, chain, sources, config={})
    assert not (data_root / "reports" / run_id / "l2_preferences.yaml").exists()
    report = json.loads((data_root / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))
    assert report["suggested_preferences"]["preferred_source"] is None


def test_same_inputs_give_identical_reports(conn, data_root, chain, sources, baseline):
    a = run_e1(conn, data_root, chain, sources, baseline)
    b = run_e1(conn, data_root, chain, sources, baseline)
    for name in ("e1_report.json", "e1_report.md", "e1_agreement.csv", "e1_labels.csv", "e1_details.csv",
                 "l2_preferences.yaml"):
        text_a = (data_root / "reports" / a / name).read_text(encoding="utf-8").replace(a, "RUN")
        text_b = (data_root / "reports" / b / name).read_text(encoding="utf-8").replace(b, "RUN")
        assert text_a == text_b, name
    for table in ("extraction_agreements", "extraction_details", "extraction_labels", "extraction_label_agreements"):
        def rows(run_id):
            return sorted(tuple(r)[1:] for r in conn.execute(f"SELECT * FROM {table} WHERE run_id = ?", (run_id,)))

        assert rows(a) == rows(b), table


def test_failure_leaves_nothing(conn, data_root, chain, sources, monkeypatch):
    def boom(*args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(validate.runs, "complete", boom)
    with pytest.raises(RuntimeError, match="injected"):
        run_e1(conn, data_root, chain, sources)
    run = conn.execute("SELECT run_id, status FROM runs WHERE stage = 'E1'").fetchone()
    assert run["status"] == "failed"
    assert count(conn, "extraction_details", run["run_id"]) == 0
    assert not (data_root / "reports" / run["run_id"]).exists() or not any((data_root / "reports" / run["run_id"]).iterdir())


def test_manual_run_is_not_a_source(conn, data_root, chain, sources):
    with pytest.raises(E1InputError, match="manual import"):
        run_e1(conn, data_root, chain, {**sources, "manual": chain["manual"]})


def test_two_runs_of_one_source_are_rejected(conn, data_root, chain, sources):
    again = seed_l1(conn, "regex", identical_observations())
    with pytest.raises(E1InputError, match="both hold source regex"):
        run_e1(conn, data_root, chain, {**sources, "again": again})


def test_input_runs_must_be_complete_runs_of_their_stage(conn, data_root, chain, sources):
    with pytest.raises(ValueError, match="not L3"):
        validate.run(conn, CONFIG, data_root, manual_run_id=chain["manual"], extraction_run_ids=[],
                     l3_run_id=chain["c1"], c1_run_id=chain["c1"])
    with pytest.raises(E1InputError, match="not manual"):
        validate.run(conn, CONFIG, data_root, manual_run_id=sources["regex"], extraction_run_ids=[],
                     l3_run_id=chain["l3"], c1_run_id=chain["c1"])


def test_gold_document_missing_from_c1_run(conn, data_root, chain, sources):
    other_c1 = seed_c1(conn, rows=GOLD[:-1])
    with pytest.raises(E1InputError, match="no Document record"):
        validate.run(conn, CONFIG, data_root, manual_run_id=chain["manual"], extraction_run_ids=list(sources.values()),
                     l3_run_id=chain["l3"], c1_run_id=other_c1)
    assert conn.execute("SELECT status FROM runs WHERE stage = 'E1'").fetchone()[0] == "failed"


def test_baseline_with_an_unknown_call_is_rejected(conn, data_root, chain, sources, baseline):
    with pytest.raises(E1InputError, match="Old name: not a call of the gold set"):
        run_e1(conn, data_root, chain, sources, baseline, config={})


def test_baseline_only(conn, data_root, chain, baseline):
    run_id = run_e1(conn, data_root, chain, {}, baseline)
    sources = {r[0] for r in conn.execute("SELECT DISTINCT source FROM extraction_agreements WHERE run_id = ?", (run_id,))}
    assert sources == {"old_regex"}
    assert runs.inputs(conn, run_id) == sorted([chain["manual"], chain["l3"], chain["c1"]])
