"""The regex extractor as a stage run: lineage, contract and document set (SPEC-L1-01, -03, -05, -14)."""

import json

import pytest
from test_gold_import import fake_sha, seed_c1

from grantrisk.extraction.regex import run as regex
from grantrisk.extraction.regex.rules import RULES_VERSION
from grantrisk.labelling import consolidate, scoring
from grantrisk.labelling.scoring import FACTORS
from grantrisk.store import db, runs
from grantrisk.store.db import transaction

TEXT = (
    "<!-- page 1 -->\n"
    "|Nyújthat be támogatási kérelmet konzorcium?|A támogatási kérelem benyújtására konzorciumi formában nincs lehetőség.|\n"
    "|Mennyi támogatást lehet igényelni?|minimum 5 000 000 Ft – maximum 400 000 000 Ft|\n"
    "<!-- page 2 -->\n"
    "A támogatás maximális mértéke az Európai Unióval elszámolható összköltség 50%-a.\n"
)


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "data")
    yield c
    c.close()


def seed_c2(conn, texts, failed=(), complete=True):
    """A C2 run with the given {name: markdown} texts and failed documents; returns (run id, doc ids)."""
    names = [*texts, *failed]
    c1 = seed_c1(conn, [fake_sha(n) for n in names])
    c2 = runs.start(conn, "C2", {"test": True}, inputs=[c1])
    ids = {n: fake_sha(n)[:16] for n in names}
    with transaction(conn):
        for name in names:
            ok = name in texts
            conn.execute(
                "INSERT INTO document_texts (run_id, doc_id, status, error, markdown, converter, converter_version,"
                " settings_hash, cleaning_version) VALUES (?, ?, ?, ?, ?, 'fake', '1', 'x', '1')",
                (c2, ids[name], "ok" if ok else "failed", None if ok else "broken PDF", texts.get(name)),
            )
        if complete:
            runs.complete(conn, c2)
    return c2, ids


def observations(conn, run_id):
    return conn.execute("SELECT * FROM factor_observations WHERE run_id = ? ORDER BY doc_id, factor", (run_id,)).fetchall()


def test_run_stores_ten_regex_observations_per_document(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": TEXT, "B": "<!-- page 1 -->\nsemmi\n"})
    run_id = regex.run(conn, {"test": True}, tmp_path / "data", c2_run_id=c2)
    rows = observations(conn, run_id)
    assert len(rows) == 20
    assert {r["source"] for r in rows} == {"regex"}
    a = {r["factor"]: r for r in rows if r["doc_id"] == ids["A"]}
    assert json.loads(a["konzorcium"]["value_json"]) == 0
    assert json.loads(a["tam_osszeg"]["value_json"]) == 400_000_000
    assert (json.loads(a["max_tam_int"]["value_json"]), a["max_tam_int"]["evidence_page"]) == (50, 2)
    assert all(r["points"] is None and r["prompt_version"] is None for r in a.values())


def test_output_contract(conn, tmp_path):
    """SPEC-L1-01: one observation per factor; a value in the domain or NULL; found exactly when not NULL."""
    c2, _ = seed_c2(conn, {"A": TEXT, "B": "<!-- page 1 -->\nsemmi\n"})
    run_id = regex.run(conn, {}, tmp_path / "data", c2_run_id=c2)
    by_doc = {}
    for r in observations(conn, run_id):
        by_doc.setdefault(r["doc_id"], []).append(r["factor"])
        value = None if r["value_json"] is None else json.loads(r["value_json"])
        assert (r["status"] == "found") == (value is not None)
        if value is not None:
            assert scoring.domain_error(r["factor"], value) is None
            assert r["evidence"] or "default_value" in json.loads(r["warnings_json"])
        else:
            assert r["evidence"] is None
    assert all(sorted(fs) == sorted(FACTORS) for fs in by_doc.values())


def test_lineage_report_and_skipped_documents(conn, tmp_path):
    """SPEC-L1-03: a failed C2 document is listed as skipped."""
    c2, ids = seed_c2(conn, {"A": TEXT}, failed=["F"])
    run_id = regex.run(conn, {"test": True}, tmp_path / "data", c2_run_id=c2)
    row = runs.get(conn, run_id)
    assert (row["stage"], row["status"]) == ("L1", "complete")
    assert runs.inputs(conn, run_id) == [c2]
    assert json.loads(row["config_json"])["regex_run"]["rules_version"] == RULES_VERSION
    report = json.loads((tmp_path / "data" / row["report_path"]).read_text(encoding="utf-8"))
    assert report["documents"] == 1
    assert report["skipped"] == [{"doc_id": ids["F"], "reason": "C2 status failed: broken PDF"}]
    assert report["status_by_factor"]["konzorcium"] == {"found": 1}
    assert report["evidence_not_in_text_rate"] == 0
    assert {r["doc_id"] for r in observations(conn, run_id)} == {ids["A"]}


def test_document_filter(conn, tmp_path):
    """SPEC-L1-14: a filter processes exactly the named documents."""
    c2, ids = seed_c2(conn, {"A": TEXT, "B": TEXT, "C": TEXT})
    run_id = regex.run(conn, {}, tmp_path / "data", c2_run_id=c2, doc_ids=[ids["A"], ids["C"]])
    assert {r["doc_id"] for r in observations(conn, run_id)} == {ids["A"], ids["C"]}


def test_needs_a_complete_c2_run(conn, tmp_path):
    c2, _ = seed_c2(conn, {"A": TEXT}, complete=False)
    with pytest.raises(ValueError, match="not complete"):
        regex.run(conn, {}, tmp_path / "data", c2_run_id=c2)


def test_runs_are_deterministic(conn, tmp_path):
    """ARC-06: two runs on the same texts give identical observations."""
    c2, _ = seed_c2(conn, {"A": TEXT, "B": TEXT.replace("50%", "80%")})
    first = regex.run(conn, {}, tmp_path / "data", c2_run_id=c2)
    second = regex.run(conn, {}, tmp_path / "data", c2_run_id=c2)
    assert [tuple(r)[1:] for r in observations(conn, first)] == [tuple(r)[1:] for r in observations(conn, second)]


def test_l2_accepts_the_regex_run(conn, tmp_path):
    c2, _ = seed_c2(conn, {"A": TEXT})
    run_id = regex.run(conn, {}, tmp_path / "data", c2_run_id=c2)
    consolidate._check_source(conn, run_id, "regex")  # raises L2InputError otherwise
