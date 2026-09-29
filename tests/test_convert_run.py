"""The C2 stage run (SPEC-C2-01, -08, -09, -10) with a fake converter in one process."""

import json

import pytest
from test_gold_import import fake_sha, gold_names, seed_c1

from grantrisk.corpus import convert, converters
from grantrisk.extraction.manual import gold
from grantrisk.labelling import consolidate
from grantrisk.store import db, runs

PAGE = "a támogatás összege " * 60


class FakeConverter:
    name = "fake"
    supports_ocr = False
    calls: list[str] = []

    def version(self):
        return "fake 1.0"

    def settings(self):
        return {}

    def convert(self, pdf_path, ocr=False):
        FakeConverter.calls.append(str(pdf_path))
        if "BROKEN" in str(pdf_path):
            raise RuntimeError("password protected")
        return [f"{pdf_path} page {i} " + PAGE for i in range(3)]


CONFIG = {"convert": {"converter": "fake", "workers": 1, "ocr_min_chars_per_page": 500}}


@pytest.fixture(autouse=True)
def fake_converter(monkeypatch):
    monkeypatch.setitem(converters.ADAPTERS, "fake", FakeConverter)
    monkeypatch.setattr(convert, "_ADAPTER_CACHE", {})
    FakeConverter.calls = []


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "data")
    yield c
    c.close()


def seed(conn, names=("A", "B", "C")):
    c1 = seed_c1(conn, [fake_sha(n) for n in names])
    conn.execute("UPDATE documents SET page_count = 3, file_path = 'pdf/' || doc_id || '.pdf' WHERE run_id = ?", (c1,))
    return c1


def texts(conn, run_id):
    return {r["doc_id"]: r for r in conn.execute("SELECT * FROM document_texts WHERE run_id = ?", (run_id,))}


def test_run_converts_every_document(conn, tmp_path):
    c1 = seed(conn)
    run_id = convert.run(conn, CONFIG, tmp_path / "data", c1_run_id=c1)
    rows = texts(conn, run_id)
    assert len(rows) == 3
    assert all(r["status"] == "ok" and r["page_count"] == 3 and r["markdown"].count("<!-- page ") == 3 for r in rows.values())
    assert runs.get(conn, run_id)["status"] == "complete"
    assert runs.inputs(conn, run_id) == [c1]
    assert all((tmp_path / "data" / r["raw_path"]).exists() for r in rows.values())
    report = json.loads((tmp_path / "data" / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))
    assert (report["documents"], report["ok"], report["failed"]) == (3, 3, [])


def test_a_failed_document_does_not_stop_the_run(conn, tmp_path):
    c1 = seed(conn)
    broken = fake_sha("B")[:16]
    conn.execute("UPDATE documents SET file_path = 'pdf/BROKEN.pdf' WHERE doc_id = ?", (broken,))
    run_id = convert.run(conn, CONFIG, tmp_path / "data", c1_run_id=c1)
    rows = texts(conn, run_id)
    assert rows[broken]["status"] == "failed"
    assert "password protected" in rows[broken]["error"]
    assert runs.get(conn, run_id)["status"] == "complete"


def test_a_new_run_reuses_the_raw_output(conn, tmp_path):
    c1 = seed(conn)
    first = convert.run(conn, CONFIG, tmp_path / "data", c1_run_id=c1)
    FakeConverter.calls = []
    second = convert.run(conn, CONFIG, tmp_path / "data", c1_run_id=c1)
    assert FakeConverter.calls == []  # nothing converted again
    a, b = texts(conn, first), texts(conn, second)
    assert all(b[d]["raw_reused"] == 1 and b[d]["markdown"] == a[d]["markdown"] for d in a)


def test_an_interrupted_run_can_be_resumed(conn, tmp_path, monkeypatch):
    c1 = seed(conn)
    real = convert.convert_one
    calls = []

    def interrupted(task):
        calls.append(task["doc_id"])
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(task)

    monkeypatch.setattr(convert, "convert_one", interrupted)
    with pytest.raises(KeyboardInterrupt):
        convert.run(conn, CONFIG, tmp_path / "data", c1_run_id=c1)
    run_id = conn.execute("SELECT run_id FROM runs WHERE stage = 'C2'").fetchone()[0]
    assert runs.get(conn, run_id)["status"] == "failed"
    assert len(texts(conn, run_id)) == 1

    monkeypatch.setattr(convert, "convert_one", real)
    FakeConverter.calls = []
    assert convert.run(conn, CONFIG, tmp_path / "data", c1_run_id=c1, resume_run_id=run_id) == run_id
    assert len(FakeConverter.calls) == 2  # only the two missing documents
    assert len(texts(conn, run_id)) == 3
    assert runs.get(conn, run_id)["status"] == "complete"


def test_resume_needs_an_unfinished_c2_run(conn, tmp_path):
    c1 = seed(conn)
    done = convert.run(conn, CONFIG, tmp_path / "data", c1_run_id=c1)
    with pytest.raises(ValueError, match="not an unfinished C2 run"):
        convert.run(conn, CONFIG, tmp_path / "data", c1_run_id=c1, resume_run_id=done)


def test_the_converter_must_be_configured(conn, tmp_path):
    c1 = seed(conn)
    with pytest.raises(ValueError, match="DEC-30"):
        convert.run(conn, {"convert": {}}, tmp_path / "data", c1_run_id=c1)


def test_l2_takes_its_documents_from_the_c2_run(conn, tmp_path, gold_csv):
    names = gold_names(gold_csv)
    pins = {n: fake_sha(n) for n in names}
    c1 = seed(conn, names)
    broken = fake_sha(names[0])[:16]
    conn.execute("UPDATE documents SET file_path = 'pdf/BROKEN.pdf' WHERE doc_id = ?", (broken,))
    c2 = convert.run(conn, CONFIG, tmp_path / "data", c1_run_id=c1)
    manual = gold.run(conn, {}, tmp_path / "data", c1_run_id=c1, gold_csv=gold_csv, gold_pins=pins)
    l2 = consolidate.run(conn, {}, tmp_path / "data", manual_run_id=manual, c2_run_id=c2)
    docs = {r[0] for r in conn.execute("SELECT DISTINCT doc_id FROM consolidated_factors WHERE run_id = ?", (l2,))}
    assert len(docs) == 41 and broken not in docs  # the failed conversion is not in the document set
    assert c2 in runs.inputs(conn, l2)
