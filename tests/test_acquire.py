"""The C1 run on a small synthetic snapshot (SPEC-C1-01 … -11)."""

import csv
import hashlib
import json
import sqlite3
import zipfile

import pytest
import yaml

from grantrisk.corpus import acquire, callcode
from grantrisk.store import db, runs


def make_pdf(*lines: str) -> bytes:
    """A minimal one-page PDF with the given ASCII text lines."""
    content = "BT /F1 12 Tf 72 720 Td " + " ".join(f"({t}) Tj 0 -14 Td" for t in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        " /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


# (tender code, programme, portal file name, content). Content None means: same bytes as the entry before.
SCRAPED = [
    ("GINOP_PLUSZ-2.2.1-25", "GINOP_PLUSZ", "ginop-plusz-221-25-felhivas.pdf", make_pdf("scraped 221-25")),
    ("GINOP_PLUSZ-1.4.3-24", "GINOP_PLUSZ", "felhivas-v9.pdf", make_pdf("143 v9")),
    ("GINOP_PLUSZ-1.4.3-24", "GINOP_PLUSZ", "felhivas-v10.pdf", make_pdf("143 v10")),
    ("GINOP_PLUSZ-1.4.3-24", "GINOP_PLUSZ", "altalanos-utmutato.pdf", make_pdf("guide")),
    ("GINOP-8.8.1", "GINOP", "termekleiras-hatalyos-20200625-tol.pdf", make_pdf("product 2020")),
    ("TOP-1.1.1-16-BS1", "TOP", "top-111-16.pdf", make_pdf("PALYAZATI FELHIVAS", "TOP-1.1.1-16")),
    ("VP-6.4.1-16", "VP", "felhivas.pdf", make_pdf("vp a")),
    ("VP-6.4.1-16", "VP", "vp6-641-16-felhivas.pdf", make_pdf("vp b")),
    ("VP-6.4.1-16", "VP", "vp6-641-16-to.pdf", make_pdf("grant deed")),
    ("EFOP-3.2.1-15", "EFOP", "kerdoiv.docx", b"PK\x03\x04 not a pdf"),
    ("EFOP-7.3.1-23", "EFOP", "efop-731-vegleges.pdf", make_pdf("Eljarasrend")),
]
MANUAL = {
    "ginop-plusz-221-25-felhivas.pdf": ("GINOP_PLUSZ-2.2.1-25", make_pdf("manual 221-25 scored by the expert")),
    "ginop-plusz-221-25-copy.pdf": ("GINOP_PLUSZ-2.2.1-25", make_pdf("scraped 221-25")),  # same bytes as scraped
}
GOLD_SHA = hashlib.sha256(make_pdf("manual 221-25 scored by the expert")).hexdigest()


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "old"
    pdf_dir = root / "raw_pdfs"
    pdf_dir.mkdir(parents=True)
    with open(root / "scrape_log.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "program", "tender_code", "tender_status", "doc_title", "outcome", "file_path", "call_code", "notes"])
        w.writerow(["t", "GINOP", "GINOP-9.9.9", "Lezárva", "x.pdf", "skipped_duplicate", "", "", ""])
        for i, (code, prog, name, content) in enumerate(SCRAPED):
            stored = f"{i:02d}_{name}"
            (pdf_dir / stored).write_bytes(content)
            w.writerow(["t", prog, code, "Lezárva", name, "downloaded", f"C:\\stale\\{stored}", code.lower(), ""])
    conn = sqlite3.connect(root / "palyazat.db")
    conn.execute("CREATE TABLE documents (file_hash TEXT, source_url TEXT, download_date TEXT)")
    for _, _, name, content in SCRAPED:
        conn.execute("INSERT INTO documents VALUES (?, ?, ?)",
                     (hashlib.sha256(content).hexdigest(), f"https://example.org/{name}", "2026-07-28"))
    conn.commit()
    conn.close()
    with zipfile.ZipFile(root / "Pipeline_V0.zip", "w") as z:
        z.writestr("data/raw_pdfs/", "")
        for name, (_, content) in MANUAL.items():
            z.writestr(f"data/raw_pdfs/{name}", content)
        z.writestr("other/ignored.pdf", make_pdf("not in raw_pdfs"))
    import_cfg = {
        "manual_files": {name: code for name, (code, _) in MANUAL.items()},
        "doc_type_overrides": {},
        "version_overrides": {},
        "gold_pins": {"GINOP Plusz-2.2.1-25": GOLD_SHA},
    }
    (root / "import.yaml").write_text(yaml.safe_dump(import_cfg, allow_unicode=True), encoding="utf-8")
    sources = {
        "scraped_pdfs": pdf_dir,
        "scrape_log": root / "scrape_log.csv",
        "old_database": root / "palyazat.db",
        "manual_zip": root / "Pipeline_V0.zip",
    }
    return sources, root / "import.yaml"


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "data")
    yield c
    c.close()


def tree_hash(path):
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode() + p.read_bytes())
    return h.hexdigest()


def documents(conn, run_id):
    return {r["call_code"]: dict(r) for r in conn.execute("SELECT * FROM documents WHERE run_id = ?", (run_id,))}


def run_c1(conn, tmp_path, snapshot):
    sources, import_path = snapshot
    return acquire.run(conn, {"test": True}, tmp_path / "data", sources, import_path)


def test_the_corpus(conn, tmp_path, snapshot):
    before = tree_hash(tmp_path / "old")
    run_id = run_c1(conn, tmp_path, snapshot)
    assert tree_hash(tmp_path / "old") == before  # SPEC-C1-01: the sources are untouched
    docs = documents(conn, run_id)
    assert set(docs) == {"GINOP_PLUSZ-2.2.1-25", "GINOP_PLUSZ-1.4.3-24", "GINOP-8.8.1", "TOP-1.1.1-16"}
    # Gold pin: the expert's manual copy, not the scraped one (DEC-34).
    assert docs["GINOP_PLUSZ-2.2.1-25"]["sha256"] == GOLD_SHA
    assert docs["GINOP_PLUSZ-2.2.1-25"]["source"] == "manual"
    assert docs["GINOP_PLUSZ-2.2.1-25"]["source_url"] is None
    # Version choice, product description, first-page typing, county suffix.
    assert docs["GINOP_PLUSZ-1.4.3-24"]["original_name"] == "felhivas-v10.pdf"
    assert docs["GINOP-8.8.1"]["doc_type"] == "product_description"
    assert docs["TOP-1.1.1-16"]["programme"] == "TOP"
    assert docs["TOP-1.1.1-16"]["call_series"] == "TOP-1.1.1"
    assert docs["GINOP_PLUSZ-1.4.3-24"]["source_url"] == "https://example.org/felhivas-v10.pdf"
    for d in docs.values():
        assert d["doc_id"] == d["sha256"][:16]
        assert d["page_count"] == 1
        assert (tmp_path / "data" / d["file_path"]).read_bytes()[:4] == b"%PDF"
    assert docs["GINOP_PLUSZ-1.4.3-24"]["file_path"] == "pdf/GINOP_PLUSZ-1.4.3-24_felhivas-v10.pdf"


def test_the_report(conn, tmp_path, snapshot):
    run_id = run_c1(conn, tmp_path, snapshot)
    rep = json.loads((tmp_path / "data" / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))
    assert rep["inputs"] == {"scraped": len(SCRAPED), "manual": len(MANUAL)}
    assert len(rep["files"]) == len(SCRAPED) + len(MANUAL)  # SPEC-C1-11: every input has a fate
    reasons = {(f["original_name"], f["fate"], f["reason"]) for f in rep["files"]}
    assert ("kerdoiv.docx", "excluded", "not_pdf") in reasons
    assert ("ginop-plusz-221-25-copy.pdf", "excluded", "duplicate") in reasons
    assert ("altalanos-utmutato.pdf", "excluded", "other_type") in reasons
    assert ("vp6-641-16-to.pdf", "excluded", "other_type") in reasons
    assert ("felhivas-v9.pdf", "excluded", "older_version") in reasons
    assert ("ginop-plusz-221-25-felhivas.pdf", "excluded", "not_the_gold_version") in reasons
    assert ("efop-731-vegleges.pdf", "needs_review", "doc_type_unknown") in reasons
    assert [v["call_code"] for v in rep["review"]["version"]] == ["VP-6.4.1-16"]
    assert "VP-6.4.1-16" in rep["calls_without_document"]
    assert rep["county_suffixes_removed"] == [{"call_code": "TOP-1.1.1-16", "suffix": "BS1", "original_name": "top-111-16.pdf"}]
    assert rep["documents"]["typed_by"] == {"first_page": 1, "name": 3}
    assert rep["documents"]["chosen_by"] == {"gold_pin": 1, "only": 2, "version": 1}


def test_overrides_resolve_the_review(conn, tmp_path, snapshot):
    sources, import_path = snapshot
    cfg = yaml.safe_load(import_path.read_text(encoding="utf-8"))
    vp_b = hashlib.sha256(make_pdf("vp b")).hexdigest()[:16]
    efop = hashlib.sha256(make_pdf("Eljarasrend")).hexdigest()[:16]
    cfg["version_overrides"] = {"VP-6.4.1-16": vp_b}
    cfg["doc_type_overrides"] = {efop: "other"}
    import_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    run_id = acquire.run(conn, {}, tmp_path / "data", sources, import_path)
    docs = documents(conn, run_id)
    assert docs["VP-6.4.1-16"]["doc_id"] == vp_b
    rep = json.loads((tmp_path / "data" / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))
    assert rep["review"] == {"doc_type": [], "version": []}


def test_second_run_gives_the_same_documents(conn, tmp_path, snapshot):
    a = run_c1(conn, tmp_path, snapshot)
    b = run_c1(conn, tmp_path, snapshot)

    def ids(run_id):
        return sorted((d["call_code"], d["doc_id"], d["file_path"]) for d in documents(conn, run_id).values())

    assert ids(a) == ids(b)


def test_missing_gold_document_fails_and_leaves_nothing(conn, tmp_path, snapshot):
    sources, import_path = snapshot
    cfg = yaml.safe_load(import_path.read_text(encoding="utf-8"))
    cfg["gold_pins"]["Not In Snapshot-1.1.1-21"] = "0" * 64
    import_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    with pytest.raises(acquire.ImportConfigError, match="Not In Snapshot"):
        acquire.run(conn, {}, tmp_path / "data", sources, import_path)
    failed = conn.execute("SELECT * FROM runs WHERE stage = 'C1'").fetchone()
    assert failed["status"] == "failed"
    assert not list((tmp_path / "data").glob("pdf/*"))


def test_failure_while_writing_leaves_nothing(conn, tmp_path, snapshot, monkeypatch):
    def boom(*args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(acquire, "_insert", boom)
    with pytest.raises(RuntimeError, match="injected"):
        run_c1(conn, tmp_path, snapshot)
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert not list((tmp_path / "data").glob("pdf/*"))
    assert not list((tmp_path / "data").glob("reports/*/c1_report.json"))


def test_unknown_programme_stops_the_run(conn, tmp_path, snapshot):
    sources, import_path = snapshot
    cfg = yaml.safe_load(import_path.read_text(encoding="utf-8"))
    cfg["manual_files"]["ginop-plusz-221-25-copy.pdf"] = "KAP-1.1.1-23"
    import_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    # The copy is a duplicate, so its code is never checked; point the gold file itself at KAP.
    cfg["manual_files"]["ginop-plusz-221-25-felhivas.pdf"] = "KAP-1.1.1-23"
    import_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    with pytest.raises(callcode.UnknownProgrammeError, match="KAP"):
        acquire.run(conn, {}, tmp_path / "data", sources, import_path)


def test_manual_file_without_call_code(conn, tmp_path, snapshot):
    sources, import_path = snapshot
    cfg = yaml.safe_load(import_path.read_text(encoding="utf-8"))
    del cfg["manual_files"]["ginop-plusz-221-25-copy.pdf"]
    import_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="no call code"):
        acquire.run(conn, {}, tmp_path / "data", sources, import_path)


def test_unreadable_file_is_excluded(tmp_path):
    from grantrisk.corpus.snapshot import InputFile

    broken = InputFile("scraped", "felhivas-v2.pdf", "x", "GINOP_PLUSZ-9.9.9-25", lambda: b"%PDF-1.4\nbroken")
    good = InputFile("scraped", "felhivas-v1.pdf", "y", "GINOP_PLUSZ-9.9.9-25", lambda: make_pdf("v1"))
    p = acquire.plan([broken, good], {"doc_type_overrides": {}, "version_overrides": {}, "gold_pins": {}})
    assert [c.candidate.file.original_name for c in p.documents] == ["felhivas-v1.pdf"]
    assert {(f["original_name"], f["reason"]) for f in p.fates} >= {("felhivas-v2.pdf", "unreadable")}
