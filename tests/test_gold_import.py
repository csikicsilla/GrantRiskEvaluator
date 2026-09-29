"""The gold import (SPEC-L1-04) on the real gold file, with synthetic pins and corpus."""

import csv
import hashlib
import json

import pytest

from grantrisk.extraction.manual import gold
from grantrisk.labelling.scoring import FACTORS
from grantrisk.store import db, runs
from grantrisk.store.db import transaction


def gold_names(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r["Felhívás"].strip() for r in csv.DictReader(f, delimiter=";")]


def fake_sha(name):
    return hashlib.sha256(name.encode()).hexdigest()


def seed_c1(conn, shas):
    c1 = runs.start(conn, "C1", {"test": True})
    with transaction(conn):
        for sha in shas:
            conn.execute(
                "INSERT INTO documents (run_id, doc_id, call_code, call_series, programme, period, doc_type,"
                " source, original_name, file_path, sha256) VALUES (?, ?, ?, ?, 'GINOP_PLUSZ', '2021-2027',"
                " 'main_call', 'scraped', 'x.pdf', 'pdf/x.pdf', ?)",
                (c1, sha[:16], f"CODE-{sha[:4]}", f"CODE-{sha[:4]}", sha),
            )
        runs.complete(conn, c1)
    return c1


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "data")
    yield c
    c.close()


@pytest.fixture
def setup(conn, gold_csv):
    pins = {n: fake_sha(n) for n in gold_names(gold_csv)}
    c1 = seed_c1(conn, pins.values())
    return c1, pins


def run_import(conn, tmp_path, c1, csv_path, pins):
    return gold.run(conn, {}, tmp_path / "data", c1_run_id=c1, gold_csv=csv_path, gold_pins=pins)


def write_variant(tmp_path, gold_csv, edit):
    """A copy of the gold file with ``edit(rows)`` applied to its data rows."""
    text = gold_csv.read_bytes().decode("utf-8-sig")
    lines = text.split("\r\n")
    header, rows = lines[0], [line for line in lines[1:] if line]
    rows = edit(rows)
    path = tmp_path / "variant.csv"
    path.write_bytes(("﻿" + "\r\n".join([header, *rows]) + "\r\n").encode("utf-8"))
    return path


def test_import(conn, tmp_path, gold_csv, setup):
    c1, pins = setup
    run_id = run_import(conn, tmp_path, c1, gold_csv, pins)
    assert runs.get(conn, run_id)["stage"] == "L1"
    assert runs.inputs(conn, run_id) == [c1]
    n = conn.execute("SELECT COUNT(*) FROM gold_records WHERE run_id = ?", (run_id,)).fetchone()[0]
    assert n == 42 * len(FACTORS)
    obs = conn.execute(
        "SELECT DISTINCT source, status FROM factor_observations WHERE run_id = ?", (run_id,)
    ).fetchall()
    assert [tuple(r) for r in obs] == [("manual", "found")]
    row = conn.execute(
        "SELECT points FROM gold_records WHERE run_id = ? AND gold_name = 'GINOP Plusz-2.2.1-25' AND factor = 'bead_napok'",
        (run_id,),
    ).fetchone()
    assert row["points"] == 1  # as in the gold file
    report = json.loads((tmp_path / "data" / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))
    assert report["calls"] == 42
    assert report["gold_file_sha256"] == hashlib.sha256(gold_csv.read_bytes()).hexdigest()


def test_not_scored_cell(conn, tmp_path, gold_csv, setup):
    c1, pins = setup

    def dash_first_cell(rows):
        cells = rows[0].split(";")
        cells[1] = "-"
        return [";".join(cells), *rows[1:]]

    path = write_variant(tmp_path, gold_csv, dash_first_cell)
    run_id = run_import(conn, tmp_path, c1, path, pins)
    r = conn.execute(
        "SELECT points, status FROM factor_observations WHERE run_id = ? AND factor = 'fin_form' AND points IS NULL",
        (run_id,),
    ).fetchall()
    assert [tuple(x) for x in r] == [(None, "not_found")]


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda rows: [rows[0].replace(";3;", ";7;", 1), *rows[1:]], "not an allowed number"),
        (lambda rows: [*rows, rows[0]], "appears twice"),
    ],
)
def test_invalid_gold_file(conn, tmp_path, gold_csv, setup, edit, message):
    c1, pins = setup
    path = write_variant(tmp_path, gold_csv, edit)
    with pytest.raises(gold.GoldImportError, match=message):
        run_import(conn, tmp_path, c1, path, pins)
    failed = conn.execute("SELECT * FROM runs WHERE stage = 'L1'").fetchone()
    assert failed["status"] == "failed"
    assert conn.execute("SELECT COUNT(*) FROM gold_records").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM factor_observations").fetchone()[0] == 0


def test_row_without_pin(conn, tmp_path, gold_csv, setup):
    c1, pins = setup
    del pins["VOP Plusz-4.1.2-22"]
    with pytest.raises(gold.GoldImportError, match="VOP Plusz-4.1.2-22: no gold pin"):
        run_import(conn, tmp_path, c1, gold_csv, pins)


def test_pinned_document_missing_from_the_corpus(conn, tmp_path, gold_csv, setup):
    c1, pins = setup
    pins["VOP Plusz-4.1.2-22"] = "f" * 64
    with pytest.raises(gold.GoldImportError, match="not in C1 run"):
        run_import(conn, tmp_path, c1, gold_csv, pins)


def test_unused_pin_is_a_warning(conn, tmp_path, gold_csv, setup):
    c1, pins = setup
    run_id = run_import(conn, tmp_path, c1, gold_csv, {**pins, "Extra-1.1.1-21": fake_sha("x")})
    report = json.loads((tmp_path / "data" / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))
    assert report["warnings"] == ["gold pin without a row in the gold file: Extra-1.1.1-21"]


def test_wrong_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Felhívás;fin_form\nX;3\n", encoding="utf-8")
    with pytest.raises(gold.GoldImportError, match="unexpected columns"):
        gold.read_gold(path)
