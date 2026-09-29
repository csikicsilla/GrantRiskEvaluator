"""Sanity checks of the gold fixture: the invariants recorded in DEC-20."""

import csv

from grantrisk.labelling.scoring import ALLOWED_POINTS as ALLOWED
from grantrisk.labelling.scoring import FACTORS


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return reader.fieldnames, list(reader)


def test_columns(gold_csv):
    header, _ = read(gold_csv)
    assert header == ["Felhívás", *FACTORS]


def test_42_distinct_calls(gold_csv):
    _, rows = read(gold_csv)
    names = [r["Felhívás"].strip() for r in rows]
    assert len(names) == 42
    assert len(set(names)) == 42


def test_every_cell_scored_with_allowed_points(gold_csv):
    _, rows = read(gold_csv)
    for r in rows:
        for f in FACTORS:
            value = r[f].strip()
            assert value != "-", (r["Felhívás"], f)
            assert int(value) in ALLOWED[f], (r["Felhívás"], f, value)


def test_corrected_call_name(gold_csv):
    _, rows = read(gold_csv)
    names = {r["Felhívás"].strip() for r in rows}
    assert "GINOP Plusz-2.2.1-25" in names
    assert "GINOP Plusz-2.2.1-24" not in names
