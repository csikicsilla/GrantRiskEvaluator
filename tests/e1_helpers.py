"""Building E1 inputs for the tests: a small gold set, its documents and extraction runs."""

import hashlib
import json

from grantrisk.evaluation.validate import GoldDocument
from grantrisk.labelling.scoring import FACTORS
from grantrisk.store import runs
from grantrisk.store.db import transaction

# A value that scores each allowed number of points (Spec_ScoringSystem.md §1.2).
VALUE_FOR = {
    "fin_form": {1: "loan", 2: "conditional_grant", 3: "grant"},
    "tam_osszeg": {0: 50_000_000, 1: 200_000_000, 2: 500_000_000, 3: 700_000_000},
    "konzorcium": {0: 0, 3: 1},
    "bead_napok": {0: 40, 1: 20, 2: 10, 3: 5},
    "max_tam_int": {0: 20, 1: 40, 2: 60, 3: 80},
    "eloleg": {0: 10, 1: 40, 2: 60, 3: 80},
    "idotartam": {0: 6, 1: 15, 2: 20, 3: 30},
    "tam_tevekenyseg": {0: ["egyeb"], 2: ["kutatas_fejlesztes"], 3: ["infrastruktura_ingatlan"]},
    "egysz_elszam": {0: 1, 3: 0},
    "biztositek": {0: 1, 3: 0},
}

# Six gold calls: (name, programme, period, points in the order of FACTORS).
# Totals 30, 1, 16, 17, 20, 6: fixed labels high, low, medium, medium, high, low;
# terciles low {B, F}, medium {C, D}, high {E, A}.
GOLD = [
    ("A Plusz-1.1.1-21", "GINOP_PLUSZ", "2021-2027", [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
    ("B Plusz-1.1.2-22", "GINOP_PLUSZ", "2021-2027", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ("C Plusz-1.3.1-21", "TOP_PLUSZ", "2021-2027", [3, 2, 0, 1, 2, 1, 2, 2, 3, 0]),
    ("D-2.1.1-16", "GINOP", "2014-2020", [2, 1, 3, 0, 1, 3, 1, 3, 0, 3]),
    ("E-4.3.1-16", "TOP", "2014-2020", [3, 1, 0, 2, 3, 2, 3, 0, 3, 3]),
    ("F-7.2.1-20", "VP", "VP", [1, 0, 0, 1, 0, 0, 1, 0, 0, 3]),
]


def sha(name):
    return hashlib.sha256(name.encode()).hexdigest()


def doc_id(name):
    return sha(name)[:16]


def gold_documents():
    return [
        GoldDocument(doc_id(n), n, f"CODE-{n}", programme, period, dict(zip(FACTORS, points)))
        for n, programme, period, points in GOLD
    ]


def gold_points(name):
    return dict(zip(FACTORS, next(p for n, _, _, p in GOLD if n == name)))


def write_gold_csv(path, rows=GOLD, not_scored=()):
    """The gold file in the format of Gold_second.csv; ``not_scored`` holds (name, factor) cells written as "-"."""
    lines = ["Felhívás;" + ";".join(FACTORS)]
    for name, _, _, points in rows:
        cells = ["-" if (name, f) in not_scored else str(p) for f, p in zip(FACTORS, points)]
        lines.append(";".join([name, *cells]))
    path.write_bytes(("﻿" + "\r\n".join(lines) + "\r\n").encode("utf-8"))
    return path


def seed_c1(conn, rows=GOLD):
    c1 = runs.start(conn, "C1", {"test": True})
    with transaction(conn):
        for name, programme, period, _ in rows:
            conn.execute(
                "INSERT INTO documents (run_id, doc_id, call_code, call_series, programme, period, doc_type,"
                " source, original_name, file_path, sha256) VALUES (?, ?, ?, ?, ?, ?, 'main_call', 'scraped',"
                " 'x.pdf', 'pdf/x.pdf', ?)",
                (c1, doc_id(name), f"CODE-{name}", f"CODE-{name}", programme, period, sha(name)),
            )
        runs.complete(conn, c1)
    return c1


def seed_l1(conn, source, observations):
    """A complete L1 run. ``observations`` maps (doc_id, factor) to a value, or to a dict of columns."""
    run_id = runs.start(conn, "L1", {"test": True})
    with transaction(conn):
        for (d, f), obs in observations.items():
            o = obs if isinstance(obs, dict) else {"value": obs}
            value = o.get("value")
            status = o.get("status", "found" if value is not None else "not_found")
            conn.execute(
                "INSERT INTO factor_observations (run_id, doc_id, factor, source, value_json, evidence,"
                " evidence_page, status, warnings_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, d, f, source, None if value is None else json.dumps(value), o.get("evidence"),
                 o.get("page"), status, json.dumps(o.get("warnings", []))),
            )
        runs.complete(conn, run_id)
    return run_id


def identical_observations(rows=GOLD, null_top_amount=True):
    """Values that reproduce the gold points exactly; TOP amounts left NULL for the TOP rule (DEC-31)."""
    result = {}
    for name, programme, _, points in rows:
        for f, p in zip(FACTORS, points):
            if f == "tam_osszeg" and programme in ("TOP", "TOP_PLUSZ") and null_top_amount:
                result[(doc_id(name), f)] = {"value": None, "status": "not_found"}
            else:
                result[(doc_id(name), f)] = VALUE_FOR[f][p]
    return result
