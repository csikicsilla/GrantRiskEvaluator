"""Building L3 inputs for the tests."""

import json

from grantrisk.labelling.label import DocumentInput, FactorInput
from grantrisk.labelling.scoring import FACTORS
from grantrisk.store import runs
from grantrisk.store.db import transaction

# Values that score 0 points, except fin_form, which cannot. Not a loan, whose max_tam_int the
# loan rule would set (DEC-40): conditional_grant, 2 points.
BASE_VALUES = {
    "fin_form": "conditional_grant",
    "tam_osszeg": 50_000_000,
    "konzorcium": 0,
    "bead_napok": 40,
    "max_tam_int": 20,
    "eloleg": 10,
    "idotartam": 6,
    "tam_tevekenyseg": ["egyeb"],
    "egysz_elszam": 1,
    "biztositek": 1,
}
BASE_TOTAL = 2


def doc(doc_id, programme="GINOP_PLUSZ", **values):
    """A document with BASE_VALUES, overridden by ``values``.

    A FactorInput is used as it is; None means the factor was not determined.
    """
    factors = {}
    for f in FACTORS:
        v = values.get(f, BASE_VALUES[f])
        factors[f] = v if isinstance(v, FactorInput) else FactorInput(value=v)
    return DocumentInput(doc_id, programme, factors)


def seed_runs(conn, docs, *, skip_documents=()):
    """Write a complete C1 run with the documents and a complete L2 run with their factors."""
    c1 = runs.start(conn, "C1", {"test": True})
    with transaction(conn):
        for d in docs:
            if d.doc_id in skip_documents:
                continue
            period = "2021-2027" if (d.programme or "").endswith("_PLUSZ") else "2014-2020"
            conn.execute(
                "INSERT INTO documents (run_id, doc_id, call_code, call_series, programme, period,"
                " doc_type, source, original_name, file_path, sha256)"
                " VALUES (?, ?, ?, ?, ?, ?, 'main_call', 'scraped', 'x.pdf', 'pdf/x.pdf', ?)",
                (c1, d.doc_id, d.doc_id, d.doc_id, d.programme, period, d.doc_id),
            )
        runs.complete(conn, c1)
    l2 = runs.start(conn, "L2", {"test": True}, inputs=[c1])
    with transaction(conn):
        for d in docs:
            for f, fi in d.factors.items():
                rule = "manual" if fi.points is not None else "preferred" if fi.value is not None else "none"
                conn.execute(
                    "INSERT INTO consolidated_factors (run_id, doc_id, factor, value_json, points, rule)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (l2, d.doc_id, f, None if fi.value is None else json.dumps(fi.value), fi.points, rule),
                )
        runs.complete(conn, l2)
    return c1, l2
