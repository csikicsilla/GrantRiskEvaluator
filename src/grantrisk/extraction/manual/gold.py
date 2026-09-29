"""L1 manual extractor: the gold import (SPEC-L1-04, DEC-20, DEC-34).

Reads the expert's points from Gold_second.csv, maps each row to its document
through the gold pins, and writes one GoldRecord and one manual FactorObservation
per row and factor.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grantrisk.labelling.scoring import ALLOWED_POINTS, FACTORS
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

NAME_COLUMN = "Felhívás"
NOT_SCORED = "-"


class GoldImportError(ValueError):
    """The gold file or its pins are inconsistent; nothing is written."""


@dataclass(frozen=True)
class GoldRow:
    name: str  # the call as written in the gold file, without surrounding spaces
    points: dict[str, int | None]  # None where the expert did not score


def read_gold(path: Path) -> list[GoldRow]:
    """Read and check the gold file: ';'-separated, UTF-8 with BOM, one row per call."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        header = reader.fieldnames or []
        if header != [NAME_COLUMN, *FACTORS]:
            raise GoldImportError(f"unexpected columns in {path.name}: {header}")
        raw_rows = list(reader)
    problems = []
    rows = []
    seen = set()
    for r in raw_rows:
        name = r[NAME_COLUMN].strip()
        if name in seen:
            problems.append(f"{name}: appears twice")
        seen.add(name)
        points: dict[str, int | None] = {}
        for factor in FACTORS:
            cell = (r[factor] or "").strip()
            if cell == NOT_SCORED:
                points[factor] = None
            elif cell.isdigit() and int(cell) in ALLOWED_POINTS[factor]:
                points[factor] = int(cell)
            else:
                problems.append(f"{name} / {factor}: {cell!r} is not an allowed number of points")
        rows.append(GoldRow(name, points))
    if problems:
        raise GoldImportError("invalid gold file:\n  " + "\n  ".join(problems))
    return rows


def _corpus(conn: sqlite3.Connection, c1_run_id: str) -> dict[str, tuple[str, str]]:
    """SHA-256 → (doc_id, call_code) for the documents of a C1 run."""
    rows = conn.execute("SELECT sha256, doc_id, call_code FROM documents WHERE run_id = ?", (c1_run_id,))
    return {sha: (doc_id, code) for sha, doc_id, code in rows}


def run(
    conn: sqlite3.Connection,
    config_values: Mapping[str, Any],
    data_root: Path,
    *,
    c1_run_id: str,
    gold_csv: Path,
    gold_pins: Mapping[str, str],
    gold_set: str = "gold_second",
) -> str:
    """Import the gold set as an L1 run with source 'manual'. Returns the run id."""
    runs.require_complete(conn, c1_run_id, "C1")
    gold_sha = hashlib.sha256(gold_csv.read_bytes()).hexdigest()
    snapshot = {**config_values, "gold_file": {"path": str(gold_csv), "sha256": gold_sha}, "gold_pins": dict(gold_pins)}
    run_id = runs.start(conn, "L1", snapshot, inputs=[c1_run_id])
    report_path = None
    try:
        rows = read_gold(gold_csv)
        corpus = _corpus(conn, c1_run_id)
        problems = []
        for r in rows:
            sha = gold_pins.get(r.name)
            if sha is None:
                problems.append(f"{r.name}: no gold pin in the import configuration")
            elif sha not in corpus:
                problems.append(f"{r.name}: the pinned document {sha[:16]} is not in C1 run {c1_run_id}")
        if problems:
            raise GoldImportError("the gold rows cannot be mapped to documents:\n  " + "\n  ".join(problems))

        unused_pins = sorted(set(gold_pins) - {r.name for r in rows})
        distribution = {
            f: dict(sorted(collections.Counter(str(r.points[f]) for r in rows).items())) for f in FACTORS
        }
        report = {
            "input_runs": {"C1": c1_run_id},
            "gold_set": gold_set,
            "gold_file_sha256": gold_sha,
            "calls": len(rows),
            "records": len(rows) * len(FACTORS),
            "not_scored": sum(v is None for r in rows for v in r.points.values()),
            "points_by_factor": distribution,
            "warnings": [f"gold pin without a row in the gold file: {n}" for n in unused_pins],
        }
        report_path = files.write_json(data_root, f"reports/{run_id}/l1_manual_report.json", report)
        with transaction(conn):
            for r in rows:
                doc_id, call_code = corpus[gold_pins[r.name]]
                for factor in FACTORS:
                    points = r.points[factor]
                    conn.execute(
                        "INSERT INTO gold_records VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (run_id, gold_set, r.name, doc_id, call_code, factor, points),
                    )
                    conn.execute(
                        "INSERT INTO factor_observations (run_id, doc_id, factor, source, points, status)"
                        " VALUES (?, ?, ?, 'manual', ?, ?)",
                        (run_id, doc_id, factor, points, "not_found" if points is None else "found"),
                    )
            runs.complete(conn, run_id, report_path)
    except Exception as exc:
        if report_path:
            files.remove(data_root, report_path)
        runs.fail(conn, run_id, f"{type(exc).__name__}: {exc}")
        raise
    return run_id
