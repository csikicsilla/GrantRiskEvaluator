"""Runs and lineage (ARC-02).

Every stage execution is a run with an id. A run records its stage, the code
version, a snapshot of the configuration and the runs it read. It never changes
another run's output.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import secrets
import sqlite3
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from grantrisk.store.db import transaction

STAGES = ("C1", "C2", "L1", "L2", "L3", "M1", "M2", "E1", "E2", "E3")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def new_run_id(stage: str) -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stage}-{stamp}-{secrets.token_hex(3)}"


def code_version() -> str:
    """The git commit of the code line, with ``-dirty`` if it has uncommitted changes."""
    try:
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()

        commit = git("rev-parse", "HEAD")
        return commit + ("-dirty" if git("status", "--porcelain") else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def config_json(values: Mapping[str, Any]) -> str:
    return json.dumps(values, sort_keys=True, ensure_ascii=False, default=str)


def start(
    conn: sqlite3.Connection,
    stage: str,
    config_values: Mapping[str, Any],
    inputs: Iterable[str] = (),
) -> str:
    """Record a new run with status ``running``, and the runs it reads."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    run_id = new_run_id(stage)
    snapshot = config_json(config_values)
    with transaction(conn):
        conn.execute(
            "INSERT INTO runs (run_id, stage, status, started_at, code_version, config_json, config_hash)"
            " VALUES (?, ?, 'running', ?, ?, ?, ?)",
            (run_id, stage, _now(), code_version(), snapshot, hashlib.sha256(snapshot.encode()).hexdigest()),
        )
        conn.executemany(
            "INSERT INTO run_inputs (run_id, input_run_id) VALUES (?, ?)",
            [(run_id, i) for i in inputs],
        )
    return run_id


def complete(conn: sqlite3.Connection, run_id: str, report_path: str | None = None) -> None:
    """Mark a run complete. Call it inside the transaction that writes the run's output."""
    conn.execute(
        "UPDATE runs SET status = 'complete', finished_at = ?, report_path = ? WHERE run_id = ?",
        (_now(), report_path, run_id),
    )


def fail(conn: sqlite3.Connection, run_id: str, error: str) -> None:
    conn.execute(
        "UPDATE runs SET status = 'failed', finished_at = ?, error = ? WHERE run_id = ?",
        (_now(), error, run_id),
    )


def get(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()


def inputs(conn: sqlite3.Connection, run_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT input_run_id FROM run_inputs WHERE run_id = ? ORDER BY input_run_id", (run_id,)
    )
    return [r[0] for r in rows]


def require_complete(conn: sqlite3.Connection, run_id: str, stage: str) -> None:
    """Raise ValueError unless ``run_id`` is a complete run of ``stage``."""
    row = get(conn, run_id)
    if row is None:
        raise ValueError(f"run {run_id} does not exist")
    if row["stage"] != stage:
        raise ValueError(f"run {run_id} is a {row['stage']} run, not {stage}")
    if row["status"] != "complete":
        raise ValueError(f"run {run_id} is {row['status']}, not complete")
