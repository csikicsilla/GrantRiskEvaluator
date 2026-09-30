"""The database, runs and lineage, and the file store (DEC-22, ARC-02)."""

import pytest

from grantrisk.store import db, files, runs


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "data")
    yield c
    c.close()


def tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def test_connect_creates_the_schema(tmp_path, conn):
    assert (tmp_path / "data" / db.DB_NAME).exists()
    assert db.schema_version(conn) == len(db.migrations())
    assert {"runs", "run_inputs", "documents", "consolidated_factors", "risk_labels", "risk_label_factors"} <= tables(conn)


def test_migrations_are_applied_once(tmp_path, conn):
    conn.close()
    again = db.connect(tmp_path / "data")
    assert again.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == len(db.migrations())
    again.close()


def test_transaction_rolls_back_on_error(conn):
    run_id = runs.start(conn, "C1", {})
    with pytest.raises(RuntimeError):
        with db.transaction(conn):
            runs.complete(conn, run_id)
            raise RuntimeError("boom")
    assert runs.get(conn, run_id)["status"] == "running"


def test_run_lifecycle_and_lineage(conn):
    c1 = runs.start(conn, "C1", {"a": 1})
    with db.transaction(conn):
        runs.complete(conn, c1, "reports/x.json")
    l2 = runs.start(conn, "L2", {"a": 1}, inputs=[c1])
    row = runs.get(conn, c1)
    assert row["status"] == "complete"
    assert row["report_path"] == "reports/x.json"
    assert row["config_hash"] == runs.get(conn, l2)["config_hash"]
    assert runs.inputs(conn, l2) == [c1]
    runs.fail(conn, l2, "ValueError: x")
    assert runs.get(conn, l2)["status"] == "failed"


def test_run_ids_name_their_stage(conn):
    assert runs.start(conn, "L3", {}).startswith("L3-")
    with pytest.raises(ValueError):
        runs.start(conn, "X9", {})


def test_require_complete(conn):
    c1 = runs.start(conn, "C1", {})
    with pytest.raises(ValueError, match="running"):
        runs.require_complete(conn, c1, "C1")
    with db.transaction(conn):
        runs.complete(conn, c1)
    runs.require_complete(conn, c1, "C1")
    with pytest.raises(ValueError, match="not L2"):
        runs.require_complete(conn, c1, "L2")
    with pytest.raises(ValueError, match="does not exist"):
        runs.require_complete(conn, "nope", "C1")


def test_input_runs_must_exist(conn):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        runs.start(conn, "L2", {}, inputs=["missing-run"])


def test_file_store_never_overwrites(tmp_path):
    root = tmp_path / "data"
    assert files.write_json(root, "reports/r1/a.json", {"x": 1}) == "reports/r1/a.json"
    with pytest.raises(FileExistsError):
        files.write_json(root, "reports/r1/a.json", {"x": 2})
    files.remove(root, "reports/r1/a.json")
    assert not (root / "reports/r1/a.json").exists()


def test_a_resumed_run_records_the_code_version_that_continues_it(tmp_path):
    conn = db.connect(tmp_path)
    run_id = runs.start(conn, "C2", {})
    with pytest.raises(ValueError, match="unfinished M1"):
        runs.resume(conn, run_id, "M1")
    runs.fail(conn, run_id, "stopped")
    runs.resume(conn, run_id, "C2")
    runs.resume(conn, run_id, "C2")
    assert runs.get(conn, run_id)["status"] == "running"
    assert len(runs.code_versions(conn, run_id)) == 3
    assert [r[0] for r in conn.execute("SELECT resume FROM run_resumes WHERE run_id = ?", (run_id,))] == [1, 2]
    runs.complete(conn, run_id)
    with pytest.raises(ValueError, match="not an unfinished"):
        runs.resume(conn, run_id, "C2")
