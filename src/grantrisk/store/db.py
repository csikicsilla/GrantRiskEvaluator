"""The SQLite database under the data root, created by versioned migrations (DEC-22)."""

from __future__ import annotations

import datetime
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DB_NAME = "grantrisk.db"
MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_MIGRATION_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def connect(data_root: Path) -> sqlite3.Connection:
    """Open (and if needed create) the database, and apply pending migrations.

    The connection is in autocommit mode: a statement outside ``transaction()`` is
    committed on its own.
    """
    data_root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(data_root / DB_NAME, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Everything inside is committed together, or nothing is."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def migrations() -> list[tuple[int, Path]]:
    found = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        m = _MIGRATION_NAME.match(path.name)
        if not m:
            raise ValueError(f"badly named migration: {path.name}")
        found.append((int(m.group(1)), path))
    return found


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return row[0] or 0


def migrate(conn: sqlite3.Connection) -> None:
    """Apply every migration newer than the database, each in its own transaction."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
    )
    current = schema_version(conn)
    for version, path in migrations():
        if version <= current:
            continue
        applied_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        script = (
            f"BEGIN;\n{path.read_text(encoding='utf-8')}\n"
            f"INSERT INTO schema_migrations VALUES ({version}, '{path.name}', '{applied_at}');\n"
            "COMMIT;"
        )
        try:
            conn.executescript(script)
        except sqlite3.Error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
