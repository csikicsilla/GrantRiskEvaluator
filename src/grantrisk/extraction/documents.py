"""The document set of the automated extractors (SPEC-L1-03, -14): the texts of one C2 run."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Any


def of_c2_run(
    conn: sqlite3.Connection, c2_run_id: str, doc_ids: Sequence[str] | None = None
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """(doc_id, Markdown) of the documents with text, sorted by doc_id, and the skipped ones with the reason.

    ``doc_ids`` filters the run, e.g. to the gold documents; a requested document that is not
    in the run is skipped too.
    """
    rows = conn.execute(
        "SELECT doc_id, status, error, markdown FROM document_texts WHERE run_id = ? ORDER BY doc_id", (c2_run_id,)
    ).fetchall()
    if doc_ids is not None:
        wanted = set(doc_ids)
        rows = [r for r in rows if r["doc_id"] in wanted]
    texts = [(r["doc_id"], r["markdown"]) for r in rows if r["status"] == "ok"]
    skipped = [{"doc_id": r["doc_id"], "reason": f"C2 status {r['status']}: {r['error']}"} for r in rows if r["status"] != "ok"]
    if doc_ids is not None:
        present = {r["doc_id"] for r in rows}
        skipped += [{"doc_id": d, "reason": "not in the C2 run"} for d in sorted(set(doc_ids) - present)]
    return texts, skipped
