"""Stage C2: convert (Spec_C2_Convert.md) — the per-document conversion and the converter comparison.

``convert_document`` is shared by the comparison (SPEC-C2-11) and the stage run: it
calls the adapter, falls back to OCR for documents without a usable text layer
(SPEC-C2-05), cleans the output (SPEC-C2-06) and adds the warnings (SPEC-C2-07).
"""

from __future__ import annotations

import collections
import json
import os
import re
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from grantrisk.corpus import clean
from grantrisk.corpus.converters import Adapter, settings_hash
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

OCR_MIN_CHARS_PER_PAGE = 500


@dataclass
class TextResult:
    status: str  # ok or failed
    error: str | None = None
    markdown: str | None = None
    pages: list[str] = field(default_factory=list)  # the converter's raw output per page
    page_count: int = 0
    char_count: int = 0
    table_count: int = 0
    ocr_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    raw_reused: bool = False


def convert_document(
    adapter: Adapter,
    pdf_path: Path,
    expected_pages: int | None = None,
    ocr_min_chars_per_page: int = OCR_MIN_CHARS_PER_PAGE,
    raw: dict[str, Any] | None = None,
) -> TextResult:
    """Convert one PDF. ``raw`` is a stored earlier output of the same adapter and settings (SPEC-C2-09)."""
    start = time.perf_counter()
    try:
        if raw is not None:
            pages, ocr_pages, reused = raw["pages"], raw["ocr_pages"], True
        else:
            pages, ocr_pages, reused = adapter.convert(pdf_path), [], False
            if not pages:
                raise ValueError("the converter returned no pages")
            if sum(len(p) for p in pages) / len(pages) < ocr_min_chars_per_page:
                pages = adapter.convert(pdf_path, ocr=True)  # raises needs_ocr if impossible
                ocr_pages = list(range(1, len(pages) + 1))
        markdown = clean.assemble(pages)
    except Exception as exc:
        return TextResult(status="failed", error=f"{type(exc).__name__}: {exc}", duration_s=time.perf_counter() - start)
    return TextResult(
        status="ok",
        markdown=markdown,
        pages=list(pages),
        page_count=len(pages),
        char_count=len(markdown),
        table_count=clean.count_tables(markdown),
        ocr_pages=ocr_pages,
        warnings=clean.warnings(markdown, len(pages), expected_pages, ocr_pages),
        duration_s=raw["duration_s"] if reused else time.perf_counter() - start,
        raw_reused=reused,
    )


def raw_path(adapter: Adapter, doc_id: str) -> str:
    """Where the converter's raw output is kept: one folder per converter version and settings."""
    version = re.sub(r"[^A-Za-z0-9.]+", "_", adapter.version()).strip("_")
    return f"raw/{adapter.name}/{version}/{settings_hash(adapter)}/{doc_id}.json"


# --- The stage run (SPEC-C2-01, -08, -09, -10) ----------------------------------------

_ADAPTER_CACHE: dict[str, Adapter] = {}


def _adapter(name: str) -> Adapter:
    """One adapter per process, created on first use (workers load their own models)."""
    if name not in _ADAPTER_CACHE:
        from grantrisk.corpus import converters

        _ADAPTER_CACHE[name] = converters.create(name)
    return _ADAPTER_CACHE[name]


def convert_one(task: dict[str, Any]) -> dict[str, Any]:
    """Convert one document; runs in a worker process. Stores and reuses the raw output (SPEC-C2-09)."""
    adapter = _adapter(task["converter"])
    data_root = Path(task["data_root"])
    raw_file = data_root / task["raw_path"]
    raw = json.loads(raw_file.read_text(encoding="utf-8")) if raw_file.exists() else None
    result = convert_document(
        adapter, data_root / task["file_path"], task["expected_pages"], task["ocr_min_chars_per_page"], raw
    )
    if raw is None and result.status == "ok":
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = raw_file.with_name(raw_file.name + f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(
                {"converter": adapter.name, "converter_version": adapter.version(), "settings": adapter.settings(),
                 "pages": result.pages, "ocr_pages": result.ocr_pages, "duration_s": result.duration_s},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if raw_file.exists():  # another run stored the same output meanwhile; keep the first
            tmp.unlink()
        else:
            tmp.replace(raw_file)
    return {"doc_id": task["doc_id"], "raw_path": task["raw_path"], **asdict(result)}


def _insert_text(conn: sqlite3.Connection, run_id: str, r: dict[str, Any], adapter: Adapter) -> None:
    conn.execute(
        "INSERT INTO document_texts (run_id, doc_id, status, error, markdown, raw_path, raw_reused, char_count,"
        " page_count, table_count, ocr_pages_json, converter, converter_version, settings_hash, cleaning_version,"
        " warnings_json, duration_s) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id, r["doc_id"], r["status"], r["error"], r["markdown"],
            r["raw_path"] if r["status"] == "ok" else None, int(r["raw_reused"]), r["char_count"], r["page_count"],
            r["table_count"], json.dumps(r["ocr_pages"]), adapter.name, adapter.version(), settings_hash(adapter),
            clean.CLEANING_VERSION, json.dumps(r["warnings"]), r["duration_s"],
        ),
    )


def run(
    conn: sqlite3.Connection,
    config_values: Mapping[str, Any],
    data_root: Path,
    *,
    c1_run_id: str,
    resume_run_id: str | None = None,
    workers: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Convert every document of a C1 run. Returns the run id.

    Each finished document is stored at once, so an interrupted run can be resumed
    (``resume_run_id``); the run is marked complete only when every document has a result.
    """
    runs.require_complete(conn, c1_run_id, "C1")
    settings = config_values.get("convert", {})
    converter = settings.get("converter")
    if not converter:
        raise ValueError("convert.converter is not set in the configuration (DEC-30)")
    workers = workers or settings.get("workers", 1)
    ocr_min = settings.get("ocr_min_chars_per_page", OCR_MIN_CHARS_PER_PAGE)
    adapter = _adapter(converter)

    if resume_run_id:
        row = runs.get(conn, resume_run_id)
        if row is None or row["stage"] != "C2" or row["status"] == "complete":
            raise ValueError(f"run {resume_run_id} is not an unfinished C2 run")
        if c1_run_id not in runs.inputs(conn, resume_run_id):
            raise ValueError(f"run {resume_run_id} did not read C1 run {c1_run_id}")
        run_id = resume_run_id
        conn.execute("UPDATE runs SET status = 'running', error = NULL WHERE run_id = ?", (run_id,))
    else:
        run_id = runs.start(conn, "C2", config_values, inputs=[c1_run_id])

    try:
        done = {r[0] for r in conn.execute("SELECT doc_id FROM document_texts WHERE run_id = ?", (run_id,))}
        docs = conn.execute(
            "SELECT doc_id, file_path, page_count FROM documents WHERE run_id = ? ORDER BY doc_id", (c1_run_id,)
        ).fetchall()
        tasks = [
            {
                "converter": converter, "data_root": str(data_root), "doc_id": d["doc_id"], "file_path": d["file_path"],
                "expected_pages": d["page_count"], "ocr_min_chars_per_page": ocr_min,
                "raw_path": raw_path(adapter, d["doc_id"]),
            }
            for d in docs
            if d["doc_id"] not in done
        ]
        start, n_done = time.perf_counter(), len(done)

        def store(result: dict[str, Any]) -> None:
            nonlocal n_done
            with transaction(conn):
                _insert_text(conn, run_id, result, adapter)
            n_done += 1
            if progress:
                progress(f"[{n_done}/{len(docs)}] {result['doc_id']} {result['status']} {result['duration_s']:.1f} s")

        if workers <= 1:
            for task in tasks:
                store(convert_one(task))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for future in as_completed([pool.submit(convert_one, t) for t in tasks]):
                    store(future.result())

        report = _run_report(conn, run_id, c1_run_id, adapter, workers, time.perf_counter() - start, len(tasks))
        report_path = files.write_json(data_root, f"reports/{run_id}/c2_report.json", report)
        with transaction(conn):
            runs.complete(conn, run_id, report_path)
    except BaseException as exc:  # includes KeyboardInterrupt: the stored documents stay for --resume
        runs.fail(conn, run_id, f"{type(exc).__name__}: {exc}")
        raise
    return run_id


def _run_report(conn, run_id, c1_run_id, adapter, workers, wall_s, converted_now) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT doc_id, status, error, page_count, raw_reused, warnings_json, ocr_pages_json, duration_s"
        " FROM document_texts WHERE run_id = ? ORDER BY doc_id",
        (run_id,),
    ).fetchall()
    ok = [r for r in rows if r["status"] == "ok"]
    warnings = collections.Counter(w for r in ok for w in json.loads(r["warnings_json"]))
    pages_now = sum(r["page_count"] or 0 for r in ok)
    return {
        "input_runs": {"C1": c1_run_id},
        "converter": adapter.name,
        "converter_version": adapter.version(),
        "settings": adapter.settings(),
        "cleaning_version": clean.CLEANING_VERSION,
        "documents": len(rows),
        "ok": len(ok),
        "failed": [{"doc_id": r["doc_id"], "error": r["error"]} for r in rows if r["status"] == "failed"],
        "raw_reused": sum(r["raw_reused"] for r in ok),
        "warnings": dict(sorted(warnings.items())),
        "ocr_documents": sum(1 for r in ok if json.loads(r["ocr_pages_json"])),
        "pages": pages_now,
        "throughput": {
            "workers": workers,
            "documents_converted_in_last_invocation": converted_now,
            "wall_seconds_last_invocation": round(wall_s, 1),
            "converter_seconds_per_page": round(sum(r["duration_s"] or 0 for r in ok) / pages_now, 3) if pages_now else None,
        },
    }


# --- Converter comparison (SPEC-C2-11) -----------------------------------------------


@dataclass(frozen=True)
class SampleDocument:
    label: str  # why it is in the sample, e.g. "old-generation TOP call"
    name: str  # a short file-name-safe name
    pdf_path: Path
    expected_pages: int | None


RESULTS_FILE = "results.jsonl"


def _read_results(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / RESULTS_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare(adapters: Sequence[Adapter], sample: Sequence[SampleDocument], out_dir: Path) -> dict[str, Any]:
    """Convert the sample with every adapter; write the Markdown side by side and a summary.

    Each result is appended to results.jsonl as soon as it exists, so an interrupted
    comparison keeps what it has, and a rerun into the same folder skips the pairs
    already done. Converters can therefore run in separate processes, one after another.
    """
    out_dir.mkdir(parents=True, exist_ok=True)  # it may already hold the sample's input files
    done = {(r["converter"], r["document"]) for r in _read_results(out_dir)}
    for adapter in adapters:
        for doc in sample:
            if (adapter.name, doc.name) in done:
                continue
            md_path = out_dir / f"{doc.name}.{adapter.name}.md"
            if md_path.exists():
                raise FileExistsError(f"{md_path.name} exists without a result line; remove it or use a new folder")
            result = convert_document(adapter, doc.pdf_path, doc.expected_pages)
            if result.markdown:
                md_path.write_text(result.markdown, encoding="utf-8")
            row = {
                "converter": adapter.name,
                "converter_version": adapter.version(),
                "settings": adapter.settings(),
                "document": doc.name,
                "label": doc.label,
                **{k: v for k, v in asdict(result).items() if k not in ("markdown", "pages")},
                "seconds_per_page": round(result.duration_s / result.page_count, 3) if result.page_count else None,
            }
            with open(out_dir / RESULTS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            (out_dir / "summary.md").write_text(_summary_table(_read_results(out_dir)), encoding="utf-8")
    rows = _read_results(out_dir)
    return {"sample": [asdict(d) | {"pdf_path": str(d.pdf_path)} for d in sample], "results": rows}


def _summary_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Document | Converter | Status | Pages | s/page | Characters | Tables | Warnings |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["document"], r["converter"])):
        status = r["status"] if r["status"] == "ok" else f"failed: {r['error']}"
        lines.append(
            f"| {r['document']} | {r['converter']} | {status} | {r['page_count']} | {r['seconds_per_page']} | "
            f"{r['char_count']} | {r['table_count']} | {', '.join(r['warnings'])} |"
        )
    return "\n".join(lines) + "\n"
