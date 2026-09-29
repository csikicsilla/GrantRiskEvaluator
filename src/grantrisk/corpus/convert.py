"""Stage C2: convert (Spec_C2_Convert.md) — the per-document conversion and the converter comparison.

``convert_document`` is shared by the comparison (SPEC-C2-11) and the stage run: it
calls the adapter, falls back to OCR for documents without a usable text layer
(SPEC-C2-05), cleans the output (SPEC-C2-06) and adds the warnings (SPEC-C2-07).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from grantrisk.corpus import clean
from grantrisk.corpus.converters import Adapter, settings_hash

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
