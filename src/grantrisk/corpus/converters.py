"""Converter adapters (SPEC-C2-02). Each turns a PDF into Markdown per page.

The rest of C2 (cleaning, checks, storage) does not depend on the adapter. Heavy
libraries are imported only when an adapter is created, and they run locally
(SPEC-C2-03): Docling's models must be in the local cache before a run.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any, Protocol

IMAGE_PLACEHOLDER = re.compile(r"^\s*<!--\s*image\s*-->\s*$", re.MULTILINE)
CELL_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)


class ConversionError(RuntimeError):
    pass


class Adapter(Protocol):
    name: str
    supports_ocr: bool

    def version(self) -> str: ...

    def settings(self) -> dict[str, Any]: ...

    def convert(self, pdf_path: Path, ocr: bool = False) -> list[str]:
        """Markdown per page, in page order."""
        ...


def settings_hash(adapter: Adapter) -> str:
    return hashlib.sha256(json.dumps(adapter.settings(), sort_keys=True).encode()).hexdigest()[:12]


def _table_cells_on_one_line(markdown: str) -> str:
    """Convention of §2.2: a line break inside a table cell becomes a space."""
    return "\n".join(
        CELL_BREAK.sub(" ", line) if line.lstrip().startswith("|") else line for line in markdown.split("\n")
    )


class PyMuPDF4LLM:
    """pymupdf4llm with its layout model (a small network run on the CPU through PyMuPDF Layout).

    Its own OCR is switched off; C2's fallback (SPEC-C2-05) asks for OCR explicitly.
    Page headers and footers found by the layout model are left out.
    """

    name = "pymupdf4llm"
    supports_ocr = True  # through RapidOCR, installed with pymupdf4llm

    def __init__(self):
        import pymupdf4llm  # noqa: F401

    def version(self) -> str:
        return f"pymupdf4llm {importlib.metadata.version('pymupdf4llm')}, pymupdf {importlib.metadata.version('pymupdf')}"

    def settings(self) -> dict[str, Any]:
        return {"layout": True, "header": False, "footer": False}

    def convert(self, pdf_path: Path, ocr: bool = False) -> list[str]:
        import pymupdf4llm

        pymupdf4llm.use_layout(True)
        chunks = pymupdf4llm.to_markdown(
            str(pdf_path), page_chunks=True, write_images=False, show_progress=False,
            header=False, footer=False, use_ocr=ocr, force_ocr=ocr,
        )
        return [_table_cells_on_one_line(c["text"]) for c in chunks]


class PyMuPDF4LLMLegacy:
    """pymupdf4llm's older rule-based path: no model at all; tables through PyMuPDF's table finder."""

    name = "pymupdf4llm-legacy"
    supports_ocr = False

    def __init__(self, table_strategy: str = "lines_strict"):
        import pymupdf4llm  # noqa: F401

        self.table_strategy = table_strategy

    def version(self) -> str:
        return f"pymupdf4llm {importlib.metadata.version('pymupdf4llm')}, pymupdf {importlib.metadata.version('pymupdf')}"

    def settings(self) -> dict[str, Any]:
        return {"layout": False, "table_strategy": self.table_strategy}

    def convert(self, pdf_path: Path, ocr: bool = False) -> list[str]:
        if ocr:
            raise ConversionError("needs_ocr: the rule-based path of pymupdf4llm has no OCR")
        import pymupdf4llm

        pymupdf4llm.use_layout(False)
        chunks = pymupdf4llm.to_markdown(
            str(pdf_path), page_chunks=True, write_images=False, show_progress=False,
            table_strategy=self.table_strategy,
        )
        return [_table_cells_on_one_line(c["text"]) for c in chunks]


class Docling:
    """Docling: layout and table-structure models, run on the CPU."""

    name = "docling"

    def __init__(self, num_threads: int = 4, batch_size: int = 1):
        import importlib.util

        from docling.datamodel import pipeline_options as po
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption

        # OCR only as the safety net of SPEC-C2-05, on whole pages. EasyOCR reads Hungarian;
        # RapidOCR (installed with Docling) has Chinese/English models only.
        if importlib.util.find_spec("easyocr"):
            ocr_options, self.ocr_engine = po.EasyOcrOptions(lang=["hu", "en"], mode=po.OcrMode.FULL_PAGE), "easyocr"
        elif importlib.util.find_spec("rapidocr"):
            ocr_options, self.ocr_engine = po.RapidOcrOptions(mode=po.OcrMode.FULL_PAGE), "rapidocr"
        else:
            ocr_options, self.ocr_engine = None, None
        self.supports_ocr = ocr_options is not None
        # Frugal with memory on a 16 GB laptop: few threads, one page per batch.
        self.num_threads, self.batch_size = num_threads, batch_size
        self._converters = {}
        for ocr in (False, True):
            if ocr and not self.supports_ocr:
                continue
            options = po.PdfPipelineOptions(
                do_ocr=ocr,
                do_table_structure=True,
                accelerator_options=AcceleratorOptions(num_threads=num_threads, device="cpu"),
                layout_batch_size=batch_size,
                table_batch_size=batch_size,
                ocr_batch_size=batch_size,
            )
            if ocr:
                options.ocr_options = ocr_options
            self._converters[ocr] = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )

    def version(self) -> str:
        return f"docling {importlib.metadata.version('docling')}, docling-core {importlib.metadata.version('docling-core')}"

    def settings(self) -> dict[str, Any]:
        return {"table_structure": True, "ocr_engine": self.ocr_engine, "threads": self.num_threads, "batch_size": self.batch_size}

    def convert(self, pdf_path: Path, ocr: bool = False) -> list[str]:
        if ocr and not self.supports_ocr:
            raise ConversionError("needs_ocr: no OCR engine available for Docling")
        from io import BytesIO

        from docling.datamodel.base_models import DocumentStream

        # A stream, not a path: Docling's C++ parser cannot open paths with non-ASCII letters ("Kód").
        source = DocumentStream(name=Path(pdf_path).name, stream=BytesIO(Path(pdf_path).read_bytes()))
        doc = self._converters[ocr].convert(source).document
        pages = []
        for page_no in sorted(doc.pages):
            md = doc.export_to_markdown(page_no=page_no)
            pages.append(_table_cells_on_one_line(IMAGE_PLACEHOLDER.sub("", md)))
        return pages


ADAPTERS = {"pymupdf4llm": PyMuPDF4LLM, "pymupdf4llm-legacy": PyMuPDF4LLMLegacy, "docling": Docling}


def create(name: str, **options: Any) -> Adapter:
    if name not in ADAPTERS:
        raise ValueError(f"unknown converter {name!r}; choose one of {sorted(ADAPTERS)}")
    return ADAPTERS[name](**options)
