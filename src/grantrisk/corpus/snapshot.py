"""Reading the frozen snapshot of the old code line (SPEC-C1-01, DEC-23). Nothing here writes."""

from __future__ import annotations

import csv
import sqlite3
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MANUAL_FOLDER = "data/raw_pdfs/"


@dataclass(frozen=True)
class InputFile:
    source: str  # scraped | manual
    original_name: str  # the portal's file name (scraped) or the collected file name (manual)
    location: str  # where the file was read from, for the run report
    raw_call_code: str | None
    read: Callable[[], bytes]
    scrape_programme: str | None = None
    tender_status: str | None = None


def scraped_files(pdf_dir: Path, scrape_log: Path) -> list[InputFile]:
    """One InputFile per 'downloaded' row of the old scrape log."""
    files = []
    with open(scrape_log, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["outcome"] != "downloaded":
                continue
            stored = PurePosixPath(row["file_path"].replace("\\", "/")).name
            path = pdf_dir / stored
            files.append(
                InputFile(
                    source="scraped",
                    original_name=row["doc_title"],
                    location=f"raw_pdfs/{stored}",
                    raw_call_code=row["tender_code"],
                    read=path.read_bytes,
                    scrape_programme=row["program"],
                    tender_status=row["tender_status"] or None,
                )
            )
    return files


def manual_files(zip_path: Path, call_codes: Mapping[str, str]) -> list[InputFile]:
    """The PDFs in data/raw_pdfs/ of Pipeline_V0.zip, with the call codes of the import configuration."""
    archive = zipfile.ZipFile(zip_path)
    members = sorted(
        n for n in archive.namelist() if n.startswith(MANUAL_FOLDER) and n != MANUAL_FOLDER and not n.endswith("/")
    )
    missing = [PurePosixPath(n).name for n in members if PurePosixPath(n).name not in call_codes]
    if missing:
        raise ValueError(f"the import configuration has no call code for the manual files {missing}")
    return [
        InputFile(
            source="manual",
            original_name=PurePosixPath(n).name,
            location=f"{zip_path.name}:{n}",
            raw_call_code=call_codes[PurePosixPath(n).name],
            read=lambda n=n: archive.read(n),
        )
        for n in members
    ]


def provenance(old_database: Path) -> dict[str, tuple[str | None, str | None]]:
    """SHA-256 → (source URL, download time) from the old database, opened read-only."""
    uri = f"{old_database.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        return {h: (url, when) for h, url, when in conn.execute("SELECT file_hash, source_url, download_date FROM documents")}
    finally:
        conn.close()
