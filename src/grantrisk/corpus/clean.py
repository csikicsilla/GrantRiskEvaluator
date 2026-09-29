"""Cleaning the converter's output and assembling the Markdown (SPEC-C2-06, Spec_C2_Convert.md §2.2).

The steps are deterministic and versioned by CLEANING_VERSION. Table rows and page
markers are never changed by steps 2–4.
"""

from __future__ import annotations

import collections
import re
import unicodedata
from collections.abc import Sequence

CLEANING_VERSION = "1"

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
SPECIAL_SPACES = re.compile(r"[   -   　]")
INVISIBLE = re.compile(r"[­​‌‍﻿]")  # soft hyphen and zero-width characters
LOWER = "a-záéíóöőúüűä"
HYPHEN_AT_END = re.compile(rf"[{LOWER}]-$")
STARTS_LOWER = re.compile(rf"^[{LOWER}]")
PAGE_NUMBER = re.compile(r"^[-–]?\s*\d{1,4}\s*[-–]?$|^\d{1,4}\s*/\s*\d{1,4}$")
TABLE_SEPARATOR = re.compile(r"^\|\s*:?-{3,}")
FURNITURE_LINES = 3  # step 2 looks at the first and last three lines of each page
FURNITURE_SHARE = 0.3
LOW_TEXT_CHARS = 8000
SUSPECT_CHARACTERS = "õûÕÛ"  # the usual mis-encodings of ő and ű


def page_marker(n: int) -> str:
    return f"<!-- page {n} -->"


def is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def normalise_characters(text: str) -> str:
    """Step 1: NFC, ligatures, special spaces, soft hyphens."""
    text = unicodedata.normalize("NFC", text)
    for lig, letters in LIGATURES.items():
        text = text.replace(lig, letters)
    text = SPECIAL_SPACES.sub(" ", text)
    return INVISIBLE.sub("", text)


def _edge_indices(lines: list[str]) -> list[int]:
    """Indices of the first and last three non-empty, non-table lines of a page."""
    content = [i for i, line in enumerate(lines) if line.strip() and not is_table_row(line)]
    return sorted(set(content[:FURNITURE_LINES] + content[-FURNITURE_LINES:]))


def _furniture_key(line: str) -> str:
    return re.sub(r"\d", "#", line.strip())


def remove_page_furniture(pages: list[list[str]]) -> list[list[str]]:
    """Step 2: lines repeated at the top or bottom of > 30% of the pages (and ≥ 2), and page numbers."""
    counts = collections.Counter()
    for lines in pages:
        counts.update({_furniture_key(lines[i]) for i in _edge_indices(lines)})
    furniture = {k for k, c in counts.items() if c >= 2 and c > FURNITURE_SHARE * len(pages)}
    cleaned = []
    for lines in pages:
        edges = set(_edge_indices(lines))
        cleaned.append(
            [
                line
                for i, line in enumerate(lines)
                if not (i in edges and (_furniture_key(line) in furniture or PAGE_NUMBER.match(line.strip())))
            ]
        )
    return cleaned


def join_hyphenation(lines: list[str]) -> list[str]:
    """Step 3: join a word broken at a line end only between lower-case letters."""
    out: list[str] = []
    for line in lines:
        prev = out[-1] if out else None
        if (
            prev is not None
            and not is_table_row(prev)
            and not is_table_row(line)
            and HYPHEN_AT_END.search(prev.rstrip())
            and STARTS_LOWER.match(line.lstrip())
        ):
            out[-1] = prev.rstrip()[:-1] + line.lstrip()
        else:
            out.append(line)
    return out


def tidy_whitespace(lines: list[str]) -> list[str]:
    """Step 4: collapse runs of spaces inside a line; at most one empty line in a row."""
    out: list[str] = []
    for line in lines:
        if not is_table_row(line):
            line = re.sub(r"[ \t]+", " ", line).strip()
        if not line.strip() and (not out or not out[-1].strip()):
            continue
        out.append(line.rstrip() if is_table_row(line) else line)
    while out and not out[-1].strip():
        out.pop()
    return out


def assemble(page_texts: Sequence[str]) -> str:
    """Clean the per-page output of a converter and join it with page markers."""
    pages = [normalise_characters(t).split("\n") for t in page_texts]
    pages = remove_page_furniture(pages)
    parts = []
    for n, lines in enumerate(pages, start=1):
        body = "\n".join(tidy_whitespace(join_hyphenation(lines)))
        parts.append(page_marker(n) + ("\n" + body if body else ""))
    return "\n\n".join(parts) + "\n"


def count_tables(markdown: str) -> int:
    return sum(1 for line in markdown.split("\n") if TABLE_SEPARATOR.match(line.strip()))


def warnings(markdown: str, page_count: int, expected_pages: int | None, ocr_pages: Sequence[int]) -> list[str]:
    """SPEC-C2-07: warnings that do not stop the document."""
    found = []
    if len(markdown) < LOW_TEXT_CHARS:
        found.append("low_text")
    if ocr_pages:
        found.append("ocr_used")
    if expected_pages is not None and page_count != expected_pages:
        found.append("page_count_mismatch")
    if any(c in markdown for c in SUSPECT_CHARACTERS):
        found.append("suspect_encoding")
    return found
