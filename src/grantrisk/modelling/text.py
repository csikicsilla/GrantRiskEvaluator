"""The one plain text every representation reads (SPEC-M1-01).

Derived from the C2 Markdown: page markers, table separator rows, table pipes and
heading marks are removed; the words, numbers and table contents stay.
"""

from __future__ import annotations

import hashlib
import re

PLAIN_TEXT_VERSION = "1"  # part of every feature's provenance through the text hash

PAGE_MARKER = re.compile(r"<!-- page \d+ -->")
SEPARATOR_ROW = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
HEADING = re.compile(r"^\s*#{1,6}(\s+|$)")
SPACES = re.compile(r"[ \t]+")
BLANK_LINES = re.compile(r"\n{3,}")


def plain_text(markdown: str) -> str:
    """SPEC-M1-01: the plain text of a C2 Markdown text."""
    lines = []
    for line in PAGE_MARKER.sub("\n", markdown).split("\n"):
        if SEPARATOR_ROW.match(line):
            continue
        if line.lstrip().startswith("|"):
            line = line.replace("|", " ")
        else:
            line = HEADING.sub("", line)
        lines.append(SPACES.sub(" ", line).strip())
    return BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
