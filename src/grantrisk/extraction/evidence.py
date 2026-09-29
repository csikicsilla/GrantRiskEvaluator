"""Checking evidence quotes against the document text (SPEC-L1-02), for every extractor."""

from __future__ import annotations

import re

PAGE_MARKER = re.compile(r"<!-- page (\d+) -->")
ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
MIN_FRAGMENT = 3


def normalise(text: str) -> str:
    """Case-insensitive, whitespace collapsed, Markdown emphasis removed, table pipes and '#' as spaces."""
    text = re.sub(r"[*_`]", "", text)  # "**50%**." must still read "50%."
    text = re.sub(r"[|#]", " ", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def pages(markdown: str) -> list[tuple[int, str]]:
    """(page number, normalised text) for each page of a C2 Markdown text."""
    parts = PAGE_MARKER.split(markdown)
    return [(int(parts[i]), normalise(parts[i + 1])) for i in range(1, len(parts), 2)]


def find_page(page_texts: list[tuple[int, str]], quote: str) -> int | None:
    """The first page that contains the quote; a quote shortened with '...' must match in order."""
    fragments = [normalise(f) for f in ELLIPSIS.split(quote)]
    fragments = [f for f in fragments if len(f) >= MIN_FRAGMENT]
    if not fragments:
        return None
    for number, text in page_texts:
        position = 0
        for fragment in fragments:
            position = text.find(fragment, position)
            if position < 0:
                break
            position += len(fragment)
        else:
            return number
    return None
