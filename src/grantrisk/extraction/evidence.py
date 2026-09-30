"""Checking evidence quotes against the document text (SPEC-L1-02, DEC-55), for every extractor."""

from __future__ import annotations

import bisect
import re

PAGE_MARKER = re.compile(r"<!-- page (\d+) -->")
# "…", "...", also in brackets: "[…]", "(...)".
ELLIPSIS = re.compile(r"\s*[\[(]?\s*(?:\.\.\.|…)\s*[\])]?\s*")
MIN_PIECE = 20  # characters each piece of a shortened quote must have, so it cannot match by chance
MAX_SKIP = 300  # characters one ellipsis may stand for


def normalise(text: str) -> str:
    """Case-insensitive, whitespace collapsed, Markdown emphasis removed, table pipes and '#' as spaces."""
    text = re.sub(r"[*_`]", "", text)  # "**50%**." must still read "50%."
    text = re.sub(r"[|#]", " ", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def pages(markdown: str) -> list[tuple[int, str]]:
    """(page number, normalised text) for each page of a C2 Markdown text."""
    parts = PAGE_MARKER.split(markdown)
    return [(int(parts[i]), normalise(parts[i + 1])) for i in range(1, len(parts), 2)]


def _joined(page_texts: list[tuple[int, str]]) -> tuple[str, list[int], list[int]]:
    """The pages as one text, so a quote may run across a page break, and where each page starts."""
    starts, numbers, pos = [], [], 0
    for number, text in page_texts:
        starts.append(pos)
        numbers.append(number)
        pos += len(text) + 1
    return " ".join(text for _, text in page_texts), starts, numbers


def _find_pieces(text: str, pieces: list[str]) -> int | None:
    """The start of the first piece, if the pieces occur in order, each within MAX_SKIP of the one before."""
    position = text.find(pieces[0])
    while position >= 0:
        end = position + len(pieces[0])
        for piece in pieces[1:]:
            nxt = text.find(piece, end, end + MAX_SKIP + len(piece) + 1)
            if nxt < 0:
                break
            end = nxt + len(piece)
        else:
            return position
        position = text.find(pieces[0], position + 1)
    return None


def find_page(page_texts: list[tuple[int, str]], quote: str) -> int | None:
    """The page where the quote starts, or None if the quote is not in the text (DEC-55).

    The whole quote is looked for first, across page breaks too. A quote shortened with "…"
    passes only if every piece is at least MIN_PIECE characters long and the pieces occur in
    order, each "…" standing for at most MAX_SKIP characters.
    """
    whole = normalise(quote)
    if not whole or not page_texts:
        return None
    text, starts, numbers = _joined(page_texts)
    position = text.find(whole)
    if position < 0:
        pieces = [normalise(p) for p in ELLIPSIS.split(quote)]
        pieces = [p for p in pieces if p]
        if len(pieces) < 2 or any(len(p) < MIN_PIECE for p in pieces):
            return None
        position = _find_pieces(text, pieces)
        if position is None:
            return None
    return numbers[bisect.bisect_right(starts, position) - 1]
