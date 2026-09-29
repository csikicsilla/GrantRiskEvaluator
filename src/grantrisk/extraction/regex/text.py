"""Searching a C2 Markdown text for the regex rules (SPEC-L1-05).

The rules search a *view* of the Markdown that has the same length as the Markdown:
emphasis marks, table pipes and '#' become spaces, so a phrase matches whether it
sits in a paragraph or in a table cell, and every position in the view is the same
position in the Markdown. Evidence is always cut from the Markdown itself, so it is
verbatim (SPEC-L1-02).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import date

from grantrisk.extraction import evidence

MARKUP = re.compile(r"[*_`|#]")
# A line of a table of contents: a dot leader or an ellipsis followed by a page number.
TOC_LINE = re.compile(r"(?:\.{4,}|…{2,}|(?:\. ){4,})\s*\**\s*\d{1,3}\s*\**\s*\|?\s*$")
# Where a quote may start or end: a line break, a table cell, or the end of a sentence.
BOUNDARY = re.compile(r"\n|\||(?<=[.?!;])\s")
QUOTE_REACH = 250  # characters a quote may extend beyond the matched span on each side
# Where the answer to an anchor ends: the next heading, or the next question of a summary table.
HEADING_LINE = re.compile(r"^#", re.MULTILINE)
QUESTION_ROW = re.compile(r"^\|[^\n]{0,250}?\?", re.MULTILINE)
SENTENCE_END = re.compile(r"\n|\||(?<=[.?!])\s(?=\S)")

MONTHS = {
    "január": 1, "február": 2, "március": 3, "április": 4, "május": 5, "június": 6,
    "július": 7, "augusztus": 8, "szeptember": 9, "október": 10, "november": 11, "december": 12,
}

# An amount of money: "49,29 millió Ft", "29,65 Mrd forint", "500 ezer forint", "629 300 000 Ft", "5.000.000,- Ft".
MONEY = re.compile(
    r"(?<![\d.,])(?:(\d{1,4}(?:[.,]\d+)?)\s*(milliárd|mrd\.?|millió|ezer)\s*(?:Ft|forint|HUF\b)"
    r"|(\d{1,3}(?:[. ]\d{3})+|\d+)\s*(?:,-)?\s*(?:Ft|forint|HUF\b))",  # "Ft" may be glued to the next word
    re.IGNORECASE,
)
MULTIPLIERS = {"milliárd": 1_000_000_000, "mrd": 1_000_000_000, "millió": 1_000_000, "ezer": 1_000}
PERCENT = re.compile(r"(?<![\d.,])(\d{1,3}(?:[.,]\d{1,2})?)\s*%")
# Dates: "2021. július 12.", "2017. év február hó 27." or "2025.01.31" / "2025. 01. 31."
DATE_WORDS = re.compile(
    r"(20\d\d)\.\s*(?:év\s+)?(" + "|".join(MONTHS) + r")\s*(?:hó(?:nap)?\s+)?(\d{1,2})\b", re.IGNORECASE
)
DATE_DIGITS = re.compile(r"(20\d\d)\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\b")


def pattern(phrase: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    """A phrase as a regex in which every space matches any run of whitespace (line breaks too)."""
    return re.compile(re.sub(r" +", r"\\s+", phrase), flags)


class Document:
    """A C2 Markdown text prepared for searching."""

    def __init__(self, markdown: str) -> None:
        self.markdown = markdown
        self.view = MARKUP.sub(" ", markdown)
        self._page_texts: list[tuple[int, str]] | None = None

    def is_toc(self, pos: int) -> bool:
        """Whether ``pos`` is on a line of a table of contents."""
        line_start = self.view.rfind("\n", 0, pos) + 1
        line_end = self.view.find("\n", pos)
        return bool(TOC_LINE.search(self.markdown[line_start : len(self.markdown) if line_end < 0 else line_end]))

    def anchors(self, regex: re.Pattern[str]) -> Iterator[re.Match[str]]:
        """Every match of an anchor, outside the table of contents."""
        for m in regex.finditer(self.view):
            if not self.is_toc(m.start()):
                yield m

    def after(self, m: re.Match[str], chars: int, before: int = 0) -> tuple[int, int]:
        """The window from ``before`` characters before an anchor to ``chars`` characters after it."""
        return max(0, m.start() - before), min(len(self.view), m.end() + chars)

    def line_start(self, pos: int) -> int:
        return self.view.rfind("\n", 0, pos) + 1

    def line_end(self, pos: int) -> int:
        end = self.view.find("\n", pos)
        return len(self.view) if end < 0 else end

    def segment(self, m: re.Match[str], cap: int, before: int = 0) -> tuple[int, int]:
        """The answer to an anchor: from the anchor (or ``before`` characters earlier on its line)
        to the next heading or summary-table question after the anchor's line, at most ``cap`` characters."""
        start = max(self.line_start(m.start()), m.start() - before)
        limit = min(len(self.markdown), m.end() + cap)
        after_line = self.line_end(m.end())
        for stop in (HEADING_LINE, QUESTION_ROW):
            s = stop.search(self.markdown, after_line, limit)
            if s:
                limit = s.start()
        return start, max(limit, m.end())

    def sentence(self, pos: int, reach: int = 400) -> tuple[int, int]:
        """The sentence (or table cell, or line) around ``pos``."""
        lo = max(0, pos - reach)
        for s in SENTENCE_END.finditer(self.markdown, lo, pos):
            lo = s.end()
        s = SENTENCE_END.search(self.markdown, pos, min(len(self.markdown), pos + reach))
        return lo, s.start() if s else min(len(self.markdown), pos + reach)

    def tables(self) -> Iterator[tuple[int, int]]:
        """The spans of the Markdown tables: runs of consecutive lines that start with '|'."""
        for m in re.finditer(r"(?:^\|[^\n]*(?:\n|$))+", self.markdown, re.MULTILINE):
            yield m.start(), m.end()

    def page_of(self, pos: int) -> int | None:
        """The page of a position: the nearest page marker before it."""
        last = None
        for m in evidence.PAGE_MARKER.finditer(self.markdown, 0, pos):
            last = int(m.group(1))
        return last

    def quote(self, start: int, end: int) -> tuple[str, int]:
        """The verbatim sentence(s) around ``start``–``end``, within one page and without leading or
        trailing markup, and the position where the quote starts."""
        lo = start
        for m in BOUNDARY.finditer(self.markdown, max(0, start - QUOTE_REACH), start):
            lo = m.end()
        m = BOUNDARY.search(self.markdown, end, min(len(self.markdown), end + QUOTE_REACH))
        hi = m.start() if m else min(len(self.markdown), end + QUOTE_REACH)
        # Never cross a page marker: keep the part that holds the end of the span.
        for mk in evidence.PAGE_MARKER.finditer(self.markdown, lo, hi):
            if mk.start() < end:
                lo = max(lo, mk.end())
            else:
                hi = min(hi, mk.start())
                break
        text = self.markdown[lo:hi]
        stripped = text.lstrip(" \t\n|-•*#>")
        lo += len(text) - len(stripped)
        return stripped.rstrip(" \t\n|-•*#"), lo

    def page_texts(self) -> list[tuple[int, str]]:
        if self._page_texts is None:
            self._page_texts = evidence.pages(self.markdown)
        return self._page_texts


# --- Parsing numbers and dates ----------------------------------------------------------


def money(m: re.Match[str]) -> int | None:
    """The amount of a MONEY match in forints."""
    if m.group(2):
        base = float(m.group(1).replace(",", "."))
        return round(base * MULTIPLIERS[m.group(2).lower().rstrip(".")])
    digits = re.sub(r"[. ]", "", m.group(3))
    return int(digits) if digits.isdigit() else None


def percent(m: re.Match[str]) -> float:
    return float(m.group(1).replace(",", "."))


def dates(text: str, offset: int = 0) -> list[tuple[date, int, int]]:
    """The valid dates in ``text`` in order of appearance, with their spans (plus ``offset``)."""
    found = []
    for m in DATE_WORDS.finditer(text):
        found.append((m, int(m.group(1)), MONTHS[m.group(2).lower()], int(m.group(3))))
    for m in DATE_DIGITS.finditer(text):
        found.append((m, int(m.group(1)), int(m.group(2)), int(m.group(3))))
    result = []
    for m, y, mo, d in sorted(found, key=lambda x: x[0].start()):
        try:
            result.append((date(y, mo, d), m.start() + offset, m.end() + offset))
        except ValueError:
            continue
    return result
