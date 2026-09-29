"""Choosing one version per call from the file names (SPEC-C1-07, rule 2)."""

from __future__ import annotations

import datetime
import re
from collections.abc import Sequence

from grantrisk.corpus.doctype import fold

_DATE = re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")
_NUMBERED = [
    re.compile(r"(?<![\d.])(\d+)-?sz-?mod"),  # 2-sz-mod, 4sz-mod
    re.compile(r"mod-?(\d+)(?!\d)"),  # mod2, mod-3
    re.compile(r"(?:^|[-_])(\d+)mod"),  # 2mod
    re.compile(r"(?:^|[-_])v(\d+)(?=[-_.]|$)"),  # v9, v10
]
_UNNUMBERED = re.compile(r"(?:^|[-_])(?:mod|modositas|modositott\w*)(?=[-_.]|$)")


def marker(original_name: str) -> tuple[str, datetime.date | int]:
    """The version marker of a file name: ('date', date) or ('number', n).

    An unnumbered mod/modositas/modositott counts as 1, a name without a marker as 0.
    """
    name = fold(original_name)
    m = _DATE.search(name)
    if m:
        return "date", datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for pattern in _NUMBERED:
        m = pattern.search(name)
        if m:
            return "number", int(m.group(1))
    if _UNNUMBERED.search(name):
        return "number", 1
    return "number", 0


def choose(original_names: Sequence[str]) -> int | None:
    """The index of the latest version, or None if the names cannot be ordered."""
    if len(original_names) == 1:
        return 0
    markers = [marker(n) for n in original_names]
    if len({kind for kind, _ in markers}) != 1:
        return None  # a date against a number
    values = [value for _, value in markers]
    best = max(values)
    if values.count(best) > 1:
        return None  # e.g. two names without a marker
    return values.index(best)
