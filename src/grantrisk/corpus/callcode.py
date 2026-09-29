"""Call codes, programmes and periods (SPEC-C1-04, -05, -13).

The same normalisation serves the documents (C1) and the gold import (L1), so gold
rows and documents meet on the same code.
"""

from __future__ import annotations

import re

PLUSZ = ("DIMOP", "EFOP", "GINOP", "IKOP", "KEHOP", "MAHOP", "TOP", "VOP")
OLD_GENERATION = ("EFOP", "GINOP", "IKOP", "KEHOP", "KÖFOP", "MAHOP", "RSZTOP", "TOP", "VEKOP")

PROGRAMMES: dict[str, tuple[str, str]] = {f"{p}_PLUSZ": (f"{p}_PLUSZ", "2021-2027") for p in PLUSZ}
PROGRAMMES |= {p: (p, "2014-2020") for p in OLD_GENERATION}
PROGRAMMES["RRF"] = ("RRF", "RRF")

_VP = re.compile(r"^VP\d?$")
_COUNTY_SUFFIX = re.compile(r"-([A-Z]{2}\d)$")
_YEAR_SUFFIX = re.compile(r"-(\d{2}|\d{4})$")
_PLUSZ = re.compile(r"[\s_-]*plusz", re.IGNORECASE)


class UnknownProgrammeError(ValueError):
    """A call code whose prefix is not in the closed list of SPEC-C1-05."""


def normalise(raw: str) -> tuple[str, str | None]:
    """SPEC-C1-04: the normalised call code, and the county suffix removed from it (or None)."""
    code = raw.strip().replace("–", "-").replace("—", "-")
    code = _PLUSZ.sub("_PLUSZ", code)
    code = re.sub(r"\s+", "", code).upper()
    code = code.replace(".-", "-").rstrip(".")
    county = None
    if code.split("-")[0] in ("TOP", "TOP_PLUSZ"):
        m = _COUNTY_SUFFIX.search(code)
        if m:
            county = m.group(1)
            code = code[: m.start()]
    return code, county


def series(code: str) -> str:
    """SPEC-C1-13: the call code without a final year suffix of two or four digits."""
    return _YEAR_SUFFIX.sub("", code)


def programme_and_period(code: str) -> tuple[str, str]:
    """SPEC-C1-05: the programme and period of a normalised call code."""
    prefix = code.split("-")[0]
    if prefix in PROGRAMMES:
        return PROGRAMMES[prefix]
    if _VP.match(prefix):
        return "VP", "VP"
    raise UnknownProgrammeError(f"unknown programme prefix {prefix!r} in {code!r}")


def programme_family(scrape_log_programme: str) -> str:
    """The programme column of the old scrape log, in the form of SPEC-C1-05 (for the cross-check)."""
    p = scrape_log_programme.strip().upper()
    return "RRF" if p.startswith("RRF") else p
