"""The scoring rules: factor domains, bands, the TOP rule and the label thresholds.

This is the only module that holds a band edge or a label threshold (SPEC-L3-13).
It mirrors Spec_ScoringSystem.md §1.2, §1.2.1 and §1.4, and Spec_L3_ScoreAndLabel.md
§2.1 and SPEC-L3-01. Change RULES_VERSION whenever a rule changes.
"""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any

RULES_VERSION = "2 (Spec_ScoringSystem.md as of 2026-09-30: the loan rule of DEC-40, idotartam maximalis of DEC-37)"

FACTORS = (
    "fin_form",
    "tam_osszeg",
    "konzorcium",
    "bead_napok",
    "max_tam_int",
    "eloleg",
    "idotartam",
    "tam_tevekenyseg",
    "egysz_elszam",
    "biztositek",
)

FIN_FORM_POINTS = {"grant": 3, "conditional_grant": 2, "loan": 1}
ACTIVITY_POINTS = {"kutatas_fejlesztes": 2, "infrastruktura_ingatlan": 3, "egyeb": 0}
UNTIL_FUNDS_RUN_OUT = "keret_kimerulesig"
LONGEST_DURATION = "maximalis"  # idotartam: the longest duration, more than 24 months (DEC-37)

# DEC-31: tam_osszeg of TOP and TOP Plusz calls that state no amount.
TOP_RULE_POINTS = {"TOP": 1, "TOP_PLUSZ": 2}

# DEC-40: max_tam_int of a loan, whatever the call states: a loan is paid back in full.
LOAN_RULE_POINTS = 0
RULE_ORIGINS = ("top_rule", "loan_rule")  # points set by a rule, not from a value (ARC-04)

# The points the manual (expert) source may give (Spec_L3_ScoreAndLabel.md §2.1).
ALLOWED_POINTS = {f: frozenset({0, 1, 2, 3}) for f in FACTORS} | {
    "fin_form": frozenset({1, 2, 3}),
    "konzorcium": frozenset({0, 3}),
    "egysz_elszam": frozenset({0, 3}),
    "biztositek": frozenset({0, 3}),
    "tam_tevekenyseg": frozenset({0, 2, 3}),
}

LOW, MEDIUM, HIGH = "low", "medium", "high"
LABELS = (LOW, MEDIUM, HIGH)

MAX_TOTAL = 30  # 10 factors × 3 points
FIXED_LOW_MAX = 33  # fixed label: 0–33 low, 34–66 medium, 67–100 high (§1.4, DEC-13)
FIXED_MEDIUM_MAX = 66


def as_number(value: Any) -> Fraction | None:
    """The exact value of a finite number, or None if ``value`` is not one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        # repr gives the shortest decimal form, so 30.01 becomes 3001/100 exactly.
        return Fraction(repr(value)) if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return Fraction(value) if value.is_finite() else None
    if isinstance(value, (int, Fraction)):
        return Fraction(value)
    return None


def _is_flag(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value in (0, 1)


def domain_error(factor: str, value: Any) -> str | None:
    """Why ``value`` is outside the domain of ``factor``, or None if it is inside."""
    if factor == "fin_form":
        ok = value in FIN_FORM_POINTS
    elif factor in ("konzorcium", "egysz_elszam", "biztositek"):
        ok = _is_flag(value)
    elif factor == "bead_napok":
        n = as_number(value)
        ok = value == UNTIL_FUNDS_RUN_OUT or (n is not None and n >= 0 and n.denominator == 1)
    elif factor == "tam_osszeg":
        n = as_number(value)
        ok = n is not None and n >= 0
    elif factor in ("max_tam_int", "eloleg"):
        n = as_number(value)
        ok = n is not None and 0 <= n <= 100
    elif factor == "idotartam":
        n = as_number(value)
        ok = value == LONGEST_DURATION or (n is not None and n > 0)
    elif factor == "tam_tevekenyseg":
        ok = isinstance(value, list) and len(value) > 0 and all(v in ACTIVITY_POINTS for v in value)
    else:
        return f"unknown factor {factor!r}"
    return None if ok else f"value {value!r} is outside the domain of {factor}"


def _band(factor: str, value: Any) -> int:
    """SPEC-L3-01: the points of a value inside the factor's domain."""
    if factor == "fin_form":
        return FIN_FORM_POINTS[value]
    if factor == "konzorcium":
        return 3 if value == 1 else 0
    if factor in ("egysz_elszam", "biztositek"):
        return 0 if value == 1 else 3
    if factor == "tam_tevekenyseg":
        return max(ACTIVITY_POINTS[v] for v in value)
    if factor == "bead_napok" and value == UNTIL_FUNDS_RUN_OUT:
        return 0  # the longest submission period, more than 30 days
    if factor == "idotartam" and value == LONGEST_DURATION:
        return 3  # the longest duration, more than 24 months (DEC-37)
    x = as_number(value)
    if factor == "tam_osszeg":
        if x <= 100_000_000:
            return 0
        if x <= 300_000_000:
            return 1
        if x <= 600_000_000:
            return 2
        return 3
    if factor == "bead_napok":
        if x <= 7:
            return 3
        if x <= 15:
            return 2
        if x <= 30:
            return 1
        return 0
    if factor == "max_tam_int":
        if x <= 30:
            return 0
        if x <= 50:
            return 1
        if x < 70:
            return 2
        return 3
    if factor == "eloleg":
        if x <= 30:
            return 0
        if x <= 50:
            return 1
        if x <= 70:
            return 2
        return 3
    if factor == "idotartam":
        if x <= 12:
            return 0
        if x <= 18:
            return 1
        if x <= 24:
            return 2
        return 3
    raise ValueError(f"unknown factor {factor!r}")


def fin_form_of(value: Any = None, manual_points: int | None = None) -> str | None:
    """The financing form of a document from its fin_form value, or from the expert's fin_form points."""
    if value in FIN_FORM_POINTS:
        return value
    by_points = {p: form for form, p in FIN_FORM_POINTS.items()}
    return by_points.get(manual_points)


def points(
    factor: str, value: Any, programme: str | None = None, fin_form: str | None = None
) -> tuple[int | None, str | None]:
    """SPEC-L3-14: the points and their origin for one extracted value.

    Applies the bands (SPEC-L3-01), the TOP rule (SPEC-L3-04) and the loan rule (DEC-40),
    but not imputation. ``fin_form`` is the document's financing form: for a loan,
    ``max_tam_int`` gets LOAN_RULE_POINTS whatever its value. Returns ``(None, None)`` for a
    missing value that no rule covers. Raises ValueError for a value outside the factor's domain.
    """
    error = None if value is None else domain_error(factor, value)
    if error:
        raise ValueError(error)
    if factor == "max_tam_int" and fin_form == "loan":
        return LOAN_RULE_POINTS, "loan_rule"
    if value is None:
        if factor == "tam_osszeg" and programme in TOP_RULE_POINTS:
            return TOP_RULE_POINTS[programme], "top_rule"
        return None, None
    return _band(factor, value), "band"


def normalised(total: Fraction) -> Fraction:
    """SPEC-L3-08: the total score rescaled to 0–100."""
    return total * 100 / MAX_TOTAL


def fixed_label(normalised_score: Fraction) -> tuple[int, str]:
    """SPEC-L3-09: round half up to an integer, then apply the fixed thresholds."""
    r = math.floor(normalised_score + Fraction(1, 2))
    if r <= FIXED_LOW_MAX:
        return r, LOW
    if r <= FIXED_MEDIUM_MAX:
        return r, MEDIUM
    return r, HIGH
