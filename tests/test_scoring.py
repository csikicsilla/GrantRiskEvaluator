"""The scoring rules against Spec_L3_ScoreAndLabel.md, Appendix A.1, A.3 and A.4."""

from decimal import Decimal
from fractions import Fraction

import pytest

from grantrisk.labelling import scoring

# Appendix A.1: (factor, value, expected points).
BAND_EDGES = [
    ("fin_form", "grant", 3), ("fin_form", "conditional_grant", 2), ("fin_form", "loan", 1),
    ("tam_osszeg", 0, 0), ("tam_osszeg", 100_000_000, 0), ("tam_osszeg", 100_000_001, 1),
    ("tam_osszeg", 300_000_000, 1), ("tam_osszeg", 300_000_001, 2),
    ("tam_osszeg", 600_000_000, 2), ("tam_osszeg", 600_000_001, 3),
    ("konzorcium", 0, 0), ("konzorcium", 1, 3),
    ("bead_napok", 0, 3), ("bead_napok", 7, 3), ("bead_napok", 8, 2), ("bead_napok", 9, 2),
    ("bead_napok", 15, 2), ("bead_napok", 16, 1), ("bead_napok", 30, 1), ("bead_napok", 31, 0),
    ("bead_napok", "keret_kimerulesig", 0),
    ("max_tam_int", 0, 0), ("max_tam_int", 30, 0), ("max_tam_int", 30.01, 1), ("max_tam_int", 50, 1),
    ("max_tam_int", 50.01, 2), ("max_tam_int", 69, 2), ("max_tam_int", 69.5, 2),
    ("max_tam_int", 70, 3), ("max_tam_int", 100, 3),
    ("eloleg", 0, 0), ("eloleg", 30, 0), ("eloleg", 30.01, 1), ("eloleg", 50, 1),
    ("eloleg", 50.01, 2), ("eloleg", 70, 2), ("eloleg", 70.01, 3), ("eloleg", 100, 3),
    ("idotartam", 1, 0), ("idotartam", 12, 0), ("idotartam", 12.1, 1), ("idotartam", 18, 1),
    ("idotartam", 18.1, 2), ("idotartam", 24, 2), ("idotartam", 24.1, 3),
    ("tam_tevekenyseg", ["egyeb"], 0), ("tam_tevekenyseg", ["kutatas_fejlesztes"], 2),
    ("tam_tevekenyseg", ["infrastruktura_ingatlan"], 3),
    ("tam_tevekenyseg", ["kutatas_fejlesztes", "infrastruktura_ingatlan"], 3),
    ("tam_tevekenyseg", ["egyeb", "kutatas_fejlesztes"], 2),
    ("egysz_elszam", 1, 0), ("egysz_elszam", 0, 3),
    ("biztositek", 1, 0), ("biztositek", 0, 3),
]


@pytest.mark.parametrize("factor, value, expected", BAND_EDGES)
def test_band_edges(factor, value, expected):
    assert scoring.points(factor, value) == (expected, "band")


def test_band_edges_cover_every_factor():
    assert {f for f, _, _ in BAND_EDGES} == set(scoring.FACTORS)


@pytest.mark.parametrize("value", [Decimal("30.01"), Fraction(3001, 100)])
def test_exact_number_types(value):
    assert scoring.points("eloleg", value) == (1, "band")


# Appendix A.3, the TOP rule at the level of one value (SPEC-L3-04, -14).
@pytest.mark.parametrize(
    "programme, value, expected",
    [
        ("GINOP_PLUSZ", 50_000_000, (0, "band")),
        ("GINOP_PLUSZ", 700_000_000, (3, "band")),
        ("TOP", None, (1, "top_rule")),
        ("TOP_PLUSZ", None, (2, "top_rule")),
        ("TOP", 250_000_000, (1, "band")),
        ("TOP_PLUSZ", 700_000_000, (3, "band")),
        ("GINOP_PLUSZ", None, (None, None)),
        (None, None, (None, None)),
    ],
)
def test_top_rule(programme, value, expected):
    assert scoring.points("tam_osszeg", value, programme) == expected


def test_top_rule_applies_only_to_tam_osszeg():
    assert scoring.points("eloleg", None, "TOP") == (None, None)


@pytest.mark.parametrize(
    "factor, value",
    [
        ("fin_form", "grants"),
        ("tam_osszeg", -1),
        ("konzorcium", True),
        ("konzorcium", 2),
        ("bead_napok", 7.5),
        ("bead_napok", -1),
        ("bead_napok", "until funds run out"),
        ("max_tam_int", 100.5),
        ("eloleg", 120),
        ("idotartam", 0),
        ("tam_tevekenyseg", []),
        ("tam_tevekenyseg", ["egyeb", "other"]),
        ("tam_tevekenyseg", "egyeb"),
        ("egysz_elszam", "1"),
        ("max_tam_int", float("nan")),
    ],
)
def test_values_outside_the_domain_are_rejected(factor, value):
    assert scoring.domain_error(factor, value)
    with pytest.raises(ValueError):
        scoring.points(factor, value)


# Appendix A.4, fixed label (SPEC-L3-09).
@pytest.mark.parametrize(
    "score, r, label",
    [
        (Fraction(0), 0, "low"),
        (Fraction(100, 3), 33, "low"),
        (Fraction("33.4"), 33, "low"),
        (Fraction(67, 2), 34, "medium"),
        (Fraction(199, 3), 66, "medium"),
        (Fraction(133, 2), 67, "high"),
        (Fraction(100), 100, "high"),
    ],
)
def test_fixed_label(score, r, label):
    assert scoring.fixed_label(score) == (r, label)


def test_normalised():
    assert scoring.normalised(Fraction(30)) == 100
    assert scoring.normalised(Fraction(4)) == Fraction(40, 3)


def test_rules_version_is_set():
    assert scoring.RULES_VERSION
