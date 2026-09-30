"""L3 computation against Spec_L3_ScoreAndLabel.md, Appendix A.2 … A.6."""

import random
from fractions import Fraction

import pytest
from l3_helpers import BASE_TOTAL, doc

from grantrisk.labelling.label import FactorInput, L3InputError, compute, tercile_split

# --- A.2 Imputation example ---------------------------------------------------------


def test_imputation_example():
    # idotartam 20 → 2, 15 → 1, 30 → 3 points; eloleg 40 → 1 point.
    docs = [
        doc("D1", idotartam=20, eloleg=40),
        doc("D2", idotartam=15, eloleg=40),
        doc("D3", idotartam=30, eloleg=40),
        doc("D4", idotartam=None, eloleg=40),
    ]
    result = compute(docs)
    d4 = {r.doc_id: r for r in result.documents}["D4"]
    assert result.report["factor_means"]["idotartam"]["exact"] == "2"
    assert result.report["factor_means"]["idotartam"]["n_determined"] == 3
    assert d4.factors["idotartam"].points == 2
    assert d4.factors["idotartam"].origin == "mean"
    assert d4.total == 5  # fin_form 2 (conditional grant, DEC-40) + idotartam 2 + eloleg 1
    assert d4.normalised == Fraction(50, 3)


def test_all_points_three_give_the_maximum():
    top = dict(
        fin_form="grant", tam_osszeg=700_000_000, konzorcium=1, bead_napok=5, max_tam_int=80,
        eloleg=80, idotartam=36, tam_tevekenyseg=["infrastruktura_ingatlan"], egysz_elszam=0, biztositek=0,
    )
    result = compute([doc("MAX", **top), doc("BASE")])
    r = {r.doc_id: r for r in result.documents}["MAX"]
    assert r.total == 30
    assert r.normalised == 100


# --- A.3 TOP rule --------------------------------------------------------------------


def test_top_rule_and_its_exclusion_from_the_mean():
    docs = [
        doc("D1", "GINOP_PLUSZ", tam_osszeg=50_000_000),
        doc("D2", "GINOP_PLUSZ", tam_osszeg=700_000_000),
        doc("D3", "TOP", tam_osszeg=None),
        doc("D4", "TOP_PLUSZ", tam_osszeg=None),
        doc("D5", "TOP", tam_osszeg=250_000_000),
        doc("D6", "TOP_PLUSZ", tam_osszeg=700_000_000),
        doc("D7", "GINOP_PLUSZ", tam_osszeg=None),
    ]
    by_id = {r.doc_id: r.factors["tam_osszeg"] for r in compute(docs).documents}
    expected = {
        "D1": (0, "band"), "D2": (3, "band"), "D3": (1, "top_rule"), "D4": (2, "top_rule"),
        "D5": (1, "band"), "D6": (3, "band"), "D7": (Fraction(7, 4), "mean"),
    }
    assert {d: (r.points, r.origin) for d, r in by_id.items()} == expected


def test_manual_points_are_not_affected_by_the_top_rule():
    docs = [doc("G1"), doc("T1", "TOP", tam_osszeg=FactorInput(points=3))]
    r = {r.doc_id: r for r in compute(docs).documents}["T1"]
    assert (r.factors["tam_osszeg"].points, r.factors["tam_osszeg"].origin) == (3, "manual")


# --- A.4 Tercile label ---------------------------------------------------------------


def sizes(labels):
    values = list(labels.values())
    return values.count("low"), values.count("medium"), values.count("high")


def scores(*values):
    return [(Fraction(v), f"d{i:02d}") for i, v in enumerate(values)]


def test_t1_nine_distinct_scores():
    assert sizes(tercile_split(scores(*range(9)))[0]) == (3, 3, 3)


def test_t2_ten_distinct_scores():
    labels, info = tercile_split(scores(*range(10)))
    assert sizes(labels) == (3, 4, 3)
    assert (info["cut_1"]["position"], info["cut_2"]["position"]) == (3, 7)


def test_t3_a_tied_group_is_not_split():
    labels, info = tercile_split(scores(1, 2, 3, 3, 3, 3, 4, 5, 6))
    assert sizes(labels) == (2, 4, 3)
    assert info["cut_1"]["tied_group"] == {"score": "3", "size": 4, "went_to": "higher class"}


def test_t4_equal_distance_takes_the_smaller_position():
    labels, info = tercile_split(scores(1, 2, 2, 3, 4, 5))
    assert sizes(labels) == (1, 3, 2)  # DEC-32
    assert info["cut_1"]["tied_group"]["went_to"] == "higher class"


def test_t5_all_scores_equal():
    labels, _ = tercile_split(scores(*[5] * 6))
    assert sizes(labels) == (0, 6, 0)


def test_t6_input_order_does_not_matter():
    s = scores(*range(10))
    shuffled = s[:]
    random.Random(1).shuffle(shuffled)
    assert tercile_split(s)[0] == tercile_split(shuffled)[0]


def test_t7_equal_exact_scores_stay_together():
    # 1/10 + 2/10 and 3/10 are the same number, but not as floats.
    a, b = Fraction(1, 10) + Fraction(2, 10), Fraction(3, 10)
    exact = [(Fraction(0), "a0"), (a, "A"), (b, "B"), (Fraction(1), "c1"), (Fraction(2), "c2"), (Fraction(3), "c3")]
    labels, _ = tercile_split(exact)
    assert labels["A"] == labels["B"]
    # With floats the same split separates them, which is why L3 uses fractions (SPEC-L3-11).
    floats = [(0.0, "a0"), (0.1 + 0.2, "A"), (0.3, "B"), (1.0, "c1"), (2.0, "c2"), (3.0, "c3")]
    float_labels, _ = tercile_split(floats)
    assert float_labels["A"] != float_labels["B"]


def test_tercile_report_names_the_last_scores_of_each_class():
    _, info = tercile_split(scores(*range(9)))
    assert (info["last_low_score"], info["last_medium_score"]) == ("2", "5")


def test_labels_come_from_compute():
    docs = [doc(f"D{i}", idotartam=v) for i, v in enumerate([6, 15, 20, 30, 6, 15, 20, 30, 6])]
    result = compute(docs)
    assert sum(r.tercile_label == "low" for r in result.documents) == result.report["tercile_cuts"]["class_sizes"]["low"]


# --- A.5 Coverage --------------------------------------------------------------------


def test_document_without_any_determined_factor():
    empty = doc("Z", **{f: None for f in ("fin_form", "tam_osszeg", "konzorcium", "bead_napok", "max_tam_int",
                                          "eloleg", "idotartam", "tam_tevekenyseg", "egysz_elszam", "biztositek")})
    result = compute([doc("A"), doc("B"), empty])
    z = {r.doc_id: r for r in result.documents}["Z"]
    assert z.n_determined == 0
    assert z.low_coverage
    assert all(fr.origin == "mean" for fr in z.factors.values())
    assert z.total == BASE_TOTAL  # every mean equals the base document's points
    assert z.fixed_label and z.tercile_label


def test_low_coverage_threshold():
    four = doc("FOUR", tam_osszeg=None, konzorcium=None, bead_napok=None, max_tam_int=None,
               eloleg=None, idotartam=None)  # 4 determined
    five = doc("FIVE", tam_osszeg=None, konzorcium=None, bead_napok=None, max_tam_int=None,
               eloleg=None)  # 5 determined
    result = compute([doc("A"), four, five])
    by_id = {r.doc_id: r for r in result.documents}
    assert (by_id["FIVE"].n_determined, by_id["FIVE"].low_coverage) == (5, False)
    assert (by_id["FOUR"].n_determined, by_id["FOUR"].low_coverage) == (4, True)


def test_coverage_counts_add_up():
    docs = [doc("A"), doc("B", eloleg=None), doc("C", "TOP", tam_osszeg=None)]
    report = compute(docs).report
    for counts in report["coverage"]["by_factor"].values():
        assert sum(counts.values()) == 3
    assert report["coverage"]["by_factor"]["tam_osszeg"] == {"band": 2, "top_rule": 1}


# --- A.6 Invalid input and other errors ----------------------------------------------


def nine_rows():
    d = doc("NINE")
    return type(d)(d.doc_id, d.programme, {f: v for f, v in d.factors.items() if f != "biztositek"})


@pytest.mark.parametrize(
    "bad, message",
    [
        (nine_rows(), "missing factor rows"),
        (doc("X", eloleg=120), "outside the domain"),
        (doc("X", tam_tevekenyseg=[]), "outside the domain"),
        (doc("X", egysz_elszam=FactorInput(points=1)), "not allowed"),
        (doc("X", eloleg=FactorInput(value=40, points=1)), "both a value and points"),
    ],
)
def test_invalid_input_is_rejected(bad, message):
    with pytest.raises(L3InputError, match=message):
        compute([doc("A"), bad])


def test_no_documents():
    with pytest.raises(L3InputError, match="no documents"):
        compute([])


def test_factor_determined_in_no_document():
    with pytest.raises(L3InputError, match="bead_napok"):
        compute([doc("A", bead_napok=None), doc("B", bead_napok=None)])


# --- Report --------------------------------------------------------------------------


def test_warnings():
    report = compute([doc("A"), doc("B")]).report
    assert "fewer than 3 documents: terciles are not meaningful" in report["warnings"]
    assert "all documents have the same score" in report["warnings"]
    assert "the tercile class 'low' is empty" in report["warnings"]


def test_totals_are_exact_fractions():
    docs = [doc("A", idotartam=15), doc("B", idotartam=20), doc("C", idotartam=20), doc("D", idotartam=None)]
    d = {r.doc_id: r for r in compute(docs).documents}["D"]
    assert d.factors["idotartam"].points == Fraction(5, 3)
    assert isinstance(d.total, Fraction)


# --- DEC-40: the loan rule ------------------------------------------------------------------------


def test_loan_rule_sets_intensity_and_is_left_out_of_the_mean():
    docs = [
        doc("D1", max_tam_int=20),                         # 0 points, band
        doc("D2", max_tam_int=80),                         # 3 points, band
        doc("D3", fin_form="loan", max_tam_int=90),        # the stated 90 % does not count
        doc("D4", fin_form="loan", max_tam_int=None),
        doc("D5", fin_form=FactorInput(points=1), max_tam_int=None),  # the expert's fin_form points say loan
        doc("D6", max_tam_int=None),                       # imputed from D1 and D2 only
    ]
    result = {r.doc_id: r for r in compute(docs).documents}
    assert [(result[d].factors["max_tam_int"].points, result[d].factors["max_tam_int"].origin)
            for d in ("D1", "D2", "D3", "D4", "D5", "D6")] == [
        (0, "band"), (3, "band"), (0, "loan_rule"), (0, "loan_rule"), (0, "loan_rule"), (Fraction(3, 2), "mean")]
    assert result["D3"].n_determined == result["D1"].n_determined - 1  # a rule is not a determined value


def test_manual_intensity_points_are_not_affected_by_the_loan_rule():
    docs = [doc("D1"), doc("D2", fin_form="loan", max_tam_int=FactorInput(points=2))]
    r = {r.doc_id: r for r in compute(docs).documents}["D2"]
    assert (r.factors["max_tam_int"].points, r.factors["max_tam_int"].origin) == (2, "manual")
