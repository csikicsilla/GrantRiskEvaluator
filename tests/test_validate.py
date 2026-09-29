"""E1 against Spec_E1_ValidateExtraction.md: the pure computation (SPEC-E1-01 … -06)."""

from fractions import Fraction

import pytest
import yaml
from e1_helpers import GOLD, VALUE_FOR, doc_id, gold_documents

from grantrisk.evaluation import validate
from grantrisk.evaluation.validate import ALL, BASELINE, LabelCuts, Observation
from grantrisk.labelling import consolidate, scoring
from grantrisk.labelling.scoring import FACTORS

DOCS = {d.gold_name: d for d in gold_documents()}
A, C, D, E = DOCS["A Plusz-1.1.1-21"], DOCS["C Plusz-1.3.1-21"], DOCS["D-2.1.1-16"], DOCS["E-4.3.1-16"]
# The cuts a main L3 run over GOLD gives: every factor determined, terciles {B, F} {C, D} {E, A}.
CUTS = LabelCuts(
    means={f: Fraction(sum(p[i] for _, _, _, p in GOLD), len(GOLD)) for i, f in enumerate(FACTORS)},
    last_low=scoring.normalised(Fraction(6)),
    last_medium=scoring.normalised(Fraction(17)),
)


def found(value):
    return Observation(status="found", value=value)


def cell(doc, factor, observation):
    return validate.classify("regex", doc, factor, observation)


def identical_source(null_top_amount=True):
    obs = {}
    for d in DOCS.values():
        for f in FACTORS:
            if f == "tam_osszeg" and d.programme in ("TOP", "TOP_PLUSZ") and null_top_amount:
                obs[(d.doc_id, f)] = Observation(status="not_found")
            else:
                obs[(d.doc_id, f)] = found(VALUE_FOR[f][d.points[f]])
    return obs


# --- SPEC-E1-01: categories -----------------------------------------------------------------


def test_agree_and_disagree():
    assert cell(A, "eloleg", found(80)).category == "agree"  # 80% advance: 3 points, gold 3
    c = cell(A, "eloleg", found(40))
    assert (c.category, c.points, c.points_origin) == ("disagree", 1, "band")


@pytest.mark.parametrize("status", ["not_found", "ambiguous", "error"])
def test_each_missing_status_is_not_found(status):
    c = cell(A, "eloleg", Observation(status=status))
    assert (c.category, c.points) == ("not_found", None)


def test_document_missing_from_run():
    c = cell(A, "eloleg", None)
    assert (c.category, c.note) == ("not_found", "document_missing_from_run")


def test_gold_not_scored_is_not_comparable():
    doc = validate.GoldDocument("X", "X", "X", "GINOP", "2014-2020", {**A.points, "eloleg": None})
    assert cell(doc, "eloleg", found(80)).category == "not_comparable"
    assert cell(doc, "eloleg", None).category == "not_comparable"


def test_top_document_with_null_amount_agrees_through_the_top_rule():
    # DEC-31: TOP scores 1, TOP Plusz 2; both calls have exactly these gold points.
    c = cell(E, "tam_osszeg", Observation(status="not_found"))
    assert (c.category, c.points, c.points_origin) == ("agree", 1, "top_rule")
    assert cell(C, "tam_osszeg", Observation(status="error")).category == "agree"
    assert cell(D, "tam_osszeg", Observation(status="not_found")).category == "not_found"  # GINOP: no rule


def test_top_rule_does_not_apply_to_a_document_missing_from_the_run():
    assert cell(E, "tam_osszeg", None).category == "not_found"  # §4: its cells count as not_found


def test_until_funds_run_out_scores_zero():
    doc = validate.GoldDocument("X", "X", "X", "GINOP", "2014-2020", {**A.points, "bead_napok": 0})
    c = cell(doc, "bead_napok", found("keret_kimerulesig"))
    assert (c.category, c.points) == ("agree", 0)


def test_empty_activity_list_is_not_found():
    c = cell(A, "tam_tevekenyseg", found([]))
    assert (c.category, c.note) == ("not_found", "empty_activity_list")


def test_value_outside_the_domain_is_not_found():
    c = cell(A, "eloleg", found(150))
    assert c.category == "not_found"
    assert c.note.startswith("out_of_domain")


def test_baseline_points_are_used_as_recorded():
    c = validate.classify(BASELINE, A, "eloleg", Observation(status="found", points=3))
    assert (c.category, c.points, c.points_origin) == ("agree", 3, "given")


def test_tally_rates():
    t = validate.Tally()
    for c in [
        cell(A, "eloleg", found(80)),  # agree 3/3
        cell(A, "eloleg", found(80)),  # agree
        cell(A, "eloleg", found(40)),  # disagree 3/1
        cell(A, "eloleg", None),  # not_found
        validate.Cell("regex", "X", "eloleg", None, None, None, "not_comparable", None, None),
    ]:
        t.add(c)
    r = t.rates()
    assert (r["agree"], r["disagree"], r["not_found"], r["not_comparable"]) == (2, 1, 1, 1)
    assert r["agreement"] == pytest.approx(2 / 4)
    assert r["agreement_found"] == pytest.approx(2 / 3)
    assert r["coverage"] == pytest.approx(3 / 4)
    assert r["mean_abs_diff"] == pytest.approx(2 / 3)
    assert r["confusion"][3] == [0, 1, 0, 2]


def test_empty_tally_has_no_rates():
    r = validate.Tally().rates()
    assert r["agreement"] is r["agreement_found"] is r["coverage"] is r["mean_abs_diff"] is r["agreement_ci"] is None


# --- Measures (SPEC-E1-04, -05) ------------------------------------------------------------------


def test_wilson_interval():
    low, high = validate.wilson(5, 10)
    assert (low, high) == (pytest.approx(0.2366, abs=1e-4), pytest.approx(0.7634, abs=1e-4))
    assert validate.wilson(0, 0) is None
    assert validate.wilson(10, 10)[1] == pytest.approx(1.0)


def test_quadratic_kappa():
    ref = [0, 0, 1, 1, 2, 2, 2, 1, 0, 2]
    rated = [0, 1, 1, 2, 2, 2, 1, 1, 0, 0]
    # The value of sklearn.metrics.cohen_kappa_score(ref, rated, weights="quadratic").
    assert validate.quadratic_kappa(list(zip(ref, rated))) == Fraction(6, 13)
    assert validate.quadratic_kappa([(0, 0), (1, 1), (2, 2)]) == 1
    assert validate.quadratic_kappa([(1, 1), (1, 1)]) is None  # no variation: undefined
    assert validate.quadratic_kappa([]) is None


def test_tercile_label_uses_the_cut_scores():
    assert validate.tercile_label(Fraction(20), CUTS) == "low"  # equal to the last low score
    assert validate.tercile_label(Fraction(21), CUTS) == "medium"
    assert validate.tercile_label(Fraction(170, 3), CUTS) == "medium"
    assert validate.tercile_label(Fraction(58), CUTS) == "high"
    no_low = LabelCuts(CUTS.means, None, Fraction(50))
    assert validate.tercile_label(Fraction(0), no_low) == "medium"


# --- The whole computation ----------------------------------------------------------------------


def test_identical_source_agrees_everywhere_with_kappa_one():
    result = validate.compute(list(DOCS.values()), {"llm:m": identical_source()}, CUTS, "llm:m")
    for f in (*FACTORS, ALL):
        assert result.agreements[("llm:m", f, ALL)]["agreement"] == 1
    for kind in ("fixed", "tercile"):
        m = result.label_agreements[("llm:m", kind)]
        assert (m["agreement"], m["kappa_quadratic"], m["mean_total_diff"]) == (1, 1, 0)
        assert m["error_sizes"] == {"-2": 0, "-1": 0, "0": 6, "1": 0, "2": 0}
    reference = {lab.doc_id: lab for lab in result.labels if lab.source == "gold"}
    assert [reference[doc_id(n)].fixed_label for n, _, _, _ in GOLD] == ["high", "low", "medium", "medium", "high", "low"]
    assert [reference[doc_id(n)].tercile_label for n, _, _, _ in GOLD] == ["high", "low", "medium", "medium", "high", "low"]


def test_counts_by_factor_and_period():
    obs = identical_source()
    obs[(A.doc_id, "eloleg")] = found(40)  # 2021-2027: disagree
    obs[(D.doc_id, "eloleg")] = Observation(status="ambiguous")  # other period: not found
    result = validate.compute(list(DOCS.values()), {"regex": obs}, CUTS)
    a = result.agreements
    assert (a[("regex", "eloleg", ALL)]["agree"], a[("regex", "eloleg", ALL)]["disagree"],
            a[("regex", "eloleg", ALL)]["not_found"]) == (4, 1, 1)
    assert a[("regex", "eloleg", "2021-2027")]["disagree"] == 1
    assert a[("regex", "eloleg", "other")]["not_found"] == 1
    assert a[("regex", ALL, ALL)]["agree"] == 58
    assert a[("regex", ALL, "2021-2027")]["agree"] + a[("regex", ALL, "other")]["agree"] == 58


def test_missing_document_is_listed_and_labelled_with_the_means():
    obs = {k: v for k, v in identical_source().items() if k[0] != E.doc_id}
    result = validate.compute(list(DOCS.values()), {"llm:m": obs}, CUTS)
    assert result.missing_documents == {"llm:m": [E.doc_id]}
    assert result.agreements[("llm:m", ALL, ALL)]["not_found"] == 10
    label = next(lab for lab in result.labels if lab.source == "llm:m" and lab.doc_id == E.doc_id)
    # L3 would still apply the TOP rule to its NULL amount; the other nine factors get the means.
    expected = sum((CUTS.means[f] for f in FACTORS if f != "tam_osszeg"), Fraction(1))
    assert (label.total, label.n_determined) == (expected, 0)


def test_evidence_audit():
    obs = identical_source()
    obs[(A.doc_id, "eloleg")] = Observation(status="found", value=80, warnings=("evidence_not_in_text",))
    result = validate.compute(list(DOCS.values()), {"llm:m": obs}, CUTS)
    audit = result.evidence_audit["llm:m"]
    assert audit["eloleg"] == {"found": 6, "evidence_not_in_text": 1, "share": pytest.approx(1 / 6)}
    assert audit[ALL]["found"] == 58  # the two TOP amounts were not found
    assert audit["fin_form"]["share"] == 0


def test_baseline_has_no_evidence_audit():
    baseline = {(d.doc_id, f): Observation(status="found", points=d.points[f]) for d in DOCS.values() for f in FACTORS}
    result = validate.compute(list(DOCS.values()), {BASELINE: baseline}, CUTS)
    assert BASELINE not in result.evidence_audit
    assert result.agreements[(BASELINE, ALL, ALL)]["agreement"] == 1


# --- SPEC-E1-06: preferences ----------------------------------------------------------------------


def test_preferences_higher_agreement_and_tie_to_regex():
    regex, llm = identical_source(), identical_source()
    regex[(A.doc_id, "eloleg")] = found(40)  # llm better on eloleg
    llm[(A.doc_id, "idotartam")] = found(6)  # regex better on idotartam
    result = validate.compute(list(DOCS.values()), {"regex": regex, "llm:m": llm}, CUTS, "llm:m")
    prefs = result.preferences["preferred_source"]
    assert prefs["eloleg"] == "llm"
    assert prefs["idotartam"] == "regex"
    assert {prefs[f] for f in FACTORS if f not in ("eloleg", "idotartam")} == {"regex"}  # ties


def test_preferences_yaml_pastes_into_the_l2_configuration():
    prefs = {f: "llm" for f in FACTORS} | {"bead_napok": "regex"}
    text = validate.preferences_yaml(prefs, "E1-x", "llm:m")
    section = yaml.safe_load(text)["consolidate"]
    consolidate.check_preferences(section["preferred_source"])
    assert section == {"preferred_source": prefs, "preferences_from_e1_run": "E1-x"}
    assert list(section["preferred_source"]) == list(FACTORS)


def test_no_preferences_without_both_sources():
    result = validate.compute(list(DOCS.values()), {"regex": identical_source()}, CUTS, "llm:m")
    assert result.preferences["preferred_source"] is None
    assert "llm:m" in result.preferences["not_suggested_because"]
    result = validate.compute(list(DOCS.values()), {"llm:m": identical_source()}, CUTS, "llm:m")
    assert result.preferences["not_suggested_because"] == "no regex run was evaluated"


def test_source_order():
    assert validate.source_order(["old_regex", "llm:b", "regex", "llm:a", "other"]) == [
        "regex", "llm:a", "llm:b", "other", "old_regex"
    ]


def test_nothing_to_evaluate():
    with pytest.raises(validate.E1InputError, match="no source"):
        validate.compute(list(DOCS.values()), {}, CUTS)
