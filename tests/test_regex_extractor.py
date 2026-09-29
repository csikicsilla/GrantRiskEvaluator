"""The regex extractor for one document: output contract, evidence and text helpers (SPEC-L1-01, -02)."""

import datetime

import pytest

from grantrisk.extraction import evidence
from grantrisk.extraction.regex import extractor, rules
from grantrisk.extraction.regex.text import MONEY, Document, dates, money
from grantrisk.labelling import scoring
from grantrisk.labelling.scoring import FACTORS

MARKDOWN = (
    "<!-- page 1 -->\n"
    "|A támogatás visszatérítendő vagy vissza nem térítendő?|A támogatás vissza nem térítendő támogatásnak minősül.|\n"
    "|Mikor lehet benyújtani a támogatási kérelmet?|2023.10.20. – 2023.11.10.|\n\n"
    "<!-- page 2 -->\n"
    "## 7.1. Mennyi előleg igényelhető?\n"
    "Az igényelhető támogatási előleg mértéke legfeljebb a megítélt támogatás **25%-a**.\n"
)


def test_one_observation_per_factor_and_the_contract():
    obs = extractor.extract(MARKDOWN)
    assert list(obs) == list(FACTORS)
    for f, o in obs.items():
        assert (o.status == "found") == (o.value is not None), f
        assert o.status in ("found", "not_found", "ambiguous", "error")
        if o.value is not None:
            assert scoring.domain_error(f, o.value) is None, f
        else:
            assert o.evidence == "", f


def test_values_evidence_and_pages():
    obs = extractor.extract(MARKDOWN)
    assert (obs["fin_form"].value, obs["fin_form"].evidence_page) == ("grant", 1)
    assert (obs["bead_napok"].value, obs["bead_napok"].evidence_page) == (21, 1)
    assert (obs["eloleg"].value, obs["eloleg"].evidence_page) == (25, 2)
    assert isinstance(obs["eloleg"].value, int)  # 25.0 is stored as 25
    assert obs["biztositek"].value == 0 and obs["biztositek"].warnings == ["default_value"]


def test_evidence_is_verbatim_and_found_in_the_text():
    obs = extractor.extract(MARKDOWN)
    pages = evidence.pages(MARKDOWN)
    for f, o in obs.items():
        if o.evidence:
            assert o.evidence in MARKDOWN, f
            assert evidence.find_page(pages, o.evidence) == o.evidence_page, f
            assert "evidence_not_in_text" not in o.warnings, f


def test_evidence_never_crosses_a_page_marker():
    md = "<!-- page 1 -->\nAz igényelhető vissza nem térítendő\n<!-- page 2 -->\ntámogatás összege 50 000 000 Ft.\n"
    doc = Document(md)
    quote, start = doc.quote(md.index("Az igényelhető"), md.index(" Ft.") + 3)
    assert "<!--" not in quote and doc.page_of(start) == 2


def test_a_failing_rule_gives_error_for_its_factor_only(monkeypatch):
    def broken(doc):
        raise RuntimeError("boom")

    monkeypatch.setitem(rules.RULES, "idotartam", broken)
    obs = extractor.extract(MARKDOWN)
    assert obs["idotartam"].status == "error" and obs["idotartam"].warnings == ["RuntimeError: boom"]
    assert obs["fin_form"].status == "found"


def test_an_out_of_domain_value_is_an_error(monkeypatch):
    monkeypatch.setitem(rules.RULES, "eloleg", lambda doc: rules.found(150, (0, 10)))
    o = extractor.extract(MARKDOWN)["eloleg"]
    assert (o.status, o.value) == ("error", None)


def test_empty_text_gives_ten_observations():
    obs = extractor.extract("")
    assert len(obs) == 10
    assert obs["biztositek"].value == 0  # never NULL, as the definition requires
    assert all(o.status == "not_found" for f, o in obs.items() if f != "biztositek")


@pytest.mark.parametrize(
    "text, amount",
    [
        ("629 300 000 Ft", 629_300_000),
        ("629.300.000 Ft", 629_300_000),
        ("50 000 000,-Ft", 50_000_000),
        ("49,29 millió forint", 49_290_000),
        ("1,2 milliárd Ft", 1_200_000_000),
        ("8,24 Mrd Ft", 8_240_000_000),
        ("500 ezer forint", 500_000),
        ("175 670 000 000Ft", 175_670_000_000),
    ],
)
def test_money(text, amount):
    assert money(MONEY.search(text)) == amount


@pytest.mark.parametrize(
    "text, day",
    [
        ("2021. július 12.", datetime.date(2021, 7, 12)),
        ("2025.01.31", datetime.date(2025, 1, 31)),
        ("2023. 09. 25.", datetime.date(2023, 9, 25)),
        ("2017. év február hó 27. naptól", datetime.date(2017, 2, 27)),
        ("2016. év szeptember hónap 19. naptól", datetime.date(2016, 9, 19)),
    ],
)
def test_dates(text, day):
    assert [d for d, _, _ in dates(text)] == [day]


def test_invalid_dates_are_skipped():
    assert dates("2023.02.30 és 2023. február 31.") == []
