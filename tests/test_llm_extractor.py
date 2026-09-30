"""The LLM extractor's logic without the network (SPEC-L1-02, -07, -09, -10)."""

import json

import pytest

from grantrisk.extraction import evidence
from grantrisk.extraction.llm import extractor
from grantrisk.extraction.llm.extractor import Observation, ResponseError
from grantrisk.labelling.scoring import FACTORS

MD = (
    "<!-- page 1 -->\n# Felhívás\nA felhívás célja a vállalkozások támogatása.\n\n"
    "<!-- page 2 -->\n| Kérdés | Válasz |\n|---|---|\n| Mennyi előleg igényelhető? | Az előleg mértéke **50%**. |\n"
)


def answers(**values):
    base = {f: {"value": None, "evidence": ""} for f in FACTORS}
    for f, (v, e) in values.items():
        base[f] = {"value": v, "evidence": e}
    return base


def response(data, stop="end_turn"):
    return {"stop_reason": stop, "content": [{"type": "text", "text": json.dumps(data)}], "usage": {}}


def test_schema_requires_every_factor_and_allows_the_open_ended_period():
    schema = extractor.output_schema()
    assert schema["required"] == list(FACTORS)
    assert schema["additionalProperties"] is False
    options = schema["properties"]["bead_napok"]["properties"]["value"]["anyOf"]
    assert {"type": "string", "enum": ["keret_kimerulesig"]} in options


def test_request_omits_temperature_the_model_rejects():
    with_t = extractor.build_request("m", "P", MD, max_tokens=100, temperature=0, thinking=None)
    without = extractor.build_request("m", "P", MD, max_tokens=100, temperature=None, thinking="disabled")
    assert with_t["temperature"] == 0 and "thinking" not in with_t
    assert "temperature" not in without and without["thinking"] == {"type": "disabled"}
    assert with_t["system"] == "P" and with_t["messages"] == [{"role": "user", "content": MD}]
    assert with_t["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.parametrize(
    "bad, message",
    [
        (response(answers(), stop="refusal"), "refusal"),
        (response(answers(), stop="max_tokens"), "cut off"),
        ({"stop_reason": "end_turn", "content": []}, "no text block"),
        ({"stop_reason": "end_turn", "content": [{"type": "text", "text": "{not json"}]}, "not valid JSON"),
        (response({f: {"value": None, "evidence": ""} for f in FACTORS[:-1]}), "biztositek"),
    ],
)
def test_unusable_responses(bad, message):
    with pytest.raises(ResponseError, match=message):
        extractor.parse_response(bad)


def test_observations():
    data = extractor.parse_response(response(answers(
        eloleg=(50, "Az előleg mértéke 50%."),
        max_tam_int=(150, "száz ötven"),
        tam_tevekenyseg=([], ""),
        idotartam=(24, "a projekt 24 hónap alatt"),
    )))
    obs = extractor.observe(data, evidence.pages(MD))
    assert (obs["eloleg"].status, obs["eloleg"].value, obs["eloleg"].evidence_page) == ("found", 50, 2)
    assert obs["eloleg"].warnings == []
    assert (obs["max_tam_int"].status, obs["max_tam_int"].value) == ("error", None)  # 150% is outside the domain
    assert (obs["tam_tevekenyseg"].status, obs["tam_tevekenyseg"].value) == ("found", [])  # kept for L2 (DEC-33)
    assert obs["idotartam"].warnings == ["evidence_not_in_text"]
    assert obs["fin_form"].status == "not_found"


@pytest.mark.parametrize(
    "quote, page",
    [
        ("A FELHÍVÁS célja   a vállalkozások", 1),  # case and whitespace
        ("előleg mértéke 50%", 2),  # Markdown emphasis in the text
        ("Mennyi előleg igényelhető? … Az előleg mértéke 50%", 2),  # shortened; every piece is long (DEC-55)
        ("Mennyi előleg igényelhető? ... 50%", None),  # "50%" is too short to prove anything
        ("A felhívás célja … 50%.", None),
        ("vállalkozások támogatása. Kérdés Válasz", 1),  # across a page break: the page where it starts
        ("ez nincs a szövegben", None),
        ("", None),
    ],
)
def test_find_page(quote, page):
    assert evidence.find_page(evidence.pages(MD), quote) == page


def test_an_ellipsis_stands_for_a_limited_stretch_of_text():
    def md(words):
        return "<!-- page 1 -->\nA támogatás maximális összege " + "szöveg " * words + "legfeljebb 50 000 000 Ft lehet.\n"

    quote = "A támogatás maximális összege … legfeljebb 50 000 000 Ft lehet."
    assert evidence.find_page(evidence.pages(md(30)), quote) == 1  # 210 characters left out
    assert evidence.find_page(evidence.pages(md(100)), quote) is None  # 700 characters left out


def test_split_and_plan_parts():
    md = "".join(f"<!-- page {i} -->\n" + "x" * 100 + "\n" for i in range(1, 7))
    pages = extractor.split_pages(md)
    assert len(pages) == 6 and "".join(pages) == md
    parts = extractor.plan_parts(md, len, 250)
    assert len(parts) == 3 and "".join(parts) == md
    assert all(p.startswith("<!-- page ") for p in parts)
    with pytest.raises(ResponseError, match="single page"):
        extractor.plan_parts(md, len, 50)


def found(value, page=1):
    return Observation(value=value, evidence="q", evidence_page=page, status="found")


def parts_with(factor, *values):
    out = []
    for v in values:
        p = {f: Observation() for f in FACTORS}
        if v is not None:
            p[factor] = found(v)
        out.append(p)
    return out


@pytest.mark.parametrize(
    "factor, values, expected",
    [
        ("bead_napok", (30, 20, "keret_kimerulesig"), 20),  # the shortest
        ("tam_osszeg", (100, 300), 300),  # the largest
        ("eloleg", (50, None, 70), 70),
        ("tam_tevekenyseg", (["egyeb"], ["kutatas_fejlesztes"]), ["egyeb", "kutatas_fejlesztes"]),
        ("biztositek", (0, 1), 1),
        ("fin_form", ("grant", "conditional_grant"), "conditional_grant"),
        ("idotartam", (24, 24), 24),
    ],
)
def test_merge_rules(factor, values, expected):
    merged = extractor.merge_parts(parts_with(factor, *values))[factor]
    assert (merged.status, merged.value) == ("found", expected)
    assert "split_document" in merged.warnings


def test_merge_conflict_without_a_rule_is_ambiguous():
    merged = extractor.merge_parts(parts_with("konzorcium", 0, 1))
    assert merged["konzorcium"].status == "ambiguous"
    assert merged["fin_form"].status == "not_found"


def test_cost():
    assert extractor.cost({"input_tokens": 2_000_000, "output_tokens": 100_000}, {"input": 1.0, "output": 5.0}) == 2.5
