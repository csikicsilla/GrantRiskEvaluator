"""C2 cleaning and conversion logic (SPEC-C2-05 … -07, -09, -11), with a fake converter."""

import json

import pytest

from grantrisk.corpus import clean, convert
from grantrisk.corpus.converters import ConversionError

# --- SPEC-C2-06 Cleaning -------------------------------------------------------------


def test_characters():
    text = "érték ﬁnanszírozás és­támogatás​"
    assert clean.normalise_characters(text) == "érték finanszírozás éstámogatás"


@pytest.mark.parametrize(
    "lines, expected",
    [
        (["a támoga-", "tás összege"], ["a támogatás összege"]),  # joined
        (["a 2021-", "2027 közötti"], ["a 2021-", "2027 közötti"]),  # a digit follows: kept
        (["az EU-", "támogatás"], ["az EU-", "támogatás"]),  # an upper-case letter precedes: kept
        (["az elő-", "leg mértéke"], ["az előleg mértéke"]),  # Hungarian letters
        (["kis-", "- és közép"], ["kis-", "- és közép"]),  # a list item follows: kept
    ],
)
def test_hyphenation(lines, expected):
    assert clean.join_hyphenation(lines) == expected


def body(page):
    """Eight lines of body text that differ from page to page."""
    return [f"a(z) {letter}{page} bekezdés" for letter in "abcdefgh"]


def test_repeated_footer_with_changing_page_numbers_is_removed():
    pages = [["Pályázati felhívás GINOP", *body(chr(96 + i)), f"{i}. oldal"] for i in range(1, 6)]
    cleaned = clean.remove_page_furniture(pages)
    assert cleaned[0] == body("a")
    assert all(p == body(chr(96 + i)) for i, p in enumerate(cleaned, start=1))


def test_a_line_on_two_of_ten_pages_is_kept():
    pages = [["Fejezet", f"szöveg {i}"] for i in range(10)]
    pages[0].insert(0, "Egyszeri cím")
    pages[1].insert(0, "Egyszeri cím")
    assert "Egyszeri cím" in clean.remove_page_furniture(pages)[0]


def test_page_numbers_at_the_edge_are_removed():
    assert clean.remove_page_furniture([["12", "szöveg", "folytatás", "a", "b", "c", "d", "- 13 -"]]) == [
        ["szöveg", "folytatás", "a", "b", "c", "d"]
    ]


def test_table_rows_are_never_changed():
    header = "| Megnevezés   |  Összeg |"
    pages = [[header, "|---|---|", f"| sor {i}  |  {i}  |", "szöveg"] for i in range(5)]
    cleaned = clean.remove_page_furniture(pages)
    assert all(p[0] == header for p in cleaned)  # repeated table headers survive
    assert clean.tidy_whitespace(["| a  |  b |", "x   y"]) == ["| a  |  b |", "x y"]


def test_whitespace():
    assert clean.tidy_whitespace(["a   b", "", "", "", "c  ", ""]) == ["a b", "", "c"]


def test_assemble_adds_page_markers():
    md = clean.assemble(["első oldal", "", "harmadik"])
    assert md == "<!-- page 1 -->\nelső oldal\n\n<!-- page 2 -->\n\n<!-- page 3 -->\nharmadik\n"


def test_count_tables():
    md = "| a | b |\n|---|---|\n| 1 | 2 |\n\nszöveg\n\n| c |\n| :--- |\n| 3 |\n"
    assert clean.count_tables(md) == 2


# --- SPEC-C2-05, -07, -09: conversion with a fake converter ----------------------------


class Fake:
    name = "fake"

    def __init__(self, pages, ocr_pages=None, supports_ocr=True):
        self._pages, self._ocr_pages, self.supports_ocr = pages, ocr_pages, supports_ocr
        self.calls = []

    def version(self):
        return "fake 1.0"

    def settings(self):
        return {"x": 1}

    def convert(self, pdf_path, ocr=False):
        self.calls.append(ocr)
        if ocr:
            if not self.supports_ocr:
                raise ConversionError("needs_ocr: no OCR engine")
            return self._ocr_pages
        return self._pages


LONG_PAGE = "támogatás " * 120  # 1200 characters


def distinct_pages(n):
    return [f"oldal {chr(97 + i)} " + LONG_PAGE for i in range(n)]


def test_ok_document():
    r = convert.convert_document(Fake(distinct_pages(8)), "x.pdf", expected_pages=8)
    assert r.status == "ok"
    assert r.page_count == 8
    assert r.markdown.count("<!-- page ") == 8
    assert r.warnings == []  # 9600 characters, no OCR, pages match


def test_warnings():
    r = convert.convert_document(Fake(["õ " + LONG_PAGE]), "x.pdf", expected_pages=2)
    assert r.warnings == ["low_text", "page_count_mismatch", "suspect_encoding"]


def test_ocr_fallback():
    fake = Fake(["kevés"] * 3, ocr_pages=[LONG_PAGE] * 3)
    r = convert.convert_document(fake, "x.pdf", expected_pages=3)
    assert fake.calls == [False, True]
    assert r.ocr_pages == [1, 2, 3]
    assert "ocr_used" in r.warnings


def test_needs_ocr_fails_the_document():
    r = convert.convert_document(Fake(["kevés"], supports_ocr=False), "x.pdf")
    assert r.status == "failed"
    assert "needs_ocr" in r.error


def test_converter_error_fails_the_document():
    class Broken(Fake):
        def convert(self, pdf_path, ocr=False):
            raise RuntimeError("password protected")

    r = convert.convert_document(Broken([]), "x.pdf")
    assert (r.status, r.error) == ("failed", "RuntimeError: password protected")


def test_stored_raw_output_is_reused():
    fake = Fake([LONG_PAGE])
    raw = {"pages": ["tárolt " * 200], "ocr_pages": [], "duration_s": 4.2}
    r = convert.convert_document(fake, "x.pdf", raw=raw)
    assert fake.calls == []
    assert (r.raw_reused, r.duration_s) == (True, 4.2)
    assert "tárolt" in r.markdown


def test_same_input_same_markdown():
    a = convert.convert_document(Fake([LONG_PAGE, "| a |\n|---|\n| 1 |"]), "x.pdf").markdown
    b = convert.convert_document(Fake([LONG_PAGE, "| a |\n|---|\n| 1 |"]), "x.pdf").markdown
    assert a == b


def test_raw_path_names_converter_version_and_settings():
    path = convert.raw_path(Fake([]), "abc123")
    assert path.startswith("raw/fake/fake_1.0/") and path.endswith("/abc123.json")


# --- SPEC-C2-11: comparison ------------------------------------------------------------


def test_compare_writes_markdown_and_results(tmp_path):
    sample = [convert.SampleDocument("a call", "call", tmp_path / "c.pdf", 2)]
    summary = convert.compare([Fake(distinct_pages(2))], sample, tmp_path / "cmp")
    assert (tmp_path / "cmp" / "call.fake.md").exists()
    lines = (tmp_path / "cmp" / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["page_count"] == 2
    assert "| call | fake | ok | 2 |" in (tmp_path / "cmp" / "summary.md").read_text(encoding="utf-8")
    assert summary["results"][0]["seconds_per_page"] is not None


def test_compare_resumes_and_adds_converters(tmp_path):
    (tmp_path / "cmp" / "input").mkdir(parents=True)  # a folder that already holds input files
    sample = [convert.SampleDocument("a call", "call", tmp_path / "c.pdf", 1)]
    first = Fake([LONG_PAGE])
    convert.compare([first], sample, tmp_path / "cmp")
    again = Fake([LONG_PAGE])
    convert.compare([again], sample, tmp_path / "cmp")  # already done: skipped
    assert again.calls == []

    class Other(Fake):
        name = "other"

    convert.compare([Other([LONG_PAGE])], sample, tmp_path / "cmp")  # a second converter, separate process
    assert {r["converter"] for r in convert._read_results(tmp_path / "cmp")} == {"fake", "other"}


def test_compare_never_overwrites_an_orphan_file(tmp_path):
    (tmp_path / "cmp").mkdir()
    (tmp_path / "cmp" / "call.fake.md").write_text("left over", encoding="utf-8")
    sample = [convert.SampleDocument("a call", "call", tmp_path / "c.pdf", 1)]
    with pytest.raises(FileExistsError):
        convert.compare([Fake([LONG_PAGE])], sample, tmp_path / "cmp")
