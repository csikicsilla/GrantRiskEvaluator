"""C1 rules against Spec_C1_Acquire.md: SPEC-C1-04, -05, -13 and Appendix A.1, A.2."""

import datetime

import pytest

from grantrisk.corpus import callcode, doctype, versions

# SPEC-C1-04: (input, call code, county suffix).
CALL_CODES = [
    ("GINOP Plusz-1.2.1-21", "GINOP_PLUSZ-1.2.1-21", None),
    ("KEHOP-Plusz-4.2.3-25", "KEHOP_PLUSZ-4.2.3-25", None),
    ("EFOP Plusz-3.4.1 -25", "EFOP_PLUSZ-3.4.1-25", None),
    ("MAHOP Plusz–1.3.1-25", "MAHOP_PLUSZ-1.3.1-25", None),
    ("KÖFOP-3.3.2.-16", "KÖFOP-3.3.2-16", None),
    ("EFOP-1.2.12-17 ", "EFOP-1.2.12-17", None),
    ("TOP-1.1.1-16-BS1", "TOP-1.1.1-16", "BS1"),
    ("TOP_PLUSZ-1.1.1-21-SB1", "TOP_PLUSZ-1.1.1-21", "SB1"),
    ("DIMOP_PLUSZ-1.1.2/A-24", "DIMOP_PLUSZ-1.1.2/A-24", None),
    ("VP5-8.2.1-16", "VP5-8.2.1-16", None),
    ("GINOP Plusz – 2.5.2-26", "GINOP_PLUSZ-2.5.2-26", None),
    ("KEHOP-1.5.0.", "KEHOP-1.5.0", None),
    ("ginop_plusz-2.1.1-24", "GINOP_PLUSZ-2.1.1-24", None),
]


@pytest.mark.parametrize("raw, code, county", CALL_CODES)
def test_normalise(raw, code, county):
    assert callcode.normalise(raw) == (code, county)


def test_county_suffix_only_for_top():
    assert callcode.normalise("VEKOP-1.1.1-19-AB1") == ("VEKOP-1.1.1-19-AB1", None)


@pytest.mark.parametrize(
    "code, programme, period",
    [
        ("GINOP_PLUSZ-2.2.1-25", "GINOP_PLUSZ", "2021-2027"),
        ("TOP_PLUSZ-1.1.1-21", "TOP_PLUSZ", "2021-2027"),
        ("TOP-1.1.1-16", "TOP", "2014-2020"),
        ("KÖFOP-3.3.2-16", "KÖFOP", "2014-2020"),
        ("RSZTOP-1.1.1-15", "RSZTOP", "2014-2020"),
        ("VEKOP-5.2.1-17", "VEKOP", "2014-2020"),
        ("RRF-1.1.2-21", "RRF", "RRF"),
        ("VP-6.4.1-16", "VP", "VP"),
        ("VP5-8.2.1-16", "VP", "VP"),
        ("GINOP-1.1.2-VEKOP-17", "GINOP", "2014-2020"),
    ],
)
def test_programme_and_period(code, programme, period):
    assert callcode.programme_and_period(code) == (programme, period)


def test_unknown_programme_stops():
    with pytest.raises(callcode.UnknownProgrammeError, match="KAP"):
        callcode.programme_and_period("KAP-1.1.1-23")


@pytest.mark.parametrize(
    "code, series",
    [
        ("GINOP_PLUSZ-2.2.1-24", "GINOP_PLUSZ-2.2.1"),
        ("GINOP_PLUSZ-2.2.1-25", "GINOP_PLUSZ-2.2.1"),
        ("VP-19.1-15", "VP-19.1"),
        ("MAHOP-2.5.1-2017", "MAHOP-2.5.1"),
        ("KEHOP_PLUSZ-1.2.21", "KEHOP_PLUSZ-1.2.21"),
    ],
)
def test_series(code, series):
    assert callcode.series(code) == series


def test_programme_family():
    assert callcode.programme_family("RRF-MMSZA-1") == "RRF"
    assert callcode.programme_family("GINOP_PLUSZ") == "GINOP_PLUSZ"


# Appendix A.1: document type by name.
@pytest.mark.parametrize(
    "name, expected",
    [
        ("felhivas.pdf", "main_call"),
        ("ginop-plusz-221-25-felhivas.pdf", "main_call"),
        ("termekleiras-ginop-826-18-hatalyos-20190402-tol.pdf", "product_description"),
        ("altalanos-utmutato.pdf", "other"),
        ("altalanos-utmutato-a-felhivasokhoz.pdf", "other"),
        ("tamogatasi-szerzodes-sablon.pdf", "other"),
        ("3szmelleklet-annex-lista.pdf", "other"),
        ("vp2-4111-16-to.pdf", "other"),
        ("to-sablon-vp6-1931-17-leader-egyuttmukodes.pdf", "other"),
        ("top-plusz-131-21-TSM.pdf", "other"),
        ("dimop-plusz-123B-24-felhívás.pdf", "main_call"),
        ("efop-731-vegleges.pdf", None),
        ("vp6-1691-17-szolidaris-gazdalkodas.pdf", None),
    ],
)
def test_doc_type_by_name(name, expected):
    assert doctype.by_name(name) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Magyarország Kormánya\nPÁLYÁZATI FELHÍVÁS\nA ...", "main_call"),
        ("MFB Zrt.\nTermékleírás\nHitelprogram", "product_description"),
        ("Eljárásrend\n1. Bevezetés", None),
        ("F\nELHÍVÁS\nA térségi gazdasági környezet", "main_call"),  # a drop capital
        ("Általános Szerződési Feltételek\na Vidékfejlesztési Program", None),
        ("\n".join(["x"] * 10 + ["felhívás"]), None),  # below the title lines
    ],
)
def test_doc_type_by_first_page(text, expected):
    assert doctype.by_first_page(text) == expected


# Appendix A.2: version choice.
@pytest.mark.parametrize(
    "names, expected",
    [
        (["felhivas-ginop-plusz-143-24-v9.pdf", "felhivas-ginop-plusz-143-24-v10.pdf"], 1),
        (["termekleiras-hatalyos-20190401-tol.pdf", "termekleiras-hatalyos-20200625-tol.pdf"], 1),
        (["vop-plusz-211-22-felhivas.pdf", "vop-plusz-211-22-felhivas-mod2.pdf"], 1),
        (["x-felhivas-mod.pdf", "x-felhivas.pdf"], 0),
        (["felhivas.pdf", "vp6-641-16-felhivas.pdf"], None),
        (["ginop-826-18-termekleiras.pdf", "termekleiras-ginop-826-18-20200611-1.pdf"], None),
        (["only.pdf"], 0),
    ],
)
def test_version_choice(names, expected):
    assert versions.choose(names) == expected


@pytest.mark.parametrize(
    "name, marker",
    [
        ("dimop-plusz-212-23-felhivas-4sz-mod.pdf", ("number", 4)),
        ("vop-plusz-415-24-felhivas-2-sz-mod.pdf", ("number", 2)),
        ("vop-plusz-111-22-felhivas-2mod.pdf", ("number", 2)),
        ("rrf-823-24-okfo-mod2.pdf", ("number", 2)),
        ("ginopplusz-211-21-mod.pdf", ("number", 1)),  # not 21: the year is not a version
        ("efop-416-16-modositas.pdf", ("number", 1)),
        ("efop-417-16-modositottdocx.pdf", ("number", 1)),
        ("ginop-plusz-511-24-felhivas-vegl.pdf", ("number", 0)),
        ("termekleiras-ginop-826-18-20200611-1.pdf", ("date", datetime.date(2020, 6, 11))),
    ],
)
def test_version_marker(name, marker):
    assert versions.marker(name) == marker
