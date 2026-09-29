"""Acceptance of the regex rules against the gold set and the old baseline (SPEC-L1-06)."""

from pathlib import Path

import pytest

from grantrisk import config as config_mod
from grantrisk.extraction.regex import acceptance
from grantrisk.extraction.regex.acceptance import GoldDocument
from grantrisk.labelling.scoring import FACTORS

TEXT = (
    "<!-- page 1 -->\n"
    "|Nyújthat be támogatási kérelmet konzorcium?|A támogatási kérelem benyújtására konzorciumi formában nincs lehetőség.|\n"
    "|Mennyi támogatást lehet igényelni?|maximum 400 000 000 Ft|\n"
)


def points(**given):
    return {f: given.get(f, 3) for f in FACTORS}


def test_agreement_counts_like_e1():
    pairs = [(3, 3), (3, 0), (2, None), (None, 1)]
    assert acceptance.agreement(pairs) == {"agree": 1, "disagree": 1, "not_found": 1, "not_comparable": 1}


def test_old_baseline_with_renamed_calls(tmp_path):
    path = tmp_path / "baseline.csv"
    path.write_text(
        "﻿document_id,felhivas,factor,gold,regex,e5,haiku\n"
        "9,GINOP Plusz-2.2.1-24,fin_form,3.0,3.0,,3.0\n"
        "9,GINOP Plusz-2.2.1-24,tam_osszeg,3.0,,,3.0\n",
        encoding="utf-8",
    )
    old = acceptance.read_old_baseline(path, {"GINOP Plusz-2.2.1-24": "GINOP Plusz-2.2.1-25"})
    assert old == {"GINOP Plusz-2.2.1-25": {"fin_form": 3, "tam_osszeg": None}}


def test_compare_counts_new_and_old_on_documents_with_text():
    docs = [
        GoldDocument("A", "a", "GINOP_PLUSZ", points(konzorcium=0, tam_osszeg=2), points(konzorcium=3), TEXT),
        GoldDocument("B", "b", "GINOP_PLUSZ", points(), points(), None),
    ]
    report = acceptance.compare(docs)
    assert (report["documents"], report["documents_with_text"], report["missing_texts"]) == (2, 1, ["B"])
    assert report["factors"]["konzorcium"]["new"]["agree"] == 1
    assert report["factors"]["konzorcium"]["old"]["disagree"] == 1
    assert report["factors"]["tam_osszeg"]["new"]["agree"] == 1  # 400 000 000 Ft → 2 points
    assert report["factors"]["idotartam"]["new"]["not_found"] == 1
    assert "| konzorcium | 1 | 0 | 0 | 0 | 1 | 0 | yes |" in acceptance.format_table(report)


def test_top_rule_applies_to_a_missing_amount():
    docs = [GoldDocument("T", "t", "TOP", points(tam_osszeg=1), points(), "<!-- page 1 -->\nsemmi\n")]
    assert acceptance.compare(docs)["factors"]["tam_osszeg"]["new"]["agree"] == 1


def _real_gold_documents():
    """The gold documents of the real data root, read-only; skips while their texts are not converted."""
    cfg = config_mod.load()
    db_path = cfg.data_root / "grantrisk.db"
    if not db_path.exists() or "validate" not in cfg.values:
        pytest.skip("no database under the data root")
    from grantrisk.corpus.acquire import load_import_config

    pins = load_import_config(cfg.resolve(cfg.values["acquire"]["import_config"]))["gold_pins"]
    conn = acceptance.open_read_only(db_path)
    try:
        c2 = acceptance.complete_c2_with(conn, [s[:16] for s in pins.values()])
        if c2 is None:
            pytest.skip("no complete C2 run holds all gold texts yet")
        validate = cfg.values["validate"]
        baseline = cfg.resolve(validate["old_baseline"])
        if not Path(baseline).exists() or not cfg.source("gold_csv").exists():
            pytest.skip("the old baseline or the gold file is not available")
        return acceptance.gold_documents(
            conn, c1_run_id=acceptance.c1_of(conn, c2), c2_run_id=c2, gold_csv=cfg.source("gold_csv"),
            gold_pins=pins, old_baseline=baseline, renames=validate.get("old_baseline_renamed"),
        )
    finally:
        conn.close()


def test_each_rule_agrees_with_gold_at_least_as_well_as_the_old_regex():
    """SPEC-L1-06 on the 42 gold documents; the agreement is in-sample (ISS-26)."""
    docs = _real_gold_documents()
    report = acceptance.compare(docs)
    assert report["documents_with_text"] == 42
    worse = {f: (v["new"]["agree"], v["old"]["agree"]) for f, v in report["factors"].items() if not v["accepted"]}
    assert not worse, acceptance.format_table(report)
