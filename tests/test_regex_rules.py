"""The regex rules on short text snippets, one or more per rule and special case (SPEC-L1-05)."""

import pytest

from grantrisk.extraction.regex import rules
from grantrisk.extraction.regex.text import Document


def doc(*pages):
    """A C2 Markdown text with one page marker per page."""
    return Document("".join(f"<!-- page {i} -->\n{text}\n\n" for i, text in enumerate(pages, start=1)))


def value(rule, *pages):
    finding = rule(doc(*pages))
    return finding.value if finding.status == "found" else finding.status


SUMMARY = "|{q}|{a}|\n|---|---|\n|Hol valósítható meg a projekt?|Magyarország|"


def summary(question, answer):
    return SUMMARY.format(q=question, a=answer)


# --- fin_form ----------------------------------------------------------------------------


def test_fin_form_from_the_summary_answer():
    q = "A támogatás visszatérítendő vagy vissza nem térítendő? (Részletes információk a felhívás 3. fejezetében)"
    assert value(rules.fin_form, summary(q, "A támogatás vissza nem térítendő támogatásnak minősül.")) == "grant"


def test_fin_form_conditional_wins_over_grant():
    text = (
        "### 5.1 A támogatás formája\nA támogatás vissza nem térítendő támogatásnak minősül.\n\n"
        "A támogatás feltételesen vissza nem térítendő támogatásnak minősül."
    )
    assert value(rules.fin_form, text) == "conditional_grant"


def test_fin_form_loan_label_decides_first():
    text = "## Kölcsön típusa\nÉven túli lejáratú Kölcsön.\n\nkivéve a vissza nem térítendő támogatás előfinanszírozására"
    assert value(rules.fin_form, text) == "loan"


def test_fin_form_table_field():
    assert value(rules.fin_form, "|**Támogatás formája**|vissza nem térítendő|") == "grant"


def test_fin_form_needs_an_anchor():
    # ISS-14: the phrase alone, outside a financing-form anchor, decides nothing.
    assert value(rules.fin_form, "nem használható fel vissza nem térítendő támogatás előfinanszírozására") == "not_found"


def test_fin_form_conflict_is_ambiguous():
    text = "### A támogatás formája\nA támogatás visszatérítendő.\n\n### A támogatás jellege\nvissza nem térítendő"
    assert value(rules.fin_form, text) == "ambiguous"


# --- tam_osszeg --------------------------------------------------------------------------


def test_amount_upper_bound_of_a_range():
    q = "Mennyi támogatást lehet igényelni? (Részletes információk a felhívás 3. fejezetében)"
    assert value(rules.tam_osszeg, summary(q, "minimum 2 500 000 Ft – maximum 6 000 000 Ft")) == 6_000_000


def test_amount_with_milliard_and_decimal_comma():
    q = "Mennyi támogatást lehet igényelni?"
    assert value(rules.tam_osszeg, summary(q, "minimum 1,5 milliárd Ft – maximum 40,46 milliárd Ft")) == 40_460_000_000


def test_amount_old_generation_sentence_with_mrd():
    text = "Az igényelhető vissza nem térítendő támogatás összege 29,65 Mrd forint."
    assert value(rules.tam_osszeg, text) == 29_650_000_000


def test_amount_dotted_thousands():
    text = "a) Az igényelhető vissza nem térítendő támogatás összege: minimum 446.250.000 Ft, maximum 7.546.875.000 Ft"
    assert value(rules.tam_osszeg, text) == 7_546_875_000


def test_amount_is_never_the_call_budget():
    # ISS-13: the call's total budget (keretösszeg) is not the per-project maximum.
    text = (
        "A Felhívás meghirdetésekor a támogatásra rendelkezésre álló tervezett keretösszeg 9 600 000 000 Ft.\n\n"
        "Az igényelhető támogatás összege, a rendelkezésre álló keretösszeg 9 600 000 000 Ft terhére, legfeljebb 50 000 000 Ft."
    )
    assert value(rules.tam_osszeg, text) == 50_000_000


def test_amount_skips_a_price_per_unit():
    q = "Mennyi támogatást lehet igényelni?"
    answer = "lakásonként legfeljebb 150 000 000 Ft, projektenként legfeljebb 90 000 000 Ft."
    assert value(rules.tam_osszeg, summary(q, answer)) == 90_000_000


def test_amount_anchor_with_an_inserted_clause():
    text = (
        "Az igényelhető vissza nem térítendő támogatás - konzorcium esetében is - összege: "
        "minimum 5.000.000 forint, maximum 25.000.000 forint."
    )
    assert value(rules.tam_osszeg, text) == 25_000_000


def test_amount_glued_to_the_next_word():
    text = "Az igényelhető vissza nem térítendő támogatás összege minimum 500 000 000 Ftmaximum 105 000 000 000 Ft."
    assert value(rules.tam_osszeg, text) == 105_000_000_000


def test_amount_of_the_advance_is_not_the_grant():
    text = "Az igényelhető támogatási előleg mértéke legfeljebb 100%, maximum 175 670 000 000 Ft."
    assert value(rules.tam_osszeg, text) == "not_found"


def test_unit_price_per_hectare_leads_to_the_unit_cost_average():
    text = (
        "|Várhatóan hány projekt kap támogatást?|100 db|\n\n"
        "A Felhívás meghirdetésekor a támogatásra rendelkezésre álló tervezett keretösszeg 1 000 000 000 Ft.\n\n"
        "Az igényelhető vissza nem térítendő, egységköltség formában meghatározott támogatás összege: 70.200 Ft/hektár/5 év."
    )
    assert value(rules.tam_osszeg, text) == 10_000_000


def test_amount_of_a_loan_programme():
    text = "## 4. Kölcsön összege\nminimum 20 millió Forint – maximum 200 millió Forint"
    assert value(rules.tam_osszeg, text) == 200_000_000


def test_amount_left_to_an_annex_is_not_found():
    text = "Az igényelhető vissza nem térítendő támogatás minimum és maximum összege a területspecifikus mellékletben olvasható."
    assert value(rules.tam_osszeg, text) == "not_found"


def test_amount_ignores_the_table_of_contents():
    text = "|3.3. Mennyi támogatást lehet igényelni? ........................ 23|\n\nA keretösszeg 5 000 000 000 Ft."
    assert value(rules.tam_osszeg, text) == "not_found"


def test_unit_cost_amount_is_budget_per_expected_application():
    # Scoring System §1.1.1: budget / expected number of supported applications.
    text = (
        "|Várhatóan hány projekt kap támogatást?|40 db|\n\n"
        "A felhívás meghirdetésekor a támogatásra rendelkezésre álló tervezett keretösszeg 2 000 000 000 Ft.\n\n"
        "Az igényelhető vissza nem térítendő támogatás összege egységköltség alapján kerül meghatározásra."
    )
    finding = rules.tam_osszeg(doc(text))
    assert (finding.value, finding.warnings) == (50_000_000, ["unit_cost_average"])


def test_unit_cost_with_a_range_of_applications_is_ambiguous():
    text = (
        "|Várhatóan hány projekt kap támogatást?|minimum 10 db – maximum 40 db|\n\n"
        "A felhívás meghirdetésekor a támogatásra rendelkezésre álló tervezett keretösszeg 2 000 000 000 Ft.\n\n"
        "Az igényelhető vissza nem térítendő támogatás összege egységköltség alapján kerül meghatározásra."
    )
    assert value(rules.tam_osszeg, text) == "ambiguous"


# --- konzorcium --------------------------------------------------------------------------

CONSORTIUM_Q = "Nyújthat be támogatási kérelmet konzorcium? (Részletes információk a felhívás 1.1 fejezetében)"


def test_consortium_excluded():
    answer = "A támogatási kérelem benyújtására konzorciumi formában nincs lehetőség."
    assert value(rules.konzorcium, summary(CONSORTIUM_Q, answer)) == 0


def test_consortium_allowed():
    answer = "A támogatási kérelem benyújtására konzorciumi formában is van lehetőség."
    assert value(rules.konzorcium, summary(CONSORTIUM_Q, answer)) == 1


def test_consortium_only():
    assert value(rules.konzorcium, "A felhívásra kizárólag konzorcium nyújthat be támogatási kérelmet.") == 1


def test_consortium_negative_wins():
    text = (
        "A konzorciumi partnerként nyújthatnak be kérelmet az önkormányzatok.\n\n"
        "A támogatási kérelem benyújtására konzorciumi formában nincs lehetőség."
    )
    assert value(rules.konzorcium, text) == 0


def test_consortium_table_question_with_answer():
    assert value(rules.konzorcium, "|**Konzorcium támogatható?**|Nem|") == 0


def test_consortium_question_alone_is_not_a_statement():
    assert value(rules.konzorcium, f"|{CONSORTIUM_Q}||") == "not_found"


def test_consortium_boilerplate_is_ignored():
    text = "Ön és az esetleges konzorciumi partnereinek személyes adatainak kezelése, benyújtás esetén."
    assert value(rules.konzorcium, text) == "not_found"


# --- bead_napok --------------------------------------------------------------------------

SUBMISSION_Q = "Mikor lehet benyújtani a támogatási kérelmet? (Részletes információk a felhívás 1.3 fejezetében)"


def test_submission_days_numeric_dates():
    assert value(rules.bead_napok, summary(SUBMISSION_Q, "2023.10.20. – 2023.11.10.")) == 21


def test_submission_days_word_dates_with_times():
    text = (
        "## 1.3. Mikor lehet benyújtani a támogatási kérelmet?\n"
        "A támogatási kérelmet 2025. október 31. 9:00-tól 2025. december 1. 12:00-ig nyújthatja be."
    )
    assert value(rules.bead_napok, text) == 31


def test_submission_days_shortest_stage():
    answer = (
        "Első szakasz: 2026. január 13. – 2026. február 13. Második szakasz: 2026. március 9. – 2026. április 10. "
        "Harmadik szakasz: 2026. április 14. – 2026. április 24."
    )
    assert value(rules.bead_napok, summary(SUBMISSION_Q, answer)) == 10


def test_submission_days_old_generation_wording():
    text = (
        "### 4.3. A támogatási kérelem benyújtásának határideje és módja\n"
        "Jelen felhívás keretében a támogatási kérelmek benyújtására **2017.** év **február** hó **27.** naptól "
        "**2019.** év **február** hó **28.** napig van lehetőség."
    )
    assert value(rules.bead_napok, text) == 731


def test_submission_stage_end_and_next_start_are_not_a_period():
    text = (
        "## 1.3. Mikor lehet benyújtani a támogatási kérelmet?\n"
        "- 2025. augusztus 10. 00 óra 00 perctől 2025. augusztus 31. 23 óra 59 percig\n"
        "- 2025. szeptember 1. 00 óra 00 perctől 2025. szeptember 30. 23 óra 59 percig"
    )
    assert value(rules.bead_napok, text) == 21


def test_submission_until_funds_run_out():
    text = (
        "## 1.3. Mikor lehet benyújtani a támogatási kérelmet?\n"
        "A támogatási kérelmek 2024.07.26-tól a rendelkezésre álló keret kimerüléséig nyújthatók be."
    )
    assert value(rules.bead_napok, text) == "keret_kimerulesig"


def test_submission_closing_date_beats_until_funds_run_out():
    text = (
        "## 1.3. Mikor lehet benyújtani a támogatási kérelmet?\n"
        "A kérelmek 2024.07.26-tól 2024.08.26-ig, legfeljebb a keret kimerüléséig nyújthatók be."
    )
    assert value(rules.bead_napok, text) == 31


def test_submission_deadline_labels_in_a_table():
    text = "|**Beadás kezdete**|2022.január 20.|\n|**Beadási határidő**|2022. február 20.|"
    assert value(rules.bead_napok, text) == 31


def test_submission_left_to_an_annex_is_not_found():
    text = (
        "### 4.3. A támogatási kérelem benyújtásának határideje és módja\n"
        "A benyújtási határidő a Felhívás területspecifikus mellékletében található."
    )
    assert value(rules.bead_napok, text) == "not_found"


# --- max_tam_int -------------------------------------------------------------------------


def test_intensity_from_the_sentence():
    text = "Általános csekély összegű támogatás esetén a támogatás maximális mértéke az elszámolható költségek **90%** -a."
    assert value(rules.max_tam_int, text) == 90


def test_intensity_from_a_table():
    text = "|Régió|Támogatási intenzitás|\n|---|---|\n|Észak-Alföld|50%|\n|Dél-Alföld|70%|"
    assert value(rules.max_tam_int, text) == 70


def test_intensity_from_the_own_contribution():
    finding = rules.max_tam_int(doc("|Kell-e önerő a projekthez?|Igen, 10% mértékben|"))
    assert (finding.value, finding.warnings) == (90, ["from_own_contribution"])


def test_advance_percentage_is_not_an_intensity():
    assert value(rules.max_tam_int, "Az előleg mértéke a támogatás összegének legfeljebb 25%-a.") == "not_found"


# --- eloleg ------------------------------------------------------------------------------

ADVANCE_Q = "Mennyi előleg igényelhető? (Részletes információk a felhívás 7.1. fejezetében)"


def test_advance_largest_percentage():
    answer = "Önkormányzatok esetében 100 %-os előleg igényelhető, civil és egyházi szervezetek esetében 25%."
    assert value(rules.eloleg, summary(ADVANCE_Q, answer)) == 100


def test_advance_of_a_loan_is_100():
    finding = rules.eloleg(doc("## Kölcsön típusa\nÉven túli lejáratú Kölcsön."))
    assert (finding.value, finding.warnings) == (100, ["loan"])


def test_advance_of_a_loan_without_the_loan_label_is_100():
    """DEC-40: whenever fin_form is a loan, not only with the loan programme's field label."""
    text = "### A támogatás formája\nA támogatás visszatérítendő támogatásnak minősül. Az előleg mértéke 50%."
    finding = rules.eloleg(doc(text))
    assert (finding.value, finding.warnings) == (100, ["loan"])


@pytest.mark.parametrize("answer", ["nem releváns", "Jelen felhívás keretében előleg igénylésére nincs mód."])
def test_advance_not_available_is_0(answer):
    assert value(rules.eloleg, summary(ADVANCE_Q, answer)) == 0


def test_advance_by_decree_needs_a_judgement():
    text = (
        "### 5.4 Előleg igénylése\nAz igényelhető előleg mértékét a 272/2014. (XI.5.) Korm. rendelet 116. § (2) "
        "bekezdése szerint kell meghatározni."
    )
    assert value(rules.eloleg, text) == "ambiguous"


def test_advance_percentage_before_the_word():
    text = "A Kormány a projektnek a megítélt támogatás legfeljebb 50%-ának megfelelő összegű támogatási előleget biztosít."
    assert value(rules.eloleg, text) == 50


def test_advance_ignores_other_percentages():
    assert value(rules.eloleg, "A támogatás maximális mértéke 100%.") == "not_found"


# --- idotartam ---------------------------------------------------------------------------


def test_duration_largest_number_of_months():
    answer = (
        "A projekt fizikai befejezésére a megkezdésétől számított legfeljebb 12 hónap áll rendelkezésre, "
        "felhőszolgáltatás beszerzése esetén legfeljebb 18 hónap."
    )
    assert value(rules.idotartam, summary("Mennyi a projekt végrehajtására rendelkezésre álló időtartam?", answer)) == 18


DURATION_Q = "Mennyi a projekt végrehajtására rendelkezésre álló időtartam?"
DEADLINE = "A projekt fizikai befejezésének határideje legkésőbb 2027.12.31."


def test_duration_from_a_deadline_date_without_a_submission_date_is_not_found():
    finding = rules.idotartam(doc(summary(DURATION_Q, DEADLINE)))
    assert (finding.status, finding.warnings) == ("not_found", ["deadline_without_submission_date"])


def test_duration_from_the_last_submission_date_to_the_deadline():
    """DEC-37 (b): the shortest time a project can have, from the closing date of the last stage."""
    submission = (
        "## 1.3. Mikor lehet benyújtani a támogatási kérelmet?\n"
        "Első szakasz: 2023. szeptember 1. – 2023. október 2. Második szakasz: 2023. október 20. – 2023. november 10."
    )
    finding = rules.idotartam(doc(submission, summary(DURATION_Q, DEADLINE)))
    assert (finding.status, finding.value) == ("found", 49.7)  # 2023-11-10 → 2027-12-31
    assert finding.warnings == ["months_from_dates: 2023-11-10 → 2027-12-31"]


def test_duration_of_whole_months_from_dates():
    submission = "## Mikor lehet benyújtani a támogatási kérelmet?\nA kérelmek 2027.10.01. – 2027.12.31. között nyújthatók be."
    text = summary(DURATION_Q, "A projekt fizikai befejezésének határideje: 2029. december 31.")
    assert value(rules.idotartam, submission, text) == 24  # exactly 24 months: 2 points (DEC-13)


def test_a_duration_that_is_not_defined_is_the_longest():
    text = ("## A projekt végrehajtására rendelkezésre álló időtartam\nA projekt fizikai befejezésére rendelkezésre "
            "álló időtartam meghatározása jelen Felhívás esetében nem értelmezett.")
    finding = rules.idotartam(doc(text))
    assert (finding.value, finding.warnings) == ("maximalis", ["duration_not_defined"])


def test_submission_open_until_the_deadline_is_the_longest_duration():
    submission = "## Mikor lehet benyújtani a támogatási kérelmet?\n2023.03.01. – 2029.12.31."
    answer = "A projekt fizikai befejezésének határideje reális véghatáridő, de legkésőbb 2029.12.31."
    finding = rules.idotartam(doc(submission, summary(DURATION_Q, answer)))
    assert (finding.value, finding.warnings) == ("maximalis", ["submission_open_until_deadline: 2029-12-31"])


def test_a_single_closing_date_is_a_submission_date():
    submission = ("### 4.3 A támogatási kérelem benyújtásának határideje és módja\n"
                  "A támogatási kérelmek benyújtása a Felhívás megjelenésétől 2016. május 31-ig lehetséges.")
    answer = "A projekt fizikai befejezésére a projekt megkezdését követően legfeljebb 2023. december 31-ig van lehetőség."
    assert value(rules.idotartam, submission, summary(DURATION_Q, answer)) == 91  # 2016-05-31 → 2023-12-31


def test_a_start_date_is_not_a_deadline():
    submission = "## Mikor lehet benyújtani a támogatási kérelmet?\nA kérelmek 2016. május 31-ig nyújthatók be."
    answer = "Az elszámolhatósági időszak kezdete nem lehet korábbi, mint 2016.01.01."
    assert value(rules.idotartam, submission, summary(DURATION_Q, answer)) == "not_found"


def test_stated_months_win_over_a_deadline_date():
    submission = "## Mikor lehet benyújtani a támogatási kérelmet?\n2023.10.20. – 2023.11.10."
    answer = "A projekt fizikai befejezésére 18 hónap áll rendelkezésre, legkésőbb 2027.12.31-ig."
    assert value(rules.idotartam, submission, summary(DURATION_Q, answer)) == 18


def test_months_between():
    from datetime import date

    assert rules.months_between(date(2027, 12, 31), date(2029, 12, 31)) == 24
    assert rules.months_between(date(2017, 4, 21), date(2019, 6, 30)) == 26.3
    assert rules.months_between(date(2024, 1, 31), date(2024, 3, 1)) == 1


def test_duration_ignores_an_extension():
    text = (
        "A projekt fizikai befejezésére legfeljebb 24 hónap áll rendelkezésre, "
        "amely további maximum 36 hónappal meghosszabbítható."
    )
    assert value(rules.idotartam, text) == 24


# --- tam_tevekenyseg ---------------------------------------------------------------------


def test_activities_infrastructure():
    text = "## 2.1.1. Önállóan támogatható tevékenységek\n- a) Az épület felújítása és/vagy bővítése."
    assert value(rules.tam_tevekenyseg, text) == ["infrastruktura_ingatlan"]


def test_activities_research_and_infrastructure():
    text = (
        "|Milyen tevékenységek támogathatóak?|• K+F tevékenység megvalósítása; • laboratórium építése|\n"
        "|Mikor lehet benyújtani a támogatási kérelmet?|2026. január 6. - 2026. február 9.|"
    )
    assert value(rules.tam_tevekenyseg, text) == ["kutatas_fejlesztes", "infrastruktura_ingatlan"]


def test_activities_without_a_pattern_are_other():
    # ISS-12: the old rule could never return egyeb.
    text = "## 2.1.1. Önállóan támogatható tevékenységek\n- a) Informatikai eszközök beszerzése\n- b) Képzés"
    finding = rules.tam_tevekenyseg(doc(text))
    assert (finding.value, finding.warnings) == (["egyeb"], ["no_research_or_infrastructure_pattern"])


def test_activities_not_financed_are_left_out():
    text = (
        "## 2.1.1. Önállóan támogatható tevékenységek\n- a) Informatikai eszközök beszerzése\n"
        "### 2.1.3. Nem finanszírozott tevékenységek\n- Épület építése"
    )
    assert value(rules.tam_tevekenyseg, text) == ["egyeb"]


def test_activities_section_ends_at_the_next_section():
    text = (
        "### 3.1.1. Önállóan támogatható tevékenységek\n- Eszközbeszerzés\n"
        "### 3.2. A projekt műszaki-szakmai tartalmával kapcsolatos elvárások\n- Az épület felújítása"
    )
    assert value(rules.tam_tevekenyseg, text) == ["egyeb"]


def test_activities_without_the_section_are_not_found():
    assert value(rules.tam_tevekenyseg, "Az épület felújítása a projekt része.") == "not_found"


# --- egysz_elszam ------------------------------------------------------------------------


def test_simplified_costs_allowed():
    assert value(rules.egysz_elszam, "- b) Százalékos átalányalapú finanszírozás költségei") == 1


def test_simplified_costs_excluded_wins_over_other_mentions():
    # ISS-19: every occurrence counts, not only the first.
    text = (
        "A piaci ár igazolása egységköltséggel is történhet.\n\n"
        "**Egyszerűsített elszámolásra jelen felhívás keretében nincs lehetőség.**"
    )
    assert value(rules.egysz_elszam, text) == 0


def test_simplified_costs_heading_with_not_relevant():
    text = "### 5.5.1 Egyszerűsített elszámolás alkalmazása\n\nJelen felhívásban nem releváns."
    assert value(rules.egysz_elszam, text) == 0


def test_simplified_costs_not_mentioned():
    assert value(rules.egysz_elszam, "A költségeket számlákkal kell igazolni.") == "not_found"


# --- biztositek --------------------------------------------------------------------------


def test_collateral_obligation():
    text = "A kedvezményezettnek kizárólag előleg igénylése esetén kell az előleg összegével megegyező összegű biztosítékot nyújtani."
    assert value(rules.biztositek, text) == 1


def test_collateral_reference_to_the_general_guide_is_an_exemption():
    text = (
        "## 6.5. Hol találhatóak a biztosítéknyújtásra vonatkozó elvárások?\n"
        "A biztosítéknyújtással kapcsolatos részletes szabályozást az ”ÁÚF 21-27” 6. fejezete tartalmazza."
    )
    finding = rules.biztositek(doc(text))
    assert (finding.value, finding.span is not None) == (0, True)


def test_collateral_conditional_on_exemption_is_not_an_obligation():
    text = (
        "A kapott támogatás ellenében – amennyiben Ön nem mentes a biztosítéknyújtási kötelezettség alól – "
        "biztosítéknyújtási kötelezettség terheli. A biztosítékot az első kifizetési kérelemmel egyidejűleg kell nyújtani."
    )
    assert value(rules.biztositek, text) == 0


def test_collateral_cost_clause_is_not_an_obligation():
    text = "- Biztosítékok költségei - ha a támogatás folyósításához a biztosítéknyújtás kötelező (pl. bankgarancia költsége)."
    assert value(rules.biztositek, text) == 0


def test_collateral_never_none():
    finding = rules.biztositek(doc("Ez a felhívás nem említi a kérdést."))
    assert (finding.value, finding.status, finding.warnings) == (0, "found", ["default_value"])


def test_collateral_not_relevant():
    assert value(rules.biztositek, "### 3.9. Biztosítékok köre\nJelen felhívás keretében nem releváns.") == 0
