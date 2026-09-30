"""The regex rules: one function per factor (SPEC-L1-05, DEC-28).

Every rule starts from an anchor phrase and looks for its value only in the answer
to that anchor: the rest of a summary-table row, the section under a heading, or the
sentence. No rule searches the whole text for a value (ISS-14). The special cases of
the factor definitions (Spec_ScoringSystem.md §1.1.1) are part of the rule of their
factor, each tied to its own anchor; there is no override layer. The TOP rule is not
applied here; L3 applies it (DEC-31).

Each rule returns a ``Finding``: the value (or None), the status, and the span of the
Markdown that supports it, from which the extractor cuts the evidence quote.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from grantrisk.extraction.regex.text import (
    MONEY,
    PERCENT,
    Document,
    dates,
    money,
    pattern,
    percent,
)
from grantrisk.labelling.scoring import LONGEST_DURATION, UNTIL_FUNDS_RUN_OUT

# Change RULES_VERSION whenever a rule changes; every run records it (Spec_L1 §2.1).
RULES_VERSION = "2"  # 2: idotartam from a deadline date and its longest duration (DEC-37); eloleg of any loan (DEC-40)


@dataclass
class Finding:
    value: Any = None
    status: str = "not_found"  # found, not_found or ambiguous
    span: tuple[int, int] | None = None  # the Markdown the evidence is cut from
    warnings: list[str] = field(default_factory=list)


def found(value: Any, span: tuple[int, int], *warnings: str) -> Finding:
    return Finding(value, "found", span, list(warnings))


def ambiguous(span: tuple[int, int] | None, reason: str) -> Finding:
    return Finding(None, "ambiguous", span, [reason])


def _span(anchor_start: int, start: int, end: int, reach: int = 400) -> tuple[int, int]:
    """The evidence span: from the anchor to the value, or the value alone if they are far apart."""
    return (anchor_start, end) if end - anchor_start <= reach else (start, end)


# --- fin_form ----------------------------------------------------------------------------

LOAN_LABEL = pattern(r"\b(?:kölcsön|hitel) típusa\b")
FIN_FORM_ANCHORS = pattern(
    r"támogatás visszatérítendő vagy vissza nem térítendő\?|vissza kell-e fizetni a támogatást\?"
    r"|\btámogatás (?:formája|jellege)\b|\bfinanszírozási forma\b"
)
FIN_FORM_STATEMENT = pattern(r"(?:feltételesen )?(?:vissza nem térítendő|visszatérítendő)[^.|\n]{0,40}?minősül")
CONDITIONAL = pattern(r"feltételesen vissza nem térítendő")
GRANT = pattern(r"vissza nem térítendő")
LOAN = pattern(r"\bvisszatérítendő|\bkölcsön\b|\bhitel\b")


def _fin_form_of(doc: Document, start: int, end: int) -> tuple[str, int] | None:
    """The financing form stated in ``start``–``end``, and where its phrase ends."""
    for form, rx in (("conditional_grant", CONDITIONAL), ("grant", GRANT), ("loan", LOAN)):
        m = rx.search(doc.view, start, end)
        if m:
            return form, m.end()
    return None


def _next_line_end(doc: Document, pos: int) -> int:
    """The end of the first non-empty line after the line of ``pos`` (a field label's value)."""
    end = doc.line_end(pos)
    m = re.compile(r"\S").search(doc.view, end)
    return doc.line_end(m.start()) if m else end


def fin_form(doc: Document) -> Finding:
    """Financing form. A loan programme's field label decides first; then „feltételesen vissza nem
    térítendő" near an anchor wins over „vissza nem térítendő" (SPEC-L1-05)."""
    for m in doc.anchors(LOAN_LABEL):
        return found("loan", (m.start(), _next_line_end(doc, m.end())))
    by_form: dict[str, tuple[int, int]] = {}
    for m in doc.anchors(FIN_FORM_ANCHORS):
        _, end = doc.segment(m, 200)
        stated = _fin_form_of(doc, m.end(), end)
        if stated:
            by_form.setdefault(stated[0], (m.start(), stated[1]))
    for m in doc.anchors(FIN_FORM_STATEMENT):
        form, _ = _fin_form_of(doc, m.start(), m.end())
        by_form.setdefault(form, (m.start(), m.end()))
    if "conditional_grant" in by_form:
        return found("conditional_grant", by_form["conditional_grant"])
    if len(by_form) > 1:
        return ambiguous(next(iter(by_form.values())), "conflicting_values: " + ", ".join(sorted(by_form)))
    if by_form:
        form, span = next(iter(by_form.items()))
        return found(form, span)
    return Finding()


# --- tam_osszeg --------------------------------------------------------------------------

AMOUNT_ANCHORS = pattern(
    r"mennyi támogatást lehet igényelni\?"
    # "Az igényelhető vissza nem térítendő támogatás – konzorcium esetében is – összege", never the advance's
    r"|igényelhető[, ]+(?:(?!előleg)[^.|\n]){0,60}?támogatás\w*(?: (?:(?!előleg)[^.|\n]){0,30}?)? (?:összege|mértéke)"
    r"|\btámogatás (?:maximum|maximális) összege|\bmaximális támogatási összeg"
    r"|projekte(?:ke)?t (?:maximum|legfeljebb)"
    r"|^[\s\d.]*(?:<mark>)?\s*(?:kölcsön|hitel) összege",  # a loan programme's field label
    re.IGNORECASE | re.MULTILINE,
)
BUDGET = pattern(r"(?:keretösszeg|forráskeret|forrás keretösszege|rendelkezésre álló forrás)")
BUDGET_REACH = 120
UNIT_COST = pattern(r"egységköltség")
# "csoportszobánként legfeljebb 33 009 019 Ft" is a price per unit, not the maximum per project
# ("projektenként" and "kérelmenként" are per project).
PER_UNIT = pattern(r"\b(?!projekt|kérelm|pályáz)\w+ként(?: (?:legalább|legfeljebb|maximum|minimum|max\.|min\.))?\s*$")
PER_UNIT_AFTER = re.compile(r"\s*/\s*\w")
CALL_BUDGET = pattern(r"támogatásra rendelkezésre álló (?:tervezett )?keretösszeg|\bforrás keretösszege")
EXPECTED_COUNT = pattern(
    r"várhatóan hány projekt kap támogatást\?|támogatható projektek száma"
    r"|támogatott (?:támogatási )?(?:kérelmek|projektek) (?:várható )?(?:darab)?száma"
)
COUNT = re.compile(r"(\d{1,3}(?: \d{3})*|\d+)\s*(?:db\b|darab)", re.IGNORECASE)


def _amounts(doc: Document, start: int, end: int) -> list[tuple[int, int, int]]:
    """(amount, start, end) of the amounts in ``start``–``end`` that are not the call's budget
    (ISS-13) and not a price per unit."""
    result = []
    previous = start
    for m in MONEY.finditer(doc.view, start, end):
        # The words that lead to this amount: since the previous amount, within its sentence.
        lead = doc.view[max(previous, doc.sentence(m.start())[0], m.start() - BUDGET_REACH) : m.start()]
        previous = m.end()
        if BUDGET.search(lead) or PER_UNIT.search(doc.view[max(start, m.start() - 30) : m.start()]):
            continue
        if PER_UNIT_AFTER.match(doc.view, m.end()):  # "70.200 Ft/hektár", "800.000 Ft/hó"
            continue
        value = money(m)
        if value is not None:
            result.append((value, m.start(), m.end()))
    return result


def _unit_cost_amount(doc: Document) -> Finding:
    """Scoring System §1.1.1: the call's budget divided by the expected number of supported applications."""
    budget = None
    for m in doc.anchors(CALL_BUDGET):
        mm = MONEY.search(doc.view, m.end(), min(len(doc.view), m.end() + 150))
        if mm and money(mm):
            budget = (money(mm), m.start(), mm.end())
            break
    count = None
    for m in doc.anchors(EXPECTED_COUNT):
        _, end = doc.segment(m, 150)
        counts = [(int(c.group(1).replace(" ", "")), c) for c in COUNT.finditer(doc.view, m.end(), end)]
        if len(counts) > 1:
            return ambiguous((m.start(), end), "expected_number_is_a_range")
        if counts:
            count = (counts[0][0], m.start(), counts[0][1].end())
            break
    if budget is None or count is None or count[0] == 0:
        return Finding(warnings=["unit_cost_without_budget_or_number"])
    return found(round(budget[0] / count[0]), (budget[1], budget[2]), "unit_cost_average")


def tam_osszeg(doc: Document) -> Finding:
    """The per-project maximum grant: the largest amount in the answers to the amount anchors, never
    the call's budget (ISS-13); for unit-cost calls, the budget divided by the expected number of
    supported applications (Scoring System §1.1.1)."""
    best = None
    unit_cost = False
    for m in doc.anchors(AMOUNT_ANCHORS):
        _, end = doc.segment(m, 300)
        amounts = _amounts(doc, m.end(), end)
        first = amounts[0][1] if amounts else end
        if UNIT_COST.search(doc.view, m.start(), first):
            unit_cost = True
        for value, s, e in amounts:
            if best is None or value > best[0]:
                best = (value, _span(m.start(), s, e))
    if best is not None:
        return found(best[0], best[1])
    if unit_cost:
        return _unit_cost_amount(doc)
    return Finding()


# --- konzorcium --------------------------------------------------------------------------

CONSORTIUM = pattern(r"konzorci\w*")
CONSORTIUM_QUESTION = pattern(r"konzorcium támogatható\?\s*(igen|nem)\b")
CONSORTIUM_NEGATIVE = pattern(
    r"nincs lehetőség|nem lehetséges|nem nyújthat\w*|nem igényelhet\w*|nem pályázhat\w*|kizárólag egyedi"
    r"|nem támogatható|nem jogosult"
)
CONSORTIUM_POSITIVE = pattern(
    r"(?:is )?van lehetőség|lehetőség van|lehetséges|kizárólag konzorci\w*|konzorciumban|konzorciumot alkot\w*"
    r"|konzorcium\w* (?:is )?(?:nyújthat\w*|adhat\w*|jogosult\w*)|konzorciumi formában is|konzorciumi partnerként"
)
CONSORTIUM_SUBMISSION = pattern(r"benyújt\w*|nyújthat\w*|adhat\w* be|igényelhet\w*|pályáz\w*")
CONSORTIUM_POSSIBILITY = pattern(r"támogatható|lehetőség|lehetséges")
CONSORTIUM_MEMBER = pattern(r"konzorciumvezető\w*|konzorcium vezető\w*|konzorciumi tag\w*|konzorcium tagja")
BOILERPLATE = pattern(r"esetleges konzorci|konzorciumi partnereinek, tulajdonosainak")


def konzorcium(doc: Document) -> Finding:
    """Consortium allowed. Statements about submitting as a consortium, sentence by sentence; a
    negative statement wins over a positive one (SPEC-L1-05). Questions are not statements."""
    for m in doc.anchors(CONSORTIUM_QUESTION):
        return found(1 if m.group(1).lower() == "igen" else 0, (m.start(), m.end()))
    negative = positive = weak_positive = member = None
    seen = set()
    for m in doc.anchors(CONSORTIUM):
        lo, hi = doc.sentence(m.start())
        if (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        text = doc.view[lo:hi]
        if text.rstrip().endswith("?") or BOILERPLATE.search(text):
            continue
        submission = CONSORTIUM_SUBMISSION.search(text)
        if not submission and not CONSORTIUM_POSSIBILITY.search(text):
            if member is None and CONSORTIUM_MEMBER.search(text):
                member = (lo, hi)
            continue
        if negative is None and CONSORTIUM_NEGATIVE.search(text):
            negative = (lo, hi)
        elif CONSORTIUM_POSITIVE.search(text):
            if submission and positive is None:
                positive = (lo, hi)
            elif weak_positive is None:
                weak_positive = (lo, hi)
    if negative:
        return found(0, negative)
    if positive or weak_positive:
        return found(1, positive or weak_positive)
    if member:
        return found(1, member, "consortium_member_mentioned")
    return Finding()


# --- bead_napok --------------------------------------------------------------------------

SUBMISSION_ANCHORS = pattern(
    r"mikor lehet benyújtani a támogatási kérelmet\?|mikor nyújtható be"
    r"|kérelm\w* benyújtás\w* (?:határideje|időszaka|kezdete)|benyújtási (?:időszak|határidő|szakasz)\w*"
    r"|beadási (?:időszak|határidő)\w*|beadás kezdete|értékelési (?:szakasz|határnap)\w*"
    r"|kérelm\w* benyújtás(?:ára|a)\b|lehet (?:kölcsön|hitel|támogatási )?kérelmet benyújtani"
)
PAIR_CONNECTOR = re.compile(r"[-–—]|t[óő]l\b|határid", re.IGNORECASE)
PERIOD_END = re.compile(r"ig\b", re.IGNORECASE)  # "-ig", "napig", "percig": the first date closes a period
MAX_CONNECTOR = 80
UNTIL_FUNDS = pattern(r"(?:keret\w*|forrás\w*) kimerüléséig")


def _periods(doc: Document, start: int, end: int) -> list[tuple[int, int, int]]:
    """(days, start, end) of the submission periods: consecutive date pairs joined by a dash, '-tól'
    or a deadline label."""
    found_dates = dates(doc.view[start:end], start)
    periods = []
    i = 0
    while i < len(found_dates) - 1:
        (d1, s1, e1), (d2, s2, e2) = found_dates[i], found_dates[i + 1]
        between = doc.view[e1:s2]
        if (
            len(between) <= MAX_CONNECTOR and PAIR_CONNECTOR.search(between)
            and not PERIOD_END.search(between) and d2 > d1
        ):
            periods.append(((d2 - d1).days, s1, e2))
            i += 2
        else:
            i += 1
    return periods


def bead_napok(doc: Document) -> Finding:
    """Days available for submission: the shortest period between the dates near the submission anchors,
    evaluation stages included; `keret_kimerulesig` if submission is open until the funds run out and no
    closing date is given (SPEC-L1-05)."""
    best = None
    until_funds = None
    for m in doc.anchors(SUBMISSION_ANCHORS):
        lo, hi = doc.segment(m, 1200, before=150)
        for days, s, e in _periods(doc, lo, hi):
            if best is None or days < best[0]:
                best = (days, _span(m.start(), s, e) if s >= m.start() else (s, e))
        if until_funds is None:
            u = UNTIL_FUNDS.search(doc.view, lo, hi)
            if u:
                until_funds = (min(m.start(), u.start()), max(m.end(), u.end()))
    if best is not None:
        return found(best[0], best[1])
    if until_funds is not None:
        return found(UNTIL_FUNDS_RUN_OUT, until_funds)
    return Finding()


# --- max_tam_int -------------------------------------------------------------------------

INTENSITY_ANCHORS = pattern(
    r"(?<!előleg\s)\btámogatás\w* (?:maximális |maximum )?(?:intenzitás\w*|mértéke)"
)
INTENSITY_TABLE = pattern(r"intenzitás|támogatás\w* mértéke")
OWN_CONTRIBUTION = pattern(r"kell-e önerő a projekthez\?|\bönerő\w*")
OWN_SHARE = re.compile(r"(?:minimum|legalább|igen,)?\s*(\d{1,3}(?:[.,]\d{1,2})?)\s*%", re.IGNORECASE)


def _percents(doc: Document, start: int, end: int) -> list[tuple[float, int, int]]:
    return [(percent(p), p.start(), p.end()) for p in PERCENT.finditer(doc.view, start, end) if percent(p) <= 100]


def max_tam_int(doc: Document) -> Finding:
    """Maximum aid intensity: the largest percentage near the intensity anchors, or in a table about
    intensity; otherwise 100 minus the stated own contribution (the old rule's fallback)."""
    best = None
    for m in doc.anchors(INTENSITY_ANCHORS):
        _, end = doc.segment(m, 250)
        for value, s, e in _percents(doc, m.end(), end):
            if best is None or value > best[0]:
                best = (value, _span(m.start(), s, e))
    for t_start, t_end in doc.tables():
        header_end = doc.line_end(t_start)
        if not INTENSITY_TABLE.search(doc.view, t_start, header_end):
            continue
        for value, s, e in _percents(doc, t_start, t_end):
            if best is None or value > best[0]:
                best = (value, (s, e))
    if best is not None:
        return found(best[0], best[1])
    for m in doc.anchors(OWN_CONTRIBUTION):
        _, end = doc.segment(m, 120)
        share = OWN_SHARE.search(doc.view, m.end(), end)
        if share and float(share.group(1).replace(",", ".")) < 100:
            value = round(100 - float(share.group(1).replace(",", ".")), 2)
            return found(value, (m.start(), share.end()), "from_own_contribution")
    return Finding()


# --- eloleg ------------------------------------------------------------------------------

ADVANCE_ANCHORS = pattern(
    r"(?:mennyi|mekkora mértékű) előleg igényelhető\?|\belőleg\w* (?:maximális )?mértéke|\belőleg igénylése\b"
)
ADVANCE_BEFORE = pattern(
    r"(\d{1,3}(?:[.,]\d{1,2})?)\s*%(?:-os|-ának megfelelő[^.|\n]{0,60}?)\s+(?:összegű\s+)?(?:támogatási\s+)?előleg"
)
NO_ADVANCE = pattern(r"nem releváns|előleg igénylésére nincs mód")
ADVANCE_DECREE = pattern(r"272/2014\.?\s*\(\s*XI\.?\s*5\.?\s*\)\s*Korm\.?\s*rendelet[^.]{0,40}116\.?\s*§")


def eloleg(doc: Document) -> Finding:
    """Advance payment: the largest percentage near „előleg"; the special cases of the definition where an
    anchor decides them: a loan (fin_form) → 100, „nem releváns" or „előleg igénylésére nincs mód" → 0. A
    reference to 272/2014. Korm. rendelet 116. § needs a judgement about the applicants → ambiguous."""
    form = fin_form(doc)
    if form.value == "loan":  # the definition: a loan's advance is always 100 % (DEC-40)
        return found(100, form.span, "loan")
    best = None
    decree = None
    for m in doc.anchors(ADVANCE_ANCHORS):
        _, end = doc.segment(m, 250)
        answer = doc.view[m.end() : end]
        if m.group(0).endswith("?"):
            no = NO_ADVANCE.search(answer)
            if no and not PERCENT.search(answer[: no.start()]):
                return found(0, (m.start(), m.end() + no.end()), "no_advance")
        for value, s, e in _percents(doc, m.end(), end):
            if best is None or value > best[0]:
                best = (value, _span(m.start(), s, e))
        if decree is None:
            d = ADVANCE_DECREE.search(doc.view, m.end(), end)
            if d:
                decree = (m.start(), d.end())
    for m in doc.anchors(ADVANCE_BEFORE):
        value = percent(m)
        if value <= 100 and (best is None or value > best[0]):
            best = (value, (m.start(), m.end()))
    if best is not None:
        return found(best[0], best[1])
    if decree is not None:
        return ambiguous(decree, "advance_by_applicant_type")
    return Finding()


# --- idotartam ---------------------------------------------------------------------------

DURATION_ANCHORS = pattern(
    r"rendelkezésre álló időtartam|fizikai befejezésére|fizikai befejezésének (?:határideje|legkésőbbi időpontja|dátuma)"
    r"|projekt\w* (?:megvalósítás\w*|végrehajtás\w*) (?:időtartama|rendelkezésre álló)"
)
MONTHS = re.compile(r"(?<![\d.,])(\d{1,3})\s*hónap(?!pal)", re.IGNORECASE)


def months_between(start: date, end: date) -> float:
    """DEC-37: the calendar months from ``start`` to ``end``, each day of difference in the day of the month
    counting as 1/30 month: 2027-12-31 → 2029-12-31 is 24, 2023-11-10 → 2027-12-31 is 49.7."""
    return round(12 * (end.year - start.year) + (end.month - start.month) + (end.day - start.day) / 30, 2)


# A date that closes something: "2016. május 31-ig", "2023.11.10. napig".
CLOSES = re.compile(r"\.?\s*(?:[-–]\s*)?(?:ig|napig)\b", re.IGNORECASE)
# A date that starts something, and so is no deadline: "2016.01.01-től", "nem lehet korábbi, mint 2016.01.01.",
# "kezdete: …".
STARTS = re.compile(r"\.?\s*(?:[-–]\s*)?t[óő]l\b", re.IGNORECASE)
START_BEFORE = pattern(r"(?:korábbi|kezdet\w*)\W+(?:\w+\W+){0,4}$")
DURATION_NOT_DEFINED = pattern(r"időtartam\w*[^.|]{0,80}?nem értelmezett")
# A loan programme's last payment: "A Hitelprogram keretében utoljára 2029. szeptember 30-án lehet a
# Kölcsönszerződés alapján a Végső Kedvezményezetteknek kifizetést teljesíteni."
LOAN_LAST_PAYMENT = pattern(r"utoljára[^|]{0,160}?kifizetést teljesíteni")


def last_submission_date(doc: Document) -> date | None:
    """The closing date of the call's last submission period or evaluation stage (DEC-37): the end of a period
    found as for bead_napok (DEC-38), or a single closing date near the submission anchors."""
    last = None
    for m in doc.anchors(SUBMISSION_ANCHORS):
        lo, hi = doc.segment(m, 1200, before=150)
        ends = [max(d for d, *_ in dates(doc.view[s:e], s)) for _, s, e in _periods(doc, lo, hi)]
        ends += [d for d, s, e in dates(doc.view[lo:hi], lo) if CLOSES.match(doc.view, e)]
        for end in ends:
            last = end if last is None else max(last, end)
    return last


def _deadlines(doc: Document, start: int, end: int) -> list[tuple[date, int, int]]:
    """The dates in ``start``–``end`` that can be a deadline: every date but those that start something."""
    return [(d, s, e) for d, s, e in dates(doc.view[start:end], start)
            if not STARTS.match(doc.view, e) and not START_BEFORE.search(doc.view, max(start, s - 60), s)]


def idotartam(doc: Document) -> Finding:
    """Project duration in months: the largest number of months in the answers to the duration anchors.

    A call that gives a completion deadline as a date instead gets the months from its last submission date
    to that deadline: the shortest time a project can have. For a loan, the deadline is the last day the loan
    can be paid out. The longest duration, `maximalis`, if the call says that the duration is not defined, or
    if submission stays open until the completion deadline itself (DEC-37). Without a submission date, no value."""
    best = None
    deadline = None
    not_defined = None
    for m in doc.anchors(DURATION_ANCHORS):
        _, end = doc.segment(m, 300)
        for mm in MONTHS.finditer(doc.view, m.end(), end):
            value = int(mm.group(1))
            if value > 0 and (best is None or value > best[0]):
                best = (value, _span(m.start(), mm.start(), mm.end()))
        for d, s, e in _deadlines(doc, m.end(), end):
            if deadline is None or d > deadline[0]:
                deadline = (d, _span(m.start(), s, e))
        if not_defined is None:
            nd = DURATION_NOT_DEFINED.search(doc.view, max(0, m.start() - 200), end)
            if nd:
                not_defined = (min(m.start(), nd.start()), max(m.end(), nd.end()))
    if best is not None:
        return found(best[0], best[1])
    if not_defined is not None:
        return found(LONGEST_DURATION, not_defined, "duration_not_defined")
    loan = ""
    if fin_form(doc).value == "loan":  # a loan's deadline: the last day it can be paid out (DEC-37)
        payments = [(d, (m.start(), e)) for m in doc.anchors(LOAN_LAST_PAYMENT)
                    for d, s, e in dates(doc.view[m.start() : m.end()], m.start())]
        if payments:
            deadline, loan = max(payments), " (loan: the last payment)"
    if deadline is None:
        return Finding()
    last = last_submission_date(doc)
    if last is None:
        return Finding(warnings=["deadline_without_submission_date"])
    if last >= deadline[0]:  # a project may be submitted until its own deadline: the longest duration
        return found(LONGEST_DURATION, deadline[1], f"submission_open_until_deadline: {deadline[0].isoformat()}")
    months = months_between(last, deadline[0])
    return found(months, deadline[1], f"months_from_dates: {last.isoformat()} → {deadline[0].isoformat()}{loan}")


# --- tam_tevekenyseg ---------------------------------------------------------------------

ACTIVITY_ANCHORS = pattern(
    r"milyen tevékenységek támogathatóak\?|(?<!nem\s)önállóan támogatható tevékenységek"
    r"|támogatható tevékenységek bemutatása|^[\s|*#\d.]*támogatható tevékenységek\b",
    re.IGNORECASE | re.MULTILINE,
)
ACTIVITY_SECTION_END = pattern(
    r"nem finanszírozott tevékenységek|mire nem kapható támogatás|nem elszámolható költségek"
    r"|támogatható tevékenységek állami támogatási"
)
RESEARCH = pattern(r"kutatás[- ]?fejlesztés\w*|kutatás és fejlesztés\w*|\bK\+F\b|kísérleti fejlesztés\w*|ipari kutatás\w*")
INFRASTRUCTURE = pattern(
    r"\bépítés\w*|\bépület\w* (?:felújítás|bővítés|átalakítás|korszerűsítés|létesítés)\w*|\bfelújítás\w*"
    r"|\bátalakítás\w*|\bbővítés\w*|\bkorszerűsítés\w*|\brekonstrukció\w*|\blétesítés\w*"
    r"|\bingatlan\w*|infrastruktúrafejlesztés\w*|infrastrukturális"
)
ACTIVITY_CAP = 2000
# A heading numbered with one or two levels ("3.2. …", "4. …"): the end of the activities' section.
MAJOR_HEADING = re.compile(r"^#+[ \t*]*\d+\.(?:\d+\.?)?[ \t*]+\S", re.MULTILINE)


def tam_tevekenyseg(doc: Document) -> Finding:
    """Supported activity types: the categories found in the section of the supported activities; if the
    section is found but has no R&D or infrastructure pattern, [egyeb] (ISS-12)."""
    categories: dict[str, tuple[int, int]] = {}
    section = None
    for m in doc.anchors(ACTIVITY_ANCHORS):
        if doc.markdown[doc.line_start(m.start())] == "|":  # a summary-table row: to the next question
            _, end = doc.segment(m, ACTIVITY_CAP)
        else:  # a heading or a paragraph: to the next chapter or section, not a subsection
            end = min(len(doc.markdown), m.end() + ACTIVITY_CAP)
            nxt = MAJOR_HEADING.search(doc.markdown, doc.line_end(m.end()), end)
            if nxt:
                end = nxt.start()
        stop = ACTIVITY_SECTION_END.search(doc.view, m.end(), end)
        if stop:
            end = stop.start()
        if section is None:
            section = (m.start(), m.end())
        for name, rx in (("kutatas_fejlesztes", RESEARCH), ("infrastruktura_ingatlan", INFRASTRUCTURE)):
            hit = rx.search(doc.view, m.end(), end)
            if hit and name not in categories:
                categories[name] = hit.span()
    if categories:
        order = ["kutatas_fejlesztes", "infrastruktura_ingatlan"]
        names = [n for n in order if n in categories]
        first = min(categories.values())
        return found(names, first)
    if section is not None:
        return found(["egyeb"], section, "no_research_or_infrastructure_pattern")
    return Finding()


# --- egysz_elszam ------------------------------------------------------------------------

SIMPLIFIED = pattern(r"egyszerűsített (?:költség)?elszámol\w*|átalány\w*|egységköltség\w*")
SIMPLIFIED_EXCLUDED = pattern(
    r"egyszerűsített (?:költség)?elszámol\w*[^.|]{0,80}?(?:nincs lehetőség|nem releváns|nem alkalmaz\w*|nem lehetséges|kizár\w*)"
    r"|(?:nincs lehetőség|nem alkalmazható)[^.|]{0,40}?egyszerűsített (?:költség)?elszámol\w*"
)


def egysz_elszam(doc: Document) -> Finding:
    """Simplified cost options allowed. Every occurrence is considered, not only the first (ISS-19):
    an explicit exclusion gives 0; otherwise a mention of simplified costs, flat rates or unit costs gives 1."""
    for m in doc.anchors(SIMPLIFIED_EXCLUDED):
        return found(0, (m.start(), m.end()))
    for m in doc.anchors(SIMPLIFIED):
        return found(1, doc.sentence(m.start()))
    return Finding()


# --- biztositek --------------------------------------------------------------------------

COLLATERAL_OBLIGATION = pattern(
    r"biztosítéko?t[^.|]{0,80}?(?:kell nyújtani|köteles nyújtani|rendelkezésre bocsát\w*)"
    r"|kell[^.|]{0,60}?biztosítékot nyújtani|köteles biztosítéko?t"
    r"|biztosítéknyújtás\w* kötelező|biztosíték adása kötelező|biztosítéknyújtási kötelezettség terheli"
    r"|nyújtható biztosítékok köre|visszafizetésének biztosítéka lehet|kötelező fedezeti körbe"
)
CONDITIONAL_EXEMPTION = pattern(r"nem mentes a biztosítéknyújtási kötelezettség alól")
CONDITION = pattern(r"\b(?:ha|amennyiben)\b")  # "ha a biztosítéknyújtás kötelező, …" states no obligation
COLLATERAL_EXEMPTION = pattern(
    r"nem szükséges biztosíték\w*|nem kell biztosítékot|mentesül\w*|biztosítéknyújtási kötelezettség nem"
    r"|biztosíték\w*[^.|\n]{0,60}?nem releváns"
    r"|ÁÚF[\s-]*(?:21[\s-]*27|14[\s-]*20)?[”\"]?\s*6\.\s*(?:fejezet|pont)\w*"
)
COLLATERAL = pattern(r"biztosíték\w*")


def biztositek(doc: Document) -> Finding:
    """Collateral required: 1 if an obligation phrase is found; 0 otherwise, including the exemptions of the
    definition. Never None, as the definition requires. An obligation made conditional on the applicant
    not being exempt is an exemption case of the definition."""
    for m in doc.anchors(COLLATERAL_OBLIGATION):
        lo, _ = doc.sentence(m.start())
        if CONDITIONAL_EXEMPTION.search(doc.view, max(0, lo - 400), m.end()) or CONDITION.search(doc.view, lo, m.start()):
            continue
        return found(1, (lo, max(m.end(), doc.sentence(m.end() - 1)[1])))
    for m in doc.anchors(COLLATERAL):
        lo, hi = doc.sentence(m.start())
        window_end = min(len(doc.view), hi + 200)
        ex = COLLATERAL_EXEMPTION.search(doc.view, lo, window_end)
        if ex:
            return found(0, (lo, max(hi, ex.end())) if ex.end() - lo <= 500 else ex.span())
    return Finding(0, "found", None, ["default_value"])


RULES: dict[str, Callable[[Document], Finding]] = {
    "fin_form": fin_form,
    "tam_osszeg": tam_osszeg,
    "konzorcium": konzorcium,
    "bead_napok": bead_napok,
    "max_tam_int": max_tam_int,
    "eloleg": eloleg,
    "idotartam": idotartam,
    "tam_tevekenyseg": tam_tevekenyseg,
    "egysz_elszam": egysz_elszam,
    "biztositek": biztositek,
}
