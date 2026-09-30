"""Stage E1: validate extraction (Spec_E1_ValidateExtraction.md).

Compares each automated source (regex, the LLM with each model, and the old regex
baseline) with the expert's points on the gold documents, at the level of points
(SPEC-E1-01), and propagates each source's points into the risk label the way L3
would (SPEC-E1-04). ``compute`` is the pure computation; ``run`` reads the input runs
from the database and writes the tables, the report and the suggested L2 preferences.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import math
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from grantrisk.labelling import scoring
from grantrisk.labelling.scoring import FACTORS, HIGH, LABELS, LOW, MEDIUM
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

CATEGORIES = ("agree", "disagree", "not_found", "not_comparable")
ALL = "all"  # the pseudo-factor and the subset over everything
NEW_PERIOD = "2021-2027"
OTHER_PERIODS = "other"  # 2014-2020, RRF and VP: the calls not of the 2021-2027 ("Plusz") generation
SUBSETS = (ALL, NEW_PERIOD, OTHER_PERIODS)
REFERENCE = "gold"
BASELINE = "old_regex"
EVIDENCE_WARNING = "evidence_not_in_text"
LABEL_RANK = {LOW: 0, MEDIUM: 1, HIGH: 2}
WILSON_Z = 1.959963984540054  # the 97.5% quantile of the standard normal: a 95% interval

LIMITATIONS = (
    "**In-sample:** the regex rules and the LLM prompt were developed while looking at these gold documents, "
    "so the agreement is optimistic (ISS-26).",
    "**One expert:** the gold points come from one annotator, the author, as points rather than raw values.",
    "**Small sample:** {n} documents; one document changes a per-factor rate by {step:.1f} percentage points.",
    "**L2 preferences:** they are chosen on the same documents, so the agreement of the consolidated values "
    "is optimistic (DEC-18). Each extractor's own agreement, reported here, is not affected by this choice.",
    "**What gold measures:** the gold points apply the same rule definitions, so they check the extraction, "
    "not the risk concept itself (ISS-27; intent §1.6).",
)


class E1InputError(ValueError):
    """The input runs or files violate the E1 contract; nothing is written."""


@dataclass(frozen=True)
class Observation:
    """One source's observation of one factor. The baseline gives points instead of a value."""

    status: str
    value: Any = None
    points: int | None = None
    value_json: str | None = None
    evidence: str | None = None
    evidence_page: int | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoldDocument:
    doc_id: str
    gold_name: str
    call_code: str
    programme: str | None
    period: str
    points: Mapping[str, int | None]  # None where the expert did not score ("-")

    @property
    def subset(self) -> str:
        return NEW_PERIOD if self.period == NEW_PERIOD else OTHER_PERIODS


@dataclass(frozen=True)
class Cell:
    """One row of the detail table: a gold document, a factor and a source."""

    source: str
    doc_id: str
    factor: str
    gold_points: int | None
    points: int | None
    points_origin: str | None
    category: str
    note: str | None
    observation: Observation | None


@dataclass(frozen=True)
class LabelCuts:
    """What SPEC-E1-04 takes from the main L3 run: the factor means and the tercile cut scores."""

    means: Mapping[str, Fraction]
    last_low: Fraction | None
    last_medium: Fraction | None


@dataclass(frozen=True)
class DocumentLabel:
    source: str
    doc_id: str
    n_determined: int
    total: Fraction
    normalised: Fraction
    fixed_label: str
    tercile_label: str


@dataclass
class Tally:
    """The counts of SPEC-E1-01 for one source, factor and subset."""

    agree: int = 0
    disagree: int = 0
    not_found: int = 0
    not_comparable: int = 0
    abs_diff: int = 0
    confusion: list[list[int]] = field(default_factory=lambda: [[0] * 4 for _ in range(4)])

    def add(self, cell: Cell) -> None:
        setattr(self, cell.category, getattr(self, cell.category) + 1)
        if cell.category in ("agree", "disagree"):
            self.abs_diff += abs(cell.points - cell.gold_points)
            self.confusion[cell.gold_points][cell.points] += 1

    @property
    def scored(self) -> int:
        return self.agree + self.disagree + self.not_found

    @property
    def found(self) -> int:
        return self.agree + self.disagree

    def rates(self) -> dict[str, Any]:
        return {
            "agree": self.agree,
            "disagree": self.disagree,
            "not_found": self.not_found,
            "not_comparable": self.not_comparable,
            "agreement": _rate(self.agree, self.scored),
            "agreement_ci": wilson(self.agree, self.scored),
            "agreement_found": _rate(self.agree, self.found),
            "agreement_found_ci": wilson(self.agree, self.found),
            "coverage": _rate(self.found, self.scored),
            "mean_abs_diff": _rate(self.abs_diff, self.found),
            "confusion": [row[:] for row in self.confusion],
        }


@dataclass(frozen=True)
class E1Result:
    cells: list[Cell]
    agreements: dict[tuple[str, str, str], dict[str, Any]]  # (source, factor, subset) → rates
    labels: list[DocumentLabel]
    label_agreements: dict[tuple[str, str], dict[str, Any]]  # (source, label kind) → measures
    evidence_audit: dict[str, dict[str, dict[str, Any]]]
    preferences: dict[str, Any] | None
    missing_documents: dict[str, list[str]]
    out_of_domain: list[dict[str, Any]]


# --- Measures --------------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float] | None:
    """SPEC-E1-05: the Wilson score interval of the rate k / n; None when n is 0."""
    if n == 0:
        return None
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def quadratic_kappa(pairs: Sequence[tuple[int, int]], k: int = 3) -> Fraction | None:
    """Cohen's kappa with quadratic weights for (reference, rated) pairs of ranks 0 … k-1 (DEC-25).

    None when it is undefined: no pairs, or no disagreement expected by chance
    (e.g. both sides give every document the same label).
    """
    n = len(pairs)
    if n == 0:
        return None
    observed = [[0] * k for _ in range(k)]
    for a, b in pairs:
        observed[a][b] += 1
    rows = [sum(observed[i]) for i in range(k)]
    cols = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    def weight(i: int, j: int) -> Fraction:
        return Fraction((i - j) ** 2, (k - 1) ** 2)

    o = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k))
    e = sum(weight(i, j) * Fraction(rows[i] * cols[j], n) for i in range(k) for j in range(k))
    if e == 0:
        return None
    return 1 - Fraction(o) / e


# --- SPEC-E1-01: points and categories -------------------------------------------------------


def source_fin_form(observation: Observation | None) -> str | None:
    """A source's own financing form of a document, for the loan rule (DEC-40): its value, or the baseline's points."""
    if observation is None or observation.status != "found":
        return None
    return scoring.fin_form_of(observation.value, observation.points)


def observed_points(
    factor: str, observation: Observation | None, programme: str | None, fin_form: str | None = None
) -> tuple[int | None, str | None, str | None]:
    """(points, origin, note) of one observation, as L3 would determine them (SPEC-L3-14).

    A value that gives no points is treated as not found, with a note saying why. A
    missing value still gets points from the TOP rule where it applies (DEC-31). The loan
    rule (DEC-40) sets ``max_tam_int`` of the documents the source itself finds to be
    loans, also over the baseline's recorded points. A document missing from the run gives
    no points at all (§4 of the chapter).
    """
    note = None
    value = None
    if observation is None:
        return None, None, "document_missing_from_run"
    if observation.status == "found":
        if observation.points is not None:  # the baseline recorded points, not values
            if factor == "max_tam_int" and fin_form == "loan":
                return scoring.LOAN_RULE_POINTS, "loan_rule", None
            return observation.points, "given", None
        value = observation.value
        if factor == "tam_tevekenyseg" and value == []:
            note, value = "empty_activity_list", None  # DEC-33
        elif value is not None and scoring.domain_error(factor, value):
            note, value = f"out_of_domain: {value!r}", None
    points, origin = scoring.points(factor, value, programme, fin_form)
    return points, origin, note


def classify(
    source: str, doc: GoldDocument, factor: str, observation: Observation | None, fin_form: str | None = None
) -> Cell:
    """SPEC-E1-01: count one gold cell into exactly one category."""
    points, origin, note = observed_points(factor, observation, doc.programme, fin_form)
    gold = doc.points[factor]
    if gold is None:
        category = "not_comparable"
    elif points is None:
        category = "not_found"
    else:
        category = "agree" if points == gold else "disagree"
    return Cell(source, doc.doc_id, factor, gold, points, origin, category, note, observation)


# --- SPEC-E1-04: labels ------------------------------------------------------------------------


def tercile_label(normalised: Fraction, cuts: LabelCuts) -> str:
    """Up to the last low score of the main L3 run: low; up to its last medium score: medium; else high."""
    if cuts.last_low is not None and normalised <= cuts.last_low:
        return LOW
    if cuts.last_medium is not None and normalised <= cuts.last_medium:
        return MEDIUM
    return HIGH


def document_label(
    source: str, doc_id: str, points: Mapping[str, tuple[int | None, str | None]], cuts: LabelCuts
) -> DocumentLabel:
    """The label of one document from one source's (points, origin) per factor; missing points get the means."""
    filled = {f: Fraction(p) if p is not None else cuts.means[f] for f, (p, _) in points.items()}
    n_determined = sum(p is not None and origin not in scoring.RULE_ORIGINS for p, origin in points.values())
    total = sum((filled[f] for f in FACTORS), Fraction(0))
    normalised = scoring.normalised(total)
    _, fixed = scoring.fixed_label(normalised)
    return DocumentLabel(source, doc_id, n_determined, total, normalised, fixed, tercile_label(normalised, cuts))


def label_agreement(reference: Sequence[DocumentLabel], rated: Sequence[DocumentLabel]) -> dict[str, dict[str, Any]]:
    """Agreement, error sizes, weighted kappa and the mean total difference, for both label kinds."""
    by_doc = {r.doc_id: r for r in reference}
    n = len(rated)
    mean_total_diff = sum((r.total - by_doc[r.doc_id].total for r in rated), Fraction(0)) / n
    result = {}
    for kind in ("fixed", "tercile"):
        attr = f"{kind}_label"
        pairs = [(LABEL_RANK[getattr(by_doc[r.doc_id], attr)], LABEL_RANK[getattr(r, attr)]) for r in rated]
        agree = sum(a == b for a, b in pairs)
        errors = collections.Counter(b - a for a, b in pairs)
        kappa = quadratic_kappa(pairs)
        result[kind] = {
            "n_documents": n,
            "agree": agree,
            "agreement": _rate(agree, n),
            "agreement_ci": wilson(agree, n),
            "kappa_quadratic": None if kappa is None else float(kappa),
            "error_sizes": {str(d): errors.get(d, 0) for d in range(-2, 3)},
            "mean_total_diff": float(mean_total_diff),
        }
    return result


# --- SPEC-E1-06: preferences ---------------------------------------------------------------------


def suggest_preferences(
    agreements: Mapping[tuple[str, str, str], Mapping[str, Any]], llm_source: str | None
) -> tuple[dict[str, str] | None, str | None]:
    """The preferred L2 source per factor: the higher agreement of regex and the corpus LLM; regex on a tie.

    Returns (preferences, None), or (None, the reason no suggestion is possible).
    """
    sources = {s for s, _, _ in agreements}
    if "regex" not in sources:
        return None, "no regex run was evaluated"
    if llm_source is None:
        return None, "the configuration names no LLM model for the corpus run (extract.llm.model)"
    if llm_source not in sources:
        return None, f"no run of the corpus model {llm_source} was evaluated"
    preferences = {}
    for f in FACTORS:
        # Both rates share their denominator (the gold cells scored), so the counts decide exactly.
        regex, llm = agreements[("regex", f, ALL)]["agree"], agreements[(llm_source, f, ALL)]["agree"]
        preferences[f] = "llm" if llm > regex else "regex"
    return preferences, None


def preferences_yaml(preferences: Mapping[str, str], run_id: str, llm_source: str) -> str:
    """SPEC-E1-06: the suggestion in the format of the L2 configuration (SPEC-L2-03)."""
    header = (
        f"# Suggested L2 preferences from E1 run {run_id} (SPEC-E1-06): for each factor, the source with the\n"
        f"# higher agreement with the gold set, between regex and {llm_source}; regex on a tie.\n"
        "# Paste this section into the configuration unchanged.\n"
    )
    body = {"consolidate": {"preferred_source": dict(preferences), "preferences_from_e1_run": run_id}}
    return header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True)


# --- The whole computation -------------------------------------------------------------------


def source_order(sources: Iterable[str]) -> list[str]:
    """Regex first, then the LLM models, then any other source, and the old baseline last."""
    def key(s: str) -> tuple[int, str]:
        return (0 if s == "regex" else 1 if s.startswith("llm:") else 3 if s == BASELINE else 2, s)

    return sorted(set(sources), key=key)


def compute(
    gold_docs: Sequence[GoldDocument],
    sources: Mapping[str, Mapping[tuple[str, str], Observation]],
    cuts: LabelCuts,
    llm_source: str | None = None,
) -> E1Result:
    """SPEC-E1-01 … -06 on the gold documents, for every source."""
    if not gold_docs:
        raise E1InputError("the gold import has no documents")
    if not sources:
        raise E1InputError("no source to evaluate: name at least one L1 run or the old baseline")
    docs = sorted(gold_docs, key=lambda d: d.doc_id)
    order = source_order(sources)

    cells = [classify(s, d, f, sources[s].get((d.doc_id, f)), source_fin_form(sources[s].get((d.doc_id, "fin_form"))))
             for s in order for d in docs for f in FACTORS]

    tallies: dict[tuple[str, str, str], Tally] = {
        (s, f, subset): Tally() for s in order for f in (*FACTORS, ALL) for subset in SUBSETS
    }
    subset_of = {d.doc_id: d.subset for d in docs}
    for c in cells:
        for f in (c.factor, ALL):
            for subset in (ALL, subset_of[c.doc_id]):
                tallies[(c.source, f, subset)].add(c)
    agreements = {key: t.rates() for key, t in tallies.items()}

    # SPEC-E1-04: the reference labels from the expert's points, then each source's.
    reference = [
        document_label(REFERENCE, d.doc_id, {f: (d.points[f], "manual") for f in FACTORS}, cuts) for d in docs
    ]
    programme = {d.doc_id: d.programme for d in docs}
    by_source_doc: dict[tuple[str, str], dict[str, tuple[int | None, str | None]]] = collections.defaultdict(dict)
    for c in cells:
        # L3 would apply the TOP rule to a document the run missed, since L2 leaves its amount NULL.
        points = (c.points, c.points_origin) if c.observation is not None else scoring.points(
            c.factor, None, programme[c.doc_id])
        by_source_doc[(c.source, c.doc_id)][c.factor] = points
    labels = list(reference)
    label_agreements = {}
    for s in order:
        rated = [document_label(s, d.doc_id, by_source_doc[(s, d.doc_id)], cuts) for d in docs]
        labels.extend(rated)
        for kind, measures in label_agreement(reference, rated).items():
            label_agreements[(s, kind)] = measures

    # SPEC-E1-03: the share of found values whose evidence is not in the text.
    evidence_audit: dict[str, dict[str, dict[str, Any]]] = {}
    for s in order:
        if s == BASELINE:
            continue  # the baseline recorded no evidence
        counts = {f: [0, 0] for f in (*FACTORS, ALL)}
        for c in cells:
            o = c.observation
            if c.source == s and o is not None and o.status == "found":
                flagged = any(w.split(":")[0] == EVIDENCE_WARNING for w in o.warnings)
                for f in (c.factor, ALL):
                    counts[f][0] += 1
                    counts[f][1] += flagged
        evidence_audit[s] = {
            f: {"found": n, "evidence_not_in_text": k, "share": _rate(k, n)} for f, (n, k) in counts.items()
        }

    preferences, reason = suggest_preferences(agreements, llm_source)
    missing = {
        s: sorted(d.doc_id for d in docs if not any((d.doc_id, f) in sources[s] for f in FACTORS)) for s in order
    }
    out_of_domain = [
        {"source": c.source, "doc_id": c.doc_id, "factor": c.factor, "note": c.note}
        for c in cells
        if c.note and c.note.startswith("out_of_domain")
    ]
    return E1Result(
        cells=cells,
        agreements=agreements,
        labels=labels,
        label_agreements=label_agreements,
        evidence_audit=evidence_audit,
        preferences={"llm_source": llm_source, "preferred_source": preferences, "not_suggested_because": reason},
        missing_documents={s: m for s, m in missing.items() if m},
        out_of_domain=out_of_domain,
    )


# --- Inputs from the database and the old baseline ------------------------------------------


def load_gold(conn: sqlite3.Connection, manual_run_id: str, c1_run_id: str) -> tuple[str, list[GoldDocument]]:
    """The gold set name and the gold documents, with each one's programme and period from the C1 run."""
    rows = conn.execute(
        "SELECT gold_set, gold_name, doc_id, call_code, factor, points FROM gold_records WHERE run_id = ?",
        (manual_run_id,),
    ).fetchall()
    if not rows:
        raise E1InputError(f"run {manual_run_id} holds no gold records")
    gold_sets = {r["gold_set"] for r in rows}
    if len(gold_sets) != 1:
        raise E1InputError(f"run {manual_run_id} holds several gold sets: {sorted(gold_sets)}")
    documents = {
        r["doc_id"]: r
        for r in conn.execute("SELECT doc_id, programme, period FROM documents WHERE run_id = ?", (c1_run_id,))
    }
    by_doc: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = by_doc.setdefault(r["doc_id"], {"gold_name": r["gold_name"], "call_code": r["call_code"], "points": {}})
        d["points"][r["factor"]] = r["points"]
    problems = [f"{doc_id}: no Document record in run {c1_run_id}" for doc_id in sorted(by_doc) if doc_id not in documents]
    problems += [
        f"{doc_id}: missing gold factors {[f for f in FACTORS if f not in d['points']]}"
        for doc_id, d in sorted(by_doc.items())
        if any(f not in d["points"] for f in FACTORS)
    ]
    if problems:
        raise E1InputError("invalid E1 input:\n  " + "\n  ".join(problems))
    docs = [
        GoldDocument(doc_id, d["gold_name"], d["call_code"], documents[doc_id]["programme"],
                     documents[doc_id]["period"], d["points"])
        for doc_id, d in sorted(by_doc.items())
    ]
    return gold_sets.pop(), docs


def run_source(conn: sqlite3.Connection, run_id: str) -> str:
    """The one automated source an L1 run holds."""
    runs.require_complete(conn, run_id, "L1")
    sources = sorted(r[0] for r in conn.execute("SELECT DISTINCT source FROM factor_observations WHERE run_id = ?", (run_id,)))
    if len(sources) != 1:
        raise E1InputError(f"run {run_id} holds sources {sources}, not exactly one")
    if sources[0] == "manual":
        raise E1InputError(f"run {run_id} is the manual import; name it as the gold import, not as a source")
    if sources[0] in (BASELINE, REFERENCE):
        raise E1InputError(f"run {run_id} holds the source {sources[0]!r}, a name E1 reserves")
    return sources[0]


def load_observations(
    conn: sqlite3.Connection, run_id: str, doc_ids: Iterable[str]
) -> dict[tuple[str, str], Observation]:
    wanted = set(doc_ids)
    rows = conn.execute(
        "SELECT doc_id, factor, status, value_json, points, evidence, evidence_page, warnings_json"
        " FROM factor_observations WHERE run_id = ?",
        (run_id,),
    )
    return {
        (r["doc_id"], r["factor"]): Observation(
            status=r["status"],
            value=None if r["value_json"] is None else json.loads(r["value_json"], parse_float=Decimal),
            points=r["points"],
            value_json=r["value_json"],
            evidence=r["evidence"],
            evidence_page=r["evidence_page"],
            warnings=tuple(json.loads(r["warnings_json"] or "[]")),
        )
        for r in rows
        if r["doc_id"] in wanted
    }


def read_old_baseline(
    path: Path, gold_docs: Sequence[GoldDocument], renamed: Mapping[str, str]
) -> dict[tuple[str, str], Observation]:
    """The old regex points (column ``regex``) of Gold_second_test_results_final.csv, for SPEC-E1-02.

    Rows are matched to the gold documents by the call's name in the gold file;
    ``renamed`` maps a name the baseline still uses to the gold file's current name.
    An empty cell means the old regex found nothing.
    """
    by_name = {d.gold_name: d.doc_id for d in gold_docs}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing_columns = [c for c in ("felhivas", "factor", "regex") if c not in (reader.fieldnames or [])]
        if missing_columns:
            raise E1InputError(f"{path.name}: missing columns {missing_columns}")
        rows = list(reader)
    problems = []
    result = {}
    for r in rows:
        name = renamed.get(r["felhivas"].strip(), r["felhivas"].strip())
        doc_id = by_name.get(name)
        factor, cell = r["factor"].strip(), r["regex"].strip()
        if doc_id is None:
            problems.append(f"{name}: not a call of the gold set (add it to validate.old_baseline_renamed?)")
            continue
        if factor not in FACTORS:
            problems.append(f"{name}: unknown factor {factor!r}")
            continue
        if cell == "":
            result[(doc_id, factor)] = Observation(status="not_found")
            continue
        try:
            points = float(cell)
        except ValueError:
            points = -1.0
        if not (points.is_integer() and 0 <= points <= 3):
            problems.append(f"{name} / {factor}: {cell!r} is not a number of points")
            continue
        result[(doc_id, factor)] = Observation(status="found", points=int(points))
    if problems:
        raise E1InputError(f"invalid old baseline {path.name}:\n  " + "\n  ".join(sorted(set(problems))))
    return result


def load_cuts(conn: sqlite3.Connection, data_root: Path, l3_run_id: str) -> LabelCuts:
    """The factor means and the tercile cut scores from the main L3 run's report."""
    row = runs.get(conn, l3_run_id)
    if not row["report_path"]:
        raise E1InputError(f"run {l3_run_id} has no report")
    report = json.loads((data_root / row["report_path"]).read_text(encoding="utf-8"))
    try:
        means = {f: Fraction(report["factor_means"][f]["exact"]) for f in FACTORS}
        cuts = report["tercile_cuts"]
        last_low, last_medium = cuts["last_low_score"], cuts["last_medium_score"]
    except KeyError as exc:
        raise E1InputError(f"the report of run {l3_run_id} lacks {exc}; rerun L3") from None
    return LabelCuts(
        means,
        None if last_low is None else Fraction(last_low),
        None if last_medium is None else Fraction(last_medium),
    )


# --- The report (INT-OUT-05) -----------------------------------------------------------------


def _fmt(x: float | None, digits: int = 3) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def _fmt_rate(rates: Mapping[str, Any], key: str = "agreement") -> str:
    num = rates["agree"]
    den = rates["agree"] + rates["disagree"] + (rates["not_found"] if key == "agreement" else 0)
    ci = rates[f"{key}_ci"]
    interval = "" if ci is None else f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    return f"{_fmt(rates[key])} ({num}/{den}){interval}"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def markdown_report(result: E1Result, meta: Mapping[str, Any]) -> str:
    order = source_order({s for s, _, _ in result.agreements})
    n_docs = meta["gold_documents"]
    a = result.agreements
    lines = [
        "# E1: validation of the extraction against the gold set",
        "",
        f"Run `{meta['run_id']}`. Gold set `{meta['gold_set']}` from the gold import `{meta['input_runs']['manual']}`: "
        f"{n_docs} documents. Main L3 run (factor means, tercile cuts): `{meta['input_runs']['L3']}`. "
        f"C1 run (programme, period): `{meta['input_runs']['C1']}`.",
        "",
        "Sources:",
        "",
    ]
    for s in order:
        origin = meta["sources"][s]
        lines.append(f"- `{s}`: " + (f"L1 run `{origin}`" if s != BASELINE else f"`{origin['file']}` (sha256 {origin['sha256'][:12]}…), column `regex`"))
    lines += [
        "",
        "## Definitions",
        "",
        "- **Reference:** the expert's points (0–3) of the gold set. Every extracted value is turned into points with "
        "L3's point function (SPEC-L3-14), including the TOP rule (DEC-31); a missing `tam_osszeg` of a TOP or "
        "TOP Plusz call therefore gets the rule's points. An empty `tam_tevekenyseg` list counts as not found (DEC-33). "
        "The old baseline recorded points, not values; they are used as recorded, and the TOP rule applies to its "
        "missing cells like to every other source.",
        "- **Categories:** each gold cell counts once: `agree` (same points), `disagree` (different points), "
        "`not_found` (the source gave no points: not found, ambiguous, error, or the document is missing from the run), "
        "`not_comparable` (the expert did not score the cell, \"-\"; outside every rate).",
        "- **agreement** = agree / (agree + disagree + not_found): a missing value counts as an extraction failure. "
        "**agreement on found values** = agree / (agree + disagree). **coverage** = (agree + disagree) / "
        "(agree + disagree + not_found). **mean |Δ|**: the mean absolute difference in points where both sides have points.",
        "- **Intervals:** 95% Wilson score intervals (SPEC-E1-05). Cells read `rate (numerator/denominator) [low, high]`.",
        "",
        "## 1. Agreement per factor (INT-RQ-A1)",
        "",
    ]
    lines += _table(
        ["Factor", *[f"`{s}`" for s in order]],
        [[f if f != ALL else "**all factors**", *[_fmt_rate(a[(s, f, ALL)]) for s in order]] for f in (*FACTORS, ALL)],
    )
    lines += ["", "## 2. Agreement on the values found, coverage and mean |Δ|", ""]
    lines += _table(
        ["Factor", *[f"`{s}` found-agreement" for s in order], *[f"`{s}` coverage" for s in order],
         *[f"`{s}` mean abs. diff" for s in order]],
        [
            [f if f != ALL else "**all factors**",
             *[_fmt_rate(a[(s, f, ALL)], "agreement_found") for s in order],
             *[_fmt(a[(s, f, ALL)]["coverage"]) for s in order],
             *[_fmt(a[(s, f, ALL)]["mean_abs_diff"]) for s in order]]
            for f in (*FACTORS, ALL)
        ],
    )
    periods = meta["periods"]
    lines += [
        "",
        "## 3. By period (INT-NFR-07)",
        "",
        f"`2021-2027`: the calls of the 2021–2027 (\"Plusz\") programmes, {periods['groups'][NEW_PERIOD]} documents. "
        f"`other`: every other period ({', '.join(f'{p}: {n}' for p, n in periods['by_period'].items() if p != NEW_PERIOD) or 'none'}), "
        f"{periods['groups'][OTHER_PERIODS]} documents. Agreement over all factors:",
        "",
    ]
    lines += _table(
        ["Source", "all", "2021-2027", "other"],
        [[f"`{s}`", *[_fmt_rate(a[(s, ALL, subset)]) for subset in SUBSETS]] for s in order],
    )
    lines += ["", "Per factor and period:", ""]
    lines += _table(
        ["Factor", *[f"`{s}` {subset}" for s in order for subset in (NEW_PERIOD, OTHER_PERIODS)]],
        [[f, *[_fmt(a[(s, f, subset)]["agreement"]) for s in order for subset in (NEW_PERIOD, OTHER_PERIODS)]]
         for f in FACTORS],
    )
    audited = [s for s in order if s in result.evidence_audit]
    lines += [
        "",
        "## 4. Evidence audit (SPEC-E1-03)",
        "",
        "The share of found values whose quoted evidence was not found in the document text "
        f"(warning `{EVIDENCE_WARNING}`, SPEC-L1-02). The old baseline recorded no evidence.",
        "",
    ]
    if audited:
        lines += _table(
            ["Factor", *[f"`{s}`" for s in audited]],
            [[f if f != ALL else "**all factors**",
              *[f"{_fmt(result.evidence_audit[s][f]['share'])} ({result.evidence_audit[s][f]['evidence_not_in_text']}/"
                f"{result.evidence_audit[s][f]['found']})" for s in audited]]
             for f in (*FACTORS, ALL)],
        )
    else:
        lines.append("No L1 source was evaluated.")
    ref_sizes = collections.Counter(lab.fixed_label for lab in result.labels if lab.source == REFERENCE)
    ref_terciles = collections.Counter(lab.tercile_label for lab in result.labels if lab.source == REFERENCE)
    lines += [
        "",
        "## 5. Label-level agreement (INT-RQ-A2, SPEC-E1-04)",
        "",
        "Each source's label is computed the way L3 would: points from the values (with the TOP rule), missing factors "
        "filled with the factor means of the main L3 run, the total and the normalised score, the fixed label, and the "
        "tercile label from the main L3 run's cut scores. The reference label is computed in the same way from the "
        "expert's points. There are no holistic expert labels (intent §1.6), so this measures how extraction errors "
        "propagate into the label, not the validity of the label itself.",
        "",
        f"Reference labels: fixed {', '.join(f'{lab} {ref_sizes.get(lab, 0)}' for lab in LABELS)}; "
        f"tercile {', '.join(f'{lab} {ref_terciles.get(lab, 0)}' for lab in LABELS)}.",
        "",
        "Error size: the source's label minus the reference label, on low = 0, medium = 1, high = 2. "
        "κw: Cohen's kappa with quadratic weights (– where undefined). Δ total: the mean of the source's total score "
        "minus the reference total (0–30 points).",
        "",
    ]
    rows = []
    for s in order:
        for kind in ("fixed", "tercile"):
            m = result.label_agreements[(s, kind)]
            ci = m["agreement_ci"]
            rows.append([
                f"`{s}`", kind,
                f"{_fmt(m['agreement'])} ({m['agree']}/{m['n_documents']})" + ("" if ci is None else f" [{ci[0]:.2f}, {ci[1]:.2f}]"),
                _fmt(m["kappa_quadratic"]),
                *[str(m["error_sizes"][str(d)]) for d in range(-2, 3)],
                _fmt(m["mean_total_diff"], 2),
            ])
    lines += _table(["Source", "Label", "Agreement", "κw", "−2", "−1", "0", "+1", "+2", "Δ total"], rows)

    prefs = result.preferences
    lines += ["", "## 6. Suggested L2 preferences (SPEC-E1-06)", ""]
    if prefs["preferred_source"] is None:
        lines.append(f"No suggestion: {prefs['not_suggested_because']}.")
    else:
        llm = prefs["llm_source"]
        lines += [
            f"For each factor, the source with the higher agreement between `regex` and `{llm}` (the model of the "
            "corpus run); `regex` on a tie, because it is local and its rules can be inspected (SPEC-L2-03). "
            "The same suggestion is in `l2_preferences.yaml`, ready to paste into the configuration.",
            "",
        ]
        lines += _table(
            ["Factor", "`regex`", f"`{llm}`", "Preferred"],
            [[f, _fmt(a[("regex", f, ALL)]["agreement"]), _fmt(a[(llm, f, ALL)]["agreement"]),
              prefs["preferred_source"][f]] for f in FACTORS],
        )
    lines += ["", "## 7. Missing documents and values outside the domain", ""]
    if result.missing_documents:
        for s, docs in result.missing_documents.items():
            lines.append(f"- `{s}` has no observation for {len(docs)} gold document(s), counted as `not_found`: "
                         + ", ".join(f"`{meta['call_codes'][d]}`" for d in docs))
    else:
        lines.append("- Every source has observations for every gold document.")
    if result.out_of_domain:
        lines.append(f"- {len(result.out_of_domain)} found value(s) lie outside the factor's domain and count as "
                     "`not_found` (see the detail table, column `note`).")
    step = 100 / n_docs
    lines += ["", "## 8. Limitations (SPEC-E1-07)", ""]
    lines += [f"- {text.format(n=n_docs, step=step)}" for text in LIMITATIONS]
    return "\n".join(lines) + "\n"


def _csv(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue()


def _ci(ci: tuple[float, float] | None, i: int) -> float | None:
    return None if ci is None else ci[i]


def agreement_csv(result: E1Result) -> str:
    header = ["source", "factor", "subset", "agree", "disagree", "not_found", "not_comparable", "agreement",
              "agreement_low", "agreement_high", "agreement_found", "agreement_found_low", "agreement_found_high",
              "coverage", "mean_abs_diff"]
    rows = [
        [s, f, subset, r["agree"], r["disagree"], r["not_found"], r["not_comparable"], r["agreement"],
         _ci(r["agreement_ci"], 0), _ci(r["agreement_ci"], 1), r["agreement_found"],
         _ci(r["agreement_found_ci"], 0), _ci(r["agreement_found_ci"], 1), r["coverage"], r["mean_abs_diff"]]
        for (s, f, subset), r in result.agreements.items()
    ]
    return _csv(header, rows)


def details_csv(result: E1Result, meta: Mapping[str, Any]) -> str:
    header = ["source", "doc_id", "call_code", "factor", "gold_points", "value", "points", "points_origin", "status",
              "category", "note", "evidence", "evidence_page", "warnings"]
    rows = []
    for c in result.cells:
        o = c.observation
        rows.append([
            c.source, c.doc_id, meta["call_codes"][c.doc_id], c.factor, c.gold_points,
            None if o is None else o.value_json, c.points, c.points_origin, None if o is None else o.status,
            c.category, c.note, None if o is None else o.evidence, None if o is None else o.evidence_page,
            None if o is None else ";".join(o.warnings),
        ])
    return _csv(header, rows)


def labels_csv(result: E1Result) -> str:
    header = ["source", "label_kind", "n_documents", "agree", "agreement", "agreement_low", "agreement_high",
              "kappa_quadratic", *[f"error_{d:+d}" for d in range(-2, 3)], "mean_total_diff"]
    rows = [
        [s, kind, m["n_documents"], m["agree"], m["agreement"], _ci(m["agreement_ci"], 0), _ci(m["agreement_ci"], 1),
         m["kappa_quadratic"], *[m["error_sizes"][str(d)] for d in range(-2, 3)], m["mean_total_diff"]]
        for (s, kind), m in result.label_agreements.items()
    ]
    return _csv(header, rows)


def json_report(result: E1Result, meta: Mapping[str, Any]) -> dict[str, Any]:
    def by_source(items: Mapping[tuple, Any]) -> dict[str, Any]:
        nested: dict[str, Any] = {}
        for key, value in items.items():
            target = nested
            for part in key[:-1]:
                target = target.setdefault(part, {})
            target[key[-1]] = value
        return nested

    return {
        **{k: v for k, v in meta.items() if k not in ("call_codes", "run_id")},
        "scoring_rules_version": scoring.RULES_VERSION,
        "definitions": {
            "agreement": "agree / (agree + disagree + not_found)",
            "agreement_found": "agree / (agree + disagree)",
            "coverage": "(agree + disagree) / (agree + disagree + not_found)",
            "mean_abs_diff": "mean |extracted points - gold points| where both have points",
            "intervals": "95% Wilson score intervals",
            "subsets": {NEW_PERIOD: "period 2021-2027", OTHER_PERIODS: "every other period"},
            "error_size": "source label minus reference label, low = 0, medium = 1, high = 2",
        },
        "agreement": by_source(result.agreements),
        "label_agreement": by_source(result.label_agreements),
        "evidence_audit": result.evidence_audit,
        "suggested_preferences": result.preferences,
        "missing_documents": result.missing_documents,
        "out_of_domain": result.out_of_domain,
        "limitations": [text.format(n=meta["gold_documents"], step=100 / meta["gold_documents"]) for text in LIMITATIONS],
    }


# --- Database run (SPEC-E1-08) ---------------------------------------------------------------


def _write_text(data_root: Path, relative_path: str, text: str) -> str:
    """Write a text file and return its relative path. Never overwrites a file (ARC-02)."""
    path = data_root / relative_path
    if path.exists():
        raise FileExistsError(f"{relative_path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)
    return relative_path


def _insert(conn: sqlite3.Connection, run_id: str, result: E1Result) -> None:
    conn.executemany(
        "INSERT INTO extraction_agreements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (run_id, s, f, subset, r["agree"], r["disagree"], r["not_found"], r["not_comparable"],
             r["agreement"], _ci(r["agreement_ci"], 0), _ci(r["agreement_ci"], 1),
             r["agreement_found"], _ci(r["agreement_found_ci"], 0), _ci(r["agreement_found_ci"], 1),
             r["coverage"], r["mean_abs_diff"], json.dumps(r["confusion"]))
            for (s, f, subset), r in result.agreements.items()
        ],
    )
    conn.executemany(
        "INSERT INTO extraction_details VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (run_id, c.source, c.doc_id, c.factor, c.gold_points,
             None if c.observation is None else c.observation.value_json, c.points, c.points_origin,
             None if c.observation is None else c.observation.status, c.category, c.note,
             None if c.observation is None else c.observation.evidence,
             None if c.observation is None else c.observation.evidence_page,
             None if c.observation is None else json.dumps(list(c.observation.warnings)))
            for c in result.cells
        ],
    )
    conn.executemany(
        "INSERT INTO extraction_labels VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (run_id, lab.source, lab.doc_id, lab.n_determined, float(lab.total), str(lab.total),
             float(lab.normalised), str(lab.normalised), lab.fixed_label, lab.tercile_label)
            for lab in result.labels
        ],
    )
    conn.executemany(
        "INSERT INTO extraction_label_agreements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (run_id, s, kind, m["n_documents"], m["agree"], m["agreement"], _ci(m["agreement_ci"], 0),
             _ci(m["agreement_ci"], 1), m["kappa_quadratic"], json.dumps(m["error_sizes"]), m["mean_total_diff"])
            for (s, kind), m in result.label_agreements.items()
        ],
    )


def run(
    conn: sqlite3.Connection,
    config_values: Mapping[str, Any],
    data_root: Path,
    *,
    manual_run_id: str,
    extraction_run_ids: Sequence[str],
    l3_run_id: str,
    c1_run_id: str,
    old_baseline_csv: Path | None = None,
) -> str:
    """Validate the named L1 runs (and the old baseline) against the gold import. Returns the new run id.

    On any error the run is marked failed, and neither rows nor report files remain.
    """
    runs.require_complete(conn, manual_run_id, "L1")
    manual_sources = {r[0] for r in conn.execute(
        "SELECT DISTINCT source FROM factor_observations WHERE run_id = ?", (manual_run_id,))}
    if manual_sources != {"manual"}:
        raise E1InputError(f"run {manual_run_id} holds sources {sorted(manual_sources)}, not manual")
    source_runs: dict[str, str] = {}
    for run_id in extraction_run_ids:
        source = run_source(conn, run_id)
        if source in source_runs:
            raise E1InputError(f"runs {source_runs[source]} and {run_id} both hold source {source}; name one")
        source_runs[source] = run_id
    runs.require_complete(conn, l3_run_id, "L3")
    runs.require_complete(conn, c1_run_id, "C1")
    settings = config_values.get("validate") or {}
    renamed = settings.get("old_baseline_renamed") or {}
    llm_model = ((config_values.get("extract") or {}).get("llm") or {}).get("model")
    llm_source = f"llm:{llm_model}" if llm_model else None

    snapshot = dict(config_values)
    baseline_meta = None
    if old_baseline_csv is not None:
        baseline_meta = {"file": Path(old_baseline_csv).name,
                         "sha256": hashlib.sha256(Path(old_baseline_csv).read_bytes()).hexdigest()}
        snapshot["old_baseline_file"] = {"path": str(old_baseline_csv), "sha256": baseline_meta["sha256"]}
    inputs = sorted({manual_run_id, *source_runs.values(), l3_run_id, c1_run_id})
    run_id = runs.start(conn, "E1", snapshot, inputs=inputs)
    written: list[str] = []
    try:
        gold_set, gold_docs = load_gold(conn, manual_run_id, c1_run_id)
        doc_ids = [d.doc_id for d in gold_docs]
        sources = {s: load_observations(conn, r, doc_ids) for s, r in source_runs.items()}
        if old_baseline_csv is not None:
            sources[BASELINE] = read_old_baseline(Path(old_baseline_csv), gold_docs, renamed)
        cuts = load_cuts(conn, data_root, l3_run_id)
        result = compute(gold_docs, sources, cuts, llm_source)

        by_period = collections.Counter(d.period for d in gold_docs)
        meta = {
            "run_id": run_id,
            "input_runs": {"manual": manual_run_id, "L3": l3_run_id, "C1": c1_run_id, "sources": dict(source_runs)},
            "gold_set": gold_set,
            "gold_documents": len(gold_docs),
            "sources": {**source_runs, **({BASELINE: baseline_meta} if baseline_meta else {})},
            "periods": {
                "by_period": dict(sorted(by_period.items())),
                "groups": {g: sum(d.subset == g for d in gold_docs) for g in (NEW_PERIOD, OTHER_PERIODS)},
            },
            "call_codes": {d.doc_id: d.call_code for d in gold_docs},
        }
        folder = f"reports/{run_id}"
        report = json_report(result, meta)
        if result.preferences["preferred_source"] is not None:
            text = preferences_yaml(result.preferences["preferred_source"], run_id, llm_source)
            written.append(_write_text(data_root, f"{folder}/l2_preferences.yaml", text))
        written.append(_write_text(data_root, f"{folder}/e1_report.md", markdown_report(result, meta)))
        written.append(_write_text(data_root, f"{folder}/e1_agreement.csv", agreement_csv(result)))
        written.append(_write_text(data_root, f"{folder}/e1_labels.csv", labels_csv(result)))
        written.append(_write_text(data_root, f"{folder}/e1_details.csv", details_csv(result, meta)))
        report["files"] = sorted(p.rsplit("/", 1)[1] for p in written)
        report_path = files.write_json(data_root, f"{folder}/e1_report.json", report)
        written.append(report_path)
        with transaction(conn):
            _insert(conn, run_id, result)
            runs.complete(conn, run_id, report_path)
    except Exception as exc:
        for path in written:
            files.remove(data_root, path)
        runs.fail(conn, run_id, f"{type(exc).__name__}: {exc}")
        raise
    return run_id
