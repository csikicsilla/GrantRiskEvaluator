"""The appendix material of the thesis (SPEC-E3-05), generated so that it cannot drift from what was run.

8.1 the factors and the scoring table, checked against the scoring module; 8.2 the
database schema as a Mermaid ER diagram; 8.3 the commands that reproduced the chain;
8.4 every use of an external model, from the run records; and the data-flow diagram.
"""

from __future__ import annotations

import collections
import json
import sqlite3
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from grantrisk.labelling import scoring
from grantrisk.labelling.scoring import FACTORS
from grantrisk.reporting.chain import Data
from grantrisk.reporting.common import fmt, md_document, md_table, provenance


class AppendixError(ValueError):
    """The scoring table of the appendix disagrees with the scoring module."""


FACTOR_NAMES = {  # Spec_ScoringSystem.md §1.2: (Hungarian, English)
    "fin_form": ("Finanszírozási forma", "Form of financing"),
    "tam_osszeg": ("Támogatási összeg maximuma", "Maximum grant amount"),
    "konzorcium": ("Konzorcium vagy egyedi pályázás", "Consortium or single applicant"),
    "bead_napok": ("Beadásra rendelkezésre álló napok száma", "Days available for submission"),
    "max_tam_int": ("Támogatási intenzitás maximuma", "Maximum aid intensity"),
    "eloleg": ("Előleg", "Advance payment"),
    "idotartam": ("Projekt befejezésére rendelkezésre álló hónapok száma", "Months available to complete the project"),
    "tam_tevekenyseg": ("A támogatható tevékenységek között van", "The eligible activities include"),
    "egysz_elszam": ("Lehetőség egyszerűsített elszámolásra", "Simplified cost options allowed"),
    "biztositek": ("Biztosítékadási kötelezettség", "Collateral required"),
}


@dataclass(frozen=True)
class Band:
    """One row of the scoring table: the values it covers and their points.

    A numeric band is the interval (lo, hi) with the given inclusiveness (hi None: no upper
    bound); a categorical band lists its values.
    """

    points: int
    hu: str
    en: str
    lo: Fraction | None = None
    lo_incl: bool = True
    hi: Fraction | None = None
    hi_incl: bool = True
    values: tuple[Any, ...] = ()

    def covers(self, value: Any) -> bool:
        if self.values:
            return value in self.values
        x = scoring.as_number(value)
        if x is None:
            return False
        above = x >= self.lo if self.lo_incl else x > self.lo
        below = self.hi is None or (x <= self.hi if self.hi_incl else x < self.hi)
        return above and below

    def probes(self, step: Fraction) -> list[Any]:
        """Values at both ends of the band, to check against the scoring module."""
        if self.values:
            return list(self.values)
        lo = self.lo if self.lo_incl else self.lo + step
        hi = (self.lo + 1000 * step) if self.hi is None else self.hi if self.hi_incl else self.hi - step
        return [lo, hi]


F = Fraction
KF, INFRA, EGYEB = "kutatas_fejlesztes", "infrastruktura_ingatlan", "egyeb"
BANDS: dict[str, list[Band]] = {
    "fin_form": [
        Band(3, "vissza nem térítendő", "non-repayable grant", values=("grant",)),
        Band(2, "feltételesen vissza nem térítendő", "conditionally non-repayable grant", values=("conditional_grant",)),
        Band(1, "visszatérítendő / hitel", "repayable / loan", values=("loan",)),
    ],
    "tam_osszeg": [
        Band(0, "0–100 000 000 Ft", "0–100,000,000 HUF", F(0), True, F(100_000_000), True),
        Band(1, "100 000 001–300 000 000 Ft", "100,000,001–300,000,000 HUF", F(100_000_000), False, F(300_000_000), True),
        Band(2, "300 000 001–600 000 000 Ft", "300,000,001–600,000,000 HUF", F(300_000_000), False, F(600_000_000), True),
        Band(3, "600 000 000 Ft felett", "above 600,000,000 HUF", F(600_000_000), False, None),
    ],
    "konzorcium": [
        Band(0, "egyedi pályázás (0)", "single applicant only (0)", values=(0,)),
        Band(3, "konzorcium megengedett (1)", "consortium allowed (1)", values=(1,)),
    ],
    "bead_napok": [
        Band(3, "legfeljebb 7 nap", "at most 7 days", F(0), True, F(7), True),
        Band(2, "8–15 nap", "8–15 days", F(7), False, F(15), True),
        Band(1, "16–30 nap", "16–30 days", F(15), False, F(30), True),
        Band(0, "legalább 31 nap", "31 days or more", F(30), False, None),
        Band(0, "a keret kimerüléséig", "until the funds run out", values=(scoring.UNTIL_FUNDS_RUN_OUT,)),
    ],
    "max_tam_int": [
        Band(0, "legfeljebb 30%", "at most 30%", F(0), True, F(30), True),
        Band(1, "30% felett, legfeljebb 50%", "above 30%, at most 50%", F(30), False, F(50), True),
        Band(2, "50% felett, 70% alatt", "above 50%, below 70%", F(50), False, F(70), False),
        Band(3, "legalább 70%", "70% or more", F(70), True, F(100), True),
    ],
    "eloleg": [
        Band(0, "legfeljebb 30%", "at most 30%", F(0), True, F(30), True),
        Band(1, "30% felett, legfeljebb 50%", "above 30%, at most 50%", F(30), False, F(50), True),
        Band(2, "50% felett, legfeljebb 70%", "above 50%, at most 70%", F(50), False, F(70), True),
        Band(3, "70% felett", "above 70%", F(70), False, F(100), True),
    ],
    "idotartam": [
        Band(0, "legfeljebb 12 hónap", "at most 12 months", F(0), False, F(12), True),
        Band(1, "12 felett, legfeljebb 18 hónap", "above 12, at most 18 months", F(12), False, F(18), True),
        Band(2, "18 felett, legfeljebb 24 hónap", "above 18, at most 24 months", F(18), False, F(24), True),
        Band(3, "24 hónap felett", "above 24 months", F(24), False, None),
        Band(3, "maximális (az időtartam nem értelmezett, vagy a kérelem a befejezési határidőig benyújtható)",
             "the longest (the duration is not defined, or applications are open until the completion deadline)",
             values=(scoring.LONGEST_DURATION,)),
    ],
    "tam_tevekenyseg": [
        Band(0, "egyik sem (csak egyéb)", "neither (other only)", values=((EGYEB,),)),
        Band(2, "kutatás-fejlesztés", "research and development", values=((KF,), (EGYEB, KF))),
        Band(3, "infrastrukturális vagy ingatlan beruházás", "infrastructure or real-estate investment",
             values=((INFRA,), (KF, INFRA), (EGYEB, INFRA))),
    ],
    "egysz_elszam": [
        Band(0, "igen (1)", "yes (1)", values=(1,)),
        Band(3, "nem (0)", "no (0)", values=(0,)),
    ],
    "biztositek": [
        Band(0, "igen (1)", "yes (1)", values=(1,)),
        Band(3, "nem (0)", "no (0)", values=(0,)),
    ],
}
INTEGER_FACTORS = ("tam_osszeg", "bead_napok")  # the smallest step between values is 1; otherwise 0.01


def _as_value(factor: str, probe: Any) -> Any:
    return list(probe) if factor == "tam_tevekenyseg" else probe


def band_of(factor: str, value: Any) -> Band | None:
    """The band of the table that covers ``value`` (independent of the scoring module)."""
    probe = tuple(value) if isinstance(value, list) else value
    found = [b for b in BANDS[factor] if b.covers(probe)]
    return found[0] if len(found) == 1 else None


def check_bands() -> None:
    """Every band gives its points at both of its ends in the scoring module (SPEC-L3-13)."""
    problems = []
    for f, bands in BANDS.items():
        step = F(1) if f in INTEGER_FACTORS else F(1, 100)
        for band in bands:
            for probe in band.probes(step):
                got, _ = scoring.points(f, _as_value(f, probe))
                if got != band.points:
                    problems.append(f"{f} {probe!r}: the table says {band.points}, the scoring module {got}")
    for programme, points in scoring.TOP_RULE_POINTS.items():
        if scoring.points("tam_osszeg", None, programme)[0] != points:
            problems.append(f"TOP rule {programme}")
    for value in (None, 90):
        if scoring.points("max_tam_int", value, fin_form="loan")[0] != scoring.LOAN_RULE_POINTS:
            problems.append(f"loan rule, max_tam_int {value!r}")
    if problems:
        raise AppendixError("appendix 8.1 disagrees with the scoring module:\n  " + "\n  ".join(problems))


def scoring_table(data: Data, today: str) -> str:
    """8.1: the factors and their bands, in Hungarian and English (SPEC-E3-05)."""
    check_bands()
    rows = []
    for f in FACTORS:
        hu, en = FACTOR_NAMES[f]
        for i, b in enumerate(BANDS[f]):
            rows.append([f"`{f}`" if i == 0 else "", hu if i == 0 else "", en if i == 0 else "", b.hu, b.en,
                         str(b.points)])
    top = ", ".join(f"{p.replace('_', ' ')} {n}" for p, n in scoring.TOP_RULE_POINTS.items())
    body = [
        f"Scoring rules version: {scoring.RULES_VERSION}. The table is generated from the scoring module and checked "
        "against it at both ends of every band (SPEC-L3-13); 3 points mean the highest risk.",
        "",
        *md_table(["Factor", "Tényező", "Factor (English)", "Sáv", "Band", "Points"], rows),
        "",
        f"**TOP rule (DEC-31):** a TOP or TOP Plusz call that states no maximum amount gets fixed points for "
        f"`tam_osszeg` ({top}); a stated amount is scored by the bands.",
        "",
        f"**Loan rule (DEC-40):** a loan (`fin_form` = loan) gets {scoring.LOAN_RULE_POINTS} points for `max_tam_int`, "
        "whatever the call states, because a loan is paid back in full.",
        "",
        "**Missing values (Spec_ScoringSystem.md §1.3):** a factor that was not determined gets the mean of its points "
        "over the documents where it was determined.",
        "",
        f"**Total and labels:** the sum of the 10 factors (0–{scoring.MAX_TOTAL}), rescaled to 0–100. Fixed label: "
        f"0–{scoring.FIXED_LOW_MAX} low, {scoring.FIXED_LOW_MAX + 1}–{scoring.FIXED_MEDIUM_MAX} medium, "
        f"{scoring.FIXED_MEDIUM_MAX + 1}–100 high, after rounding half up. Tercile label: the documents ranked by "
        "score and cut into thirds; equal scores always get the same label (DEC-15, DEC-32). The tercile label is "
        "the ML target (DEC-07).",
    ]
    return md_document("Appendix 8.1: Factors and scoring table", body, data, ["L3"], today)


# --- 8.2: the database schema -------------------------------------------------------------------


def er_diagram(conn: sqlite3.Connection, data: Data, today: str) -> str:
    """8.2: the schema of the database as Mermaid erDiagram source, read from the database itself."""
    version = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        " AND name != 'schema_migrations' ORDER BY name")]
    lines = ["erDiagram"]
    relations = []
    for t in tables:
        fks = conn.execute(f"PRAGMA foreign_key_list({t})").fetchall()
        fk_columns = {r[3] for r in fks}
        lines.append(f"    {t} {{")
        for col in conn.execute(f"PRAGMA table_info({t})"):
            keys = [k for k, on in (("PK", col[5] > 0), ("FK", col[1] in fk_columns)) if on]
            lines.append(f"        {col[2] or 'ANY'} {col[1]}" + (f" {', '.join(keys)}" if keys else ""))
        lines.append("    }")
        by_id: dict[int, list[str]] = collections.defaultdict(list)
        for r in fks:
            by_id[r[0]].append(r[3])
        for fk_id, cols in sorted(by_id.items()):
            parent = next(r[2] for r in fks if r[0] == fk_id)
            relations.append(f'    {parent} ||--o{{ {t} : "{", ".join(cols)}"')
    lines += sorted(set(relations))
    note = provenance(data, [], today, comment="%% ").replace("no stored run", f"the schema at migration {version}")
    return "\n".join(lines) + "\n" + note


# --- 8.3: reproduction ------------------------------------------------------------------------


def reproduction(data: Data, today: str) -> str:
    """8.3: the commands that produced this chain, in order, with the runs they made."""
    c = data.chain
    info = data.run_info
    steps: list[tuple[str, str, str]] = [("C1", "python -m grantrisk acquire", c.c1)]
    if c.c2:
        steps.append(("C2", f"python -m grantrisk convert --c1-run {c.c1}", c.c2))
    for source, run_id in sorted(c.l1.items()):
        if source == "manual":
            steps.append(("L1 manual", f"python -m grantrisk extract --extractor manual --c1-run {c.c1}", run_id))
        else:
            kind = "llm" if source.startswith("llm:") else source
            steps.append((f"L1 {source}", f"python -m grantrisk extract --extractor {kind} --c2-run {c.c2}", run_id))
    auto = " ".join(f"--{'llm' if s.startswith('llm:') else s}-run {r}" for s, r in sorted(c.l1.items()) if s != "manual")
    doc_set = f"--c2-run {c.c2}" if c.c2 else "--gold-only"
    steps.append(("L2", f"python -m grantrisk consolidate --manual-run {c.l1.get('manual', '?')} {auto} {doc_set}"
                        .replace("  ", " "), c.l2))
    steps.append(("L3", f"python -m grantrisk label --l2-run {c.l2} --c1-run {c.c1}", c.l3))
    if c.m1:
        steps.append(("M1", f"python -m grantrisk represent --c2-run {c.c2}", c.m1))
    def tuned(label: str) -> bool:
        return bool((json.loads(info[label]["config_json"]).get("m2_run") or {}).get("tuning")) if label in info else False

    for label, run_id in (("M2", c.m2), ("M2 beside", c.m2_beside)):
        if run_id:
            flag = "" if tuned(label) else " --no-tuning"
            steps.append((label, f"python -m grantrisk train --m1-run {c.m1} --l3-run {c.l3}{flag}", run_id))
    if c.e1:
        runs_ = ",".join(r for _, r in sorted(c.e1_l1.items()))
        extraction = f" --extraction-runs {runs_}" if runs_ else ""  # the old baseline alone needs none
        steps.append(("E1", f"python -m grantrisk validate --manual-run {c.l1.get('manual', '?')} --l3-run {c.l3} "
                            f"--c1-run {c.c1}{extraction}", c.e1))
    if c.e2:
        steps.append(("E2", f"python -m grantrisk evaluate --m2-run {c.m2}", c.e2))
    if c.e2_beside:
        steps.append(("E2 beside", f"python -m grantrisk evaluate --m2-run {c.m2_beside}", c.e2_beside))
    if c.e4:
        e2_of_e4 = ((data.e4_report or {}).get("input_runs") or {}).get("E2", c.e2)
        steps.append(("E4", f"python -m grantrisk explain --e2-run {e2_of_e4}", c.e4))
    report_args = " ".join(a for a in (f"--e2-run {c.e2}" if c.e2 else f"--l3-run {c.l3}",
                                       f"--e1-run {c.e1}" if c.e1 else "",
                                       f"--e2-beside-run {c.e2_beside}" if c.e2_beside else "",
                                       f"--e4-run {c.e4}" if c.e4 else "") if a)
    rows = [[stage, f"`{cmd}`", f"`{run_id}`", f"`{info[stage]['config_hash'][:12]}`" if stage in info else ""]
            for stage, cmd, run_id in steps]
    body = [
        "The commands that produced the runs of this chain, in order (ARC-07). Each run stores the whole configuration "
        "it used (its hash is shown) and the code version; `README.md` describes the commands and the configuration "
        "file. Each command prints the id of the run it creates, which the next command takes. The runs of stage C1 "
        "read the frozen snapshot of the downloaded files (DEC-23).",
        "",
        *md_table(["Stage", "Command", "Run", "Configuration hash"], rows),
        "",
        f"Finally: `python -m grantrisk report {report_args}` builds this report (E3).",
    ]
    return md_document("Appendix 8.3: Reproduction", body, data, [""], today)


# --- 8.4: external models ---------------------------------------------------------------------


def generative_ai(conn: sqlite3.Connection, data: Data, data_root, today: str) -> str:
    """8.4: every use of an external model in the software, from the run records (DEC-24)."""
    c = data.chain
    llm_runs = sorted({r for s, r in (*c.l1.items(), *c.e1_l1.items()) if s.startswith("llm:")})
    body = ["Every use of an external model by the software in this chain, read from the run records. The author's "
            "own use of AI tools while writing is outside the software and is documented in the thesis.", "",
            "## LLM extraction (L1)", ""]
    if llm_runs:
        rows = []
        for run_id in llm_runs:
            row = conn.execute("SELECT report_path FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            report = json.loads((data_root / row[0]).read_text(encoding="utf-8")) if row and row[0] else {}
            models = [r[0] for r in conn.execute(
                "SELECT DISTINCT source FROM factor_observations WHERE run_id = ? ORDER BY 1", (run_id,))]
            prompts = [r[0] for r in conn.execute(
                "SELECT DISTINCT prompt_version FROM factor_observations WHERE run_id = ? AND prompt_version IS NOT NULL"
                " ORDER BY 1", (run_id,))]
            docs = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM factor_observations WHERE run_id = ?",
                                (run_id,)).fetchone()[0]
            usage = report.get("usage") or {}
            rows.append([f"`{run_id}`", ", ".join(m.removeprefix("llm:") for m in models), ", ".join(prompts) or "–",
                         str(docs), str(usage.get("input_tokens", "–")), str(usage.get("output_tokens", "–")),
                         fmt(report.get("cost_usd"), 2)])
        body += ["Only public calls for proposals of palyazat.gov.hu were sent. The raw responses are stored, so the "
                 "labels can be rebuilt without calling the model again (ARC-06).", "",
                 *md_table(["L1 run", "Model", "Prompt version", "Documents", "Input tokens", "Output tokens",
                            "Cost (USD)"], rows)]
    else:
        body.append("No LLM run is part of this chain.")
    body += ["", "## Hosted embeddings (M1)", ""]
    hosted = []
    if c.m1:
        hosted = conn.execute(
            "SELECT f.representation, fc.model_id, fc.provider, COUNT(DISTINCT f.doc_id) FROM features f"
            " JOIN feature_cache fc ON fc.cache_key = f.cache_key WHERE f.run_id = ? AND fc.location = 'hosted'"
            " GROUP BY 1, 2, 3 ORDER BY 1, 2, 3", (c.m1,)).fetchall()
    if hosted:
        body += ["For the thesis experiments only (DEC-24): the same open weights served by a hosted provider, on "
                 "public documents. The deployable classification path runs locally, for lower cost and independence "
                 "from external APIs.", "",
                 *md_table(["Representation", "Model", "Provider", "Documents"],
                           [[r[0], r[1], r[2], str(r[3])] for r in hosted])]
    else:
        body.append("No embedding of this chain was computed by a hosted service." if c.m1 else
                    "Not available: M1 is not part of this chain.")
    return md_document("Appendix 8.4: Use of generative AI and external models in the software", body, data,
                       ["L1", "M1", "E1"], today)


# --- The data-flow diagram (INT-PIPE-11) ---------------------------------------------------------


def data_flow(data: Data, today: str) -> str:
    """The stage diagram of Spec_Architecture.md §3 as Mermaid source, with the runs of this chain.

    It includes the loop of DEC-54 and DEC-59: E1 labels the gold documents with L3's means and
    cut scores, and suggests the preferred source per factor that L2 uses.
    """
    runs_ = data.chain.runs()

    def node(code: str, name: str) -> str:
        ids = [r for label, r in runs_.items() if label == code or label.startswith(code + " ")]
        suffix = "<br/>" + "<br/>".join(ids) if ids else "<br/>(not run)"
        return f'{code}["{code} {name}{suffix}"]'

    lines = [
        "flowchart TD",
        '    portal[("palyazat.gov.hu")] --> ' + node("C1", "Acquire"),
        "    C1 --> " + node("C2", "Convert"),
        "    C2 --> " + node("L1", "Extract factors"),
        '    gold[("expert gold set")] --> L1',
        "    L1 --> " + node("L2", "Consolidate"),
        "    L2 --> " + node("L3", "Score & label"),
        "    C2 --> " + node("M1", "Represent"),
        "    M1 --> " + node("M2", "Train & predict"),
        "    L3 -- tercile label --> M2",
        "    L1 --> " + node("E1", "Validate extraction"),
        "    L3 -- factor means, tercile cuts --> E1",
        "    E1 -. preferred source per factor .-> L2",
        "    M2 --> " + node("E2", "Evaluate models"),
        "    L3 -- tercile and fixed labels --> E2",
        "    E2 --> " + node("E4", "Explain"),
        "    M1 -- vectors, texts --> E4",
        "    L3 -- points, scores --> E4",
        '    E1 --> E3["E3 Report"]',
        "    E2 --> E3",
        "    E4 --> E3",
    ]
    return "\n".join(lines) + "\n" + provenance(data, list(runs_), today, comment="%% ")
