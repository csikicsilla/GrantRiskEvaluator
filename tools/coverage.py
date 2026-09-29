"""Generate Spec_Coverage.md: where each INT-* requirement is covered, and how each ISS-* is closed.

Usage: python tools/coverage.py [SPEC_DIR [OUTPUT]]
Defaults: ../10_Specifikáció next to the repository, and Spec_Coverage.md inside it.
"""

import datetime
import glob
import re
import sys
from pathlib import Path

root = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2] / "10_Specifikáció")
out_path = sys.argv[2] if len(sys.argv) > 2 else f"{root}/Spec_Coverage.md"


def read(p):
    return open(f"{root}/{p}", encoding="utf-8").read()


def expand(t):
    """IDs in a text, expanding shorthand such as 'INT-EVAL-07, -08' and 'ISS-10, -11'."""
    ids = set(re.findall(r"(?:INT-[A-Z]+|ISS)-[A-Z0-9]+", t))
    for m in re.finditer(r"((?:INT-[A-Z]+|ISS))-([A-Z0-9]+)((?:,\s*-[A-Z0-9]+)+)", t):
        for x in re.findall(r"-([A-Z0-9]+)", m.group(3)):
            ids.add(f"{m.group(1)}-{x}")
    return ids


# Where each ID is referenced: requirement blocks of the stage chapters, ARC rules, decisions.
where = {}


def add(ids, label):
    for i in ids:
        where.setdefault(i, [])
        if label not in where[i]:
            where[i].append(label)


for f in sorted(glob.glob(f"{root}/Spec_[CLME][0-9]_*.md")):
    t = open(f, encoding="utf-8").read()
    for b in re.split(r"(?m)^(?=\*\*SPEC-)", t)[1:]:
        sid = re.match(r"\*\*(SPEC-[A-Z0-9]+-\d+)", b).group(1)
        body = re.split(r"(?m)^(?:---|## )", b)[0]
        add(expand(body), sid)

arch = read("Spec_Architecture.md")
for row in re.findall(r"(?m)^\| \*\*(ARC-\d+)\*\* \|(.*)$", arch):
    add(expand(row[1]), row[0])

declog = read("Spec_DecisionLog.md")
for b in re.split(r"(?m)^(?=### DEC-)", declog)[1:]:
    did = re.match(r"### (DEC-\d+)", b).group(1)
    tr = re.search(r"(?m)^- \*\*Trace:\*\*(.*)$", b)
    if tr:
        add(expand(tr.group(1)), did)

intent = read("Spec_SoftwareIntent_v7.md")
ints = []
for m in re.finditer(r"(?m)^\s*- \*\*(INT-[A-Z]+-[A-Z0-9]+)[^*]*\*\*:?\s*(.*)$", intent):
    ints.append((m.group(1), m.group(2)))

asis = read("Spec_CurrentPipeline.md")
isss = re.findall(r"(?m)^\| (ISS-\d+) \| ([HML]) \| \*\*(.+?)\*\*", asis)

notes = {
    "INT-GOAL-01": "System level: the architecture as a whole.",
    "INT-SCOPE-06": "System level: the architecture as a whole.",
    "ISS-15": "Waived for the software: the collateral rule belongs to the scoring system, which is authoritative. E3 shows its effect (SPEC-E3-01), and the thesis discusses it.",
    "ISS-17": "Waived for the software: whose risk the factors measure is a design question of the scoring system, for the thesis to discuss.",
    "ISS-18": "Closed by DEC-05: `kifiz_kerelem` is not a factor.",
    "ISS-45": "Closed by DEC-01: no code of the old line is carried over.",
    "INT-RQ-A": "Answered through its sub-questions INT-RQ-A1 and INT-RQ-A2, and INT-GOAL-05 (see those rows).",
    "INT-PIPE-01": "Architecture §3: the stage table maps every stage to one of the six steps.",
    "ISS-46": "Closed by DEC-04 and DEC-27: the new code line has no version increments; results are identified by run ids.",
}


def order(i):
    return (not i.startswith("SPEC"), not i.startswith("ARC"), i)


def cell(i):
    refs = ", ".join(sorted(where.get(i, []), key=order))
    note = notes.get(i, "")
    return "; ".join(x for x in (refs, note) if x) or "**not covered**"


lines = [
    "# Coverage of the specification",
    "",
    f"**Status:** Generated on {datetime.date.today().isoformat()} by `40_GrantRiskEstimator2/tools/coverage.py` from the stage chapters, the architecture and the decision log. Regenerate it after a chapter changes.",
    "",
    "**What this document is:** It shows where each intent requirement (`INT-*`, [intent v7](Spec_SoftwareIntent_v7.md)) is covered, and how each known issue of the old code line (`ISS-*`, [as-is spec](Spec_CurrentPipeline.md) §9) is closed or waived.",
    "",
    "## 1. Intent requirements",
    "",
    "| ID | Covered by |",
    "|---|---|",
]
for i, _ in ints:
    lines.append(f"| {i} | {cell(i)} |")
lines += ["", "## 2. Known issues of the old code line", "", "| ID | Sev | Issue | Closed by / waived |", "|---|---|---|---|"]
for i, sev, title in isss:
    lines.append(f"| {i} | {sev} | {title} | {cell(i)} |")
lines.append("")
open(out_path, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
missing = [i for i, _ in ints if cell(i) == "**not covered**"] + [i for i, _, _ in isss if cell(i) == "**not covered**"]
print("INT:", len(ints), "ISS:", len(isss), "not covered:", missing)
