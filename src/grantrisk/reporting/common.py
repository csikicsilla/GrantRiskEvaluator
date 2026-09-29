"""Formatting shared by the E3 artefacts: numbers, CSV, Markdown tables and the provenance note (SPEC-E3-06)."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Sequence
from typing import Any

from grantrisk.reporting.chain import Data

PERIODS = ("2021-2027", "2014-2020", "RRF", "VP")
LABEL_RANK = {"low": 0, "medium": 1, "high": 2}
MD_DIGITS = 3  # Markdown tables
CSV_DIGITS = 4  # the CSV exports; the stored results keep full precision (SPEC-E3-03)


def fmt(x: float | None, digits: int = MD_DIGITS) -> str:
    """A number for a Markdown table; '–' where it is undefined or was not computed."""
    if x is None:
        return "–"
    if isinstance(x, int) or float(x).is_integer():
        return str(int(x))
    return f"{x:.{digits}f}"


def rounded(x: float | None, digits: int = CSV_DIGITS) -> float | int | None:
    """A number for a CSV export."""
    if x is None:
        return None
    if isinstance(x, int) or float(x).is_integer():
        return int(x)
    return round(float(x), digits)


def hu_number(x: float | int | None, digits: int = CSV_DIGITS) -> str:
    """A number with a decimal comma, for the files opened in a Hungarian Excel (SPEC-E3-02)."""
    r = rounded(x, digits)
    return "" if r is None else str(r).replace(".", ",")


def hu_value(value_json: str | None) -> str:
    """A stored factor value as text: numbers with a decimal comma, lists joined by ', '."""
    if value_json is None:
        return ""
    value = json.loads(value_json)
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int, float)):
        return hu_number(value, 6)
    return str(value)


def csv_text(header: Sequence[str], rows: Iterable[Sequence[Any]], delimiter: str = ",") -> str:
    out = io.StringIO()
    writer = csv.writer(out, delimiter=delimiter, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue()


def md_table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def model_name(key: Sequence[str]) -> str:
    """representation/classifier, with the scheme when it is not the standard one."""
    scheme, representation, classifier = key
    return f"{representation}/{classifier}" + ("" if scheme == "stratified" else f" ({scheme})")


def provenance(data: Data, stages: Sequence[str], today: str, comment: str = "") -> str:
    """SPEC-E3-06: the runs an artefact was built from, the code version and the date.

    ``stages`` are prefixes of the chain labels (e.g. "L1" for every L1 run). ``comment``
    starts every line, for formats whose comments need a marker (e.g. "%% " in Mermaid).
    """
    chain_runs = data.chain.runs()
    named = [f"{label} `{run_id}`" for label, run_id in chain_runs.items() if any(label.startswith(s) for s in stages)]
    lines = [
        "Provenance: built from " + (", ".join(named) if named else "no stored run") + ".",
        f"Code version `{data.code_version}`. Created {today}.",
    ]
    return "\n".join(comment + line for line in lines) + "\n"


def md_document(title: str, body: Sequence[str], data: Data, stages: Sequence[str], today: str) -> str:
    """A Markdown artefact: title, body and the provenance note at the end."""
    return "\n".join([f"# {title}", "", *body, "", "---", "", provenance(data, stages, today)])
