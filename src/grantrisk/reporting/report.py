"""Stage E3: report (Spec_E3_Report.md).

Builds, from the stored results of one chain of runs, the dashboard (SPEC-E3-01), the
per-document risk dataset (-02), the thesis tables (-03), the error analysis (-04) and
the appendix material (-05). Every artefact names the runs and the code version it was
built from (-06); two builds from the same runs differ only in the creation date. With
an E2 run beside and an E4 run (DEC-70), their tables are part of the report too.
"""

from __future__ import annotations

import datetime
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from grantrisk.reporting import appendix, chain as chain_mod, dashboard, explain_tables, tables
from grantrisk.reporting.chain import E3InputError
from grantrisk.reporting.common import provenance
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

__all__ = ["run", "E3InputError"]


def _provenance_index(data: chain_mod.Data, written: Mapping[str, list[str]], today: str) -> str:
    """PROVENANCE.md: the runs behind each file, for the CSV files, which carry no note of their own."""
    lines = ["# Provenance of this report", "",
             "Every Markdown, Mermaid and HTML file ends with its own provenance note. The CSV files stay plain "
             "tables, so that they open in a spreadsheet without extra rows; their sources are listed here.", ""]
    for name, stages in sorted(written.items()):
        named = [f"{label} `{run_id}`" for label, run_id in data.chain.runs().items()
                 if any(label.startswith(s) for s in stages)]
        lines.append(f"- `{name}`: " + (", ".join(named) or "the database schema and the scoring module"))
    lines += ["", "Parts of the chain that were not run: " + (", ".join(data.chain.missing()) or "none") + ".",
              "", "---", "", provenance(data, list(data.chain.runs()), today)]
    return "\n".join(lines)


def build(conn: sqlite3.Connection, data_root: Path, data: chain_mod.Data, today: str) -> dict[str, tuple[str, str]]:
    """Every artefact: relative name → (text, encoding). The pure part of the stage."""
    out: dict[str, tuple[str, str]] = {}
    sources: dict[str, list[str]] = {}

    def add(name: str, text: str | None, stages: list[str], encoding: str = "utf-8") -> None:
        if text is not None:
            out[name] = (text, encoding)
            sources[name] = stages

    add("dashboard.html", dashboard.build(data, today), list(data.chain.runs()))
    # SPEC-E3-02: ';' and UTF-8 with BOM, like the gold file, for a Hungarian Excel.
    add("risk_dataset.csv", tables.risk_dataset(data), ["C1", "L2", "L3", "M2", "E2"], "utf-8-sig")
    for name, fn in tables.THESIS_TABLES.items():
        csv, md = fn(data, today)
        add(f"tables/{name}.csv", csv, ["E2", "M2", "L3", "E1", "C1"])
        add(f"tables/{name}.md", md, [])
    for name, fn in explain_tables.EXPLAIN_TABLES.items():
        csv, md = fn(data, today)
        add(f"tables/{name}.csv", csv, explain_tables.STAGES)
        add(f"tables/{name}.md", md, [])
    for name, text in tables.error_analysis(data, today).items():
        add(f"error_analysis/{name}", text, ["E2", "M2", "E1", "L3", "L2", "L1", "C1"],
            "utf-8-sig" if name == "misclassified.csv" else "utf-8")
    add("appendix/8_1_scoring_table.md", appendix.scoring_table(data, today), [])
    add("appendix/8_2_er_diagram.mmd", appendix.er_diagram(conn, data, today), [])
    add("appendix/8_3_reproduction.md", appendix.reproduction(data, today), [])
    add("appendix/8_4_generative_ai.md", appendix.generative_ai(conn, data, data_root, today), [])
    add("appendix/data_flow.mmd", appendix.data_flow(data, today), [])
    csv_sources = {n: s for n, s in sources.items() if n.endswith(".csv")}
    add("PROVENANCE.md", _provenance_index(data, csv_sources, today), [])
    return out


def run(
    conn: sqlite3.Connection,
    config_values: Mapping[str, Any],
    data_root: Path,
    *,
    e2_run_id: str | None = None,
    e1_run_id: str | None = None,
    l3_run_id: str | None = None,
    e2_beside_run_id: str | None = None,
    e4_run_id: str | None = None,
    today: str | None = None,
) -> str:
    """Build the report of one chain. Returns the new run id.

    The chain is named by its E2 run (or, for a report without models, its L3 run) and
    optionally its E1 run, an E2 run beside it and its E4 run (DEC-70); runs not given
    here come from ``report.chain`` in the configuration. ``today`` is the creation date written into the artefacts (default:
    the current UTC date). On any error the run is marked failed and no file remains.
    """
    configured = (config_values.get("report") or {}).get("chain") or {}
    e2_run_id = e2_run_id or configured.get("e2_run")
    e1_run_id = e1_run_id or configured.get("e1_run")
    l3_run_id = l3_run_id or configured.get("l3_run")
    e2_beside_run_id = e2_beside_run_id or configured.get("e2_beside_run")
    e4_run_id = e4_run_id or configured.get("e4_run")
    chain = chain_mod.resolve(conn, e2_run_id=e2_run_id, e1_run_id=e1_run_id, l3_run_id=l3_run_id,
                              e2_beside_run_id=e2_beside_run_id, e4_run_id=e4_run_id)
    today = today or datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    run_id = runs.start(conn, "E3", config_values, inputs=sorted(set(chain.runs().values())))
    written: list[str] = []
    try:
        data = chain_mod.load(conn, data_root, chain)
        artefacts = build(conn, data_root, data, today)
        folder = f"reports/{run_id}"
        for name, (text, encoding) in sorted(artefacts.items()):
            written.append(files.write_text(data_root, f"{folder}/{name}", text, encoding))
        report = {
            "input_runs": chain.runs(),
            "missing": chain.missing(),
            "best_model_run": "/".join(data.best) if data.best else None,
            "tied_best_model_runs": ["/".join(k) for k in data.tied_best],
            "documents": len(data.labels),
            "created": today,
            "files": sorted(artefacts),
        }
        report_path = files.write_json(data_root, f"{folder}/e3_report.json", report)
        written.append(report_path)
        with transaction(conn):
            runs.complete(conn, run_id, report_path)
    except Exception as exc:
        for path in written:
            files.remove(data_root, path)
        runs.fail(conn, run_id, f"{type(exc).__name__}: {exc}")
        raise
    return run_id
