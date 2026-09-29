"""Command-line entry point: one subcommand per stage, plus ``run-all`` (ARC-07)."""

from __future__ import annotations

import argparse
import sys

from grantrisk import __version__
from grantrisk import config as config_mod

# Subcommand, stage code (Spec_Architecture.md §3), and the chapter that specifies it.
STAGES = [
    ("acquire", "C1", "build the corpus from the snapshot (Spec_C1_Acquire.md)"),
    ("convert", "C2", "convert the PDFs to Markdown (Spec_C2_Convert.md)"),
    ("extract", "L1", "extract the factor values (Spec_L1_ExtractFactors.md)"),
    ("consolidate", "L2", "choose one value per factor (Spec_L2_Consolidate.md)"),
    ("label", "L3", "score and label (Spec_L3_ScoreAndLabel.md)"),
    ("represent", "M1", "compute the text representations (Spec_M1_Represent.md)"),
    ("train", "M2", "cross-validate the classifiers (Spec_M2_TrainPredict.md)"),
    ("validate", "E1", "compare the extractors with the gold set (Spec_E1_ValidateExtraction.md)"),
    ("evaluate", "E2", "compute the model metrics (Spec_E2_EvaluateModels.md)"),
    ("report", "E3", "build the dashboard and the exports (Spec_E3_Report.md)"),
]

EXIT_NOT_IMPLEMENTED = 3


IMPLEMENTED = {"acquire", "extract", "consolidate", "label"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grantrisk",
        description="Grant Risk Estimator: risk labelling and classification of grant calls.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", help="configuration file (default: config/default.yaml)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    stage_parsers = {name: sub.add_parser(name, help=f"{code}: {help_text}") for name, code, help_text in STAGES}
    sub.add_parser("run-all", help="run every stage in order")

    extract = stage_parsers["extract"]
    extract.add_argument("--extractor", required=True, choices=["manual"], help="the extractor to run (regex and llm follow)")
    extract.add_argument("--c1-run", required=True, help="the C1 run that holds the corpus")

    consolidate = stage_parsers["consolidate"]
    consolidate.add_argument("--manual-run", required=True, help="the L1 run of the manual (gold) import")
    consolidate.add_argument("--regex-run", help="the L1 run of the regex extractor")
    consolidate.add_argument("--llm-run", help="the L1 run of the LLM extractor")
    doc_set = consolidate.add_mutually_exclusive_group(required=True)
    doc_set.add_argument("--c2-run", help="the C2 run whose documents with text form the document set")
    doc_set.add_argument("--gold-only", action="store_true", help="the documents of the manual run only (DEC-35)")

    label = stage_parsers["label"]
    label.add_argument("--l2-run", required=True, help="the L2 run whose consolidated factors are labelled")
    label.add_argument("--c1-run", required=True, help="the C1 run that holds the documents' programmes")
    return parser


def _run_stage(name: str, work) -> int:
    """Open the database, run one stage, print its run id; report contract errors without a traceback."""
    from grantrisk.store import db

    def call(cfg: config_mod.Config, args: argparse.Namespace) -> int:
        conn = db.connect(cfg.data_root)
        try:
            run_id = work(conn, cfg, args)
        except ValueError as exc:
            print(f"grantrisk {name}: {exc}", file=sys.stderr)
            return 1
        finally:
            conn.close()
        print(run_id)
        return 0

    return call


def _extract(conn, cfg: config_mod.Config, args: argparse.Namespace) -> str:
    from grantrisk.corpus.acquire import load_import_config
    from grantrisk.extraction.manual import gold

    pins = load_import_config(cfg.resolve(cfg.values["acquire"]["import_config"]))["gold_pins"]
    return gold.run(
        conn, cfg.values, cfg.data_root, c1_run_id=args.c1_run, gold_csv=cfg.source("gold_csv"),
        gold_pins=pins, gold_set=cfg.values["extract"]["manual"]["gold_set"],
    )


def _consolidate(conn, cfg: config_mod.Config, args: argparse.Namespace) -> str:
    from grantrisk.labelling import consolidate

    return consolidate.run(
        conn, cfg.values, cfg.data_root, manual_run_id=args.manual_run, regex_run_id=args.regex_run,
        llm_run_id=args.llm_run, c2_run_id=args.c2_run, gold_only=args.gold_only,
    )


def _acquire(cfg: config_mod.Config, args: argparse.Namespace) -> int:
    from grantrisk.corpus import acquire
    from grantrisk.store import db

    sources = {k: cfg.source(k) for k in ("scraped_pdfs", "scrape_log", "old_database", "manual_zip")}
    conn = db.connect(cfg.data_root)
    try:
        run_id = acquire.run(
            conn, cfg.values, cfg.data_root, sources, cfg.resolve(cfg.values["acquire"]["import_config"])
        )
        n = conn.execute("SELECT COUNT(*) FROM documents WHERE run_id = ?", (run_id,)).fetchone()[0]
    except ValueError as exc:  # includes ImportConfigError and UnknownProgrammeError
        print(f"grantrisk acquire: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(run_id)
    print(f"{n} documents; see reports/{run_id}/c1_report.json for the review list", file=sys.stderr)
    return 0


def _label(cfg: config_mod.Config, args: argparse.Namespace) -> int:
    from grantrisk.labelling import label
    from grantrisk.store import db

    conn = db.connect(cfg.data_root)
    try:
        run_id = label.run(conn, cfg.values, cfg.data_root, l2_run_id=args.l2_run, c1_run_id=args.c1_run)
    except ValueError as exc:  # includes L3InputError
        print(f"grantrisk label: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(run_id)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_mod.load(args.config)
    if args.command == "acquire":
        return _acquire(cfg, args)
    if args.command == "extract":
        return _run_stage("extract", _extract)(cfg, args)
    if args.command == "consolidate":
        return _run_stage("consolidate", _consolidate)(cfg, args)
    if args.command == "label":
        return _label(cfg, args)
    print(f"grantrisk {args.command}: not implemented yet (data root: {cfg.data_root})", file=sys.stderr)
    return EXIT_NOT_IMPLEMENTED


if __name__ == "__main__":
    sys.exit(main())
