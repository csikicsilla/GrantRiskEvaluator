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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grantrisk",
        description="Grant Risk Estimator: risk labelling and classification of grant calls.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", help="configuration file (default: config/default.yaml)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, code, help_text in STAGES:
        sub.add_parser(name, help=f"{code}: {help_text}")
    sub.add_parser("run-all", help="run every stage in order")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_mod.load(args.config)
    print(f"grantrisk {args.command}: not implemented yet (data root: {cfg.data_root})", file=sys.stderr)
    return EXIT_NOT_IMPLEMENTED


if __name__ == "__main__":
    sys.exit(main())
