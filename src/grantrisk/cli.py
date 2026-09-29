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


IMPLEMENTED = {"acquire", "convert", "extract", "consolidate", "label"}


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

    convert = stage_parsers["convert"]
    convert.add_argument("--c1-run", required=True, help="the C1 run that holds the corpus")
    convert.add_argument("--compare", action="store_true", help="compare the converters on the sample (SPEC-C2-11)")
    convert.add_argument(
        "--converters", default="pymupdf4llm-legacy,pymupdf4llm,docling", help="converters to compare, comma-separated"
    )
    convert.add_argument("--out", help="an existing comparison folder to add to or resume (default: a new one)")
    convert.add_argument("--resume", help="an unfinished C2 run to continue")
    convert.add_argument("--workers", type=int, help="parallel worker processes (default: convert.workers)")

    extract = stage_parsers["extract"]
    extract.add_argument("--extractor", required=True, choices=["manual", "llm"], help="the extractor to run")
    extract.add_argument("--c1-run", help="manual: the C1 run that holds the corpus")
    extract.add_argument("--c2-run", help="llm: the C2 run whose texts are read")
    extract.add_argument("--model", help="llm: a model other than extract.llm.model")
    extract.add_argument("--gold-only", action="store_true", help="llm: only the gold documents (SPEC-L1-14)")
    extract.add_argument("--estimate", action="store_true", help="llm: print the cost estimate and stop (SPEC-L1-11)")
    extract.add_argument("--confirm", action="store_true", help="llm: confirm a run over the whole corpus")
    extract.add_argument("--resume", help="llm: an unfinished L1 run to continue")
    extract.add_argument("--reparse", action="store_true", help="llm: parse stored responses only, without the API")

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
        if run_id:
            print(run_id)
        return 0

    return call


def _compare_converters(cfg: config_mod.Config, args: argparse.Namespace) -> int:
    """SPEC-C2-11: convert the configured sample with each converter, for DEC-30."""
    import datetime
    import zipfile

    from grantrisk.corpus import converters, convert
    from grantrisk.store import db

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = cfg.resolve(args.out) if args.out else cfg.data_root / "reports" / "c2_comparison" / stamp
    conn = db.connect(cfg.data_root)
    try:
        docs = {
            r["doc_id"]: r
            for r in conn.execute("SELECT doc_id, file_path, page_count FROM documents WHERE run_id = ?", (args.c1_run,))
        }
    finally:
        conn.close()
    sample = []
    for item in cfg.values["convert"]["comparison_sample"]:
        if "doc_id" in item:
            d = docs[item["doc_id"]]
            sample.append(convert.SampleDocument(item["label"], item["name"], cfg.data_root / d["file_path"], d["page_count"]))
        else:  # a manual file from the archive that is not part of the corpus
            target = out_dir / "input" / item["manual_file"]
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(cfg.source("manual_zip")) as z:
                    target.write_bytes(z.read(f"data/raw_pdfs/{item['manual_file']}"))
            sample.append(convert.SampleDocument(item["label"], item["name"], target, None))
    adapters = [converters.create(name) for name in args.converters.split(",")]
    convert.compare(adapters, sample, out_dir)
    print(out_dir)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


def _convert(conn, cfg: config_mod.Config, args: argparse.Namespace) -> str:
    from grantrisk.corpus import convert

    return convert.run(
        conn, cfg.values, cfg.data_root, c1_run_id=args.c1_run, resume_run_id=args.resume, workers=args.workers,
        progress=lambda line: print(line, file=sys.stderr, flush=True),
    )


def _extract(conn, cfg: config_mod.Config, args: argparse.Namespace) -> str:
    from grantrisk.corpus.acquire import load_import_config
    from grantrisk.extraction.manual import gold

    pins = load_import_config(cfg.resolve(cfg.values["acquire"]["import_config"]))["gold_pins"]
    if args.extractor == "llm":
        return _extract_llm(conn, cfg, args, pins)
    if not args.c1_run:
        raise ValueError("the manual extractor needs --c1-run")
    return gold.run(
        conn, cfg.values, cfg.data_root, c1_run_id=args.c1_run, gold_csv=cfg.source("gold_csv"),
        gold_pins=pins, gold_set=cfg.values["extract"]["manual"]["gold_set"],
    )


def _extract_llm(conn, cfg: config_mod.Config, args: argparse.Namespace, pins: dict[str, str]) -> str:
    import json

    from grantrisk.extraction.llm import run as llm

    if not args.c2_run:
        raise ValueError("the LLM extractor needs --c2-run")
    settings = llm.settings_from_config(cfg.values, cfg.resolve, args.model)
    doc_ids = sorted(sha[:16] for sha in pins.values()) if args.gold_only else None
    client = None
    if not args.reparse:
        import anthropic  # the API key comes from the environment, never from the configuration (SPEC-L1-13)

        client = anthropic.Anthropic(max_retries=cfg.values["extract"]["llm"].get("max_retries", 5))
    if args.estimate:
        estimate = llm.estimate(client, settings, llm.documents(conn, args.c2_run, doc_ids))
        print(json.dumps(estimate, indent=2))
        return None  # an estimate is not a run
    return llm.run(
        conn, cfg.values, cfg.data_root, client, settings, c2_run_id=args.c2_run, doc_ids=doc_ids,
        confirmed=args.confirm, resume_run_id=args.resume, progress=lambda line: print(line, file=sys.stderr, flush=True),
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
    if args.command == "convert":
        return _compare_converters(cfg, args) if args.compare else _run_stage("convert", _convert)(cfg, args)
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
