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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grantrisk",
        description="Grant Risk Estimator: risk labelling and classification of grant calls.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", help="configuration file (default: config/default.yaml)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    stage_parsers = {name: sub.add_parser(name, help=f"{code}: {help_text}") for name, code, help_text in STAGES}
    run_all = sub.add_parser("run-all", help="run every stage in order")
    run_all.add_argument("--c1-run", help="reuse this C1 run instead of acquiring the corpus again")
    run_all.add_argument("--c2-run", help="reuse this C2 run instead of converting again (it takes hours)")
    run_all.add_argument("--llm-run", help="include this L1 LLM run; run-all never calls the paid API itself")
    run_all.add_argument("--representations", help="M1 and M2: comma-separated (default: represent.representations)")
    run_all.add_argument("--classifiers", help="M2: comma-separated (default: train.classifiers)")
    run_all.add_argument("--confirm", action="store_true", help="M1: confirm sending texts to a hosted provider")
    run_all.add_argument("--no-baseline", action="store_true", help="E1: leave out the old regex baseline")

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
    extract.add_argument("--extractor", required=True, choices=["manual", "regex", "llm"], help="the extractor to run")
    extract.add_argument("--c1-run", help="manual: the C1 run that holds the corpus")
    extract.add_argument("--c2-run", help="regex, llm: the C2 run whose texts are read")
    extract.add_argument("--model", help="llm: a model other than extract.llm.model")
    extract.add_argument("--gold-only", action="store_true", help="regex, llm: only the gold documents (SPEC-L1-14)")
    extract.add_argument("--estimate", action="store_true", help="llm: print the cost estimate and stop (SPEC-L1-11)")
    extract.add_argument("--confirm", action="store_true", help="llm: confirm a run over the whole corpus")
    extract.add_argument("--resume", help="llm: an unfinished L1 run to continue")
    extract.add_argument("--reparse", action="store_true", help="llm: parse stored responses only, without the API")
    extract.add_argument("--batch", action="store_true",
                         help="llm: send the requests through the Message Batches API at half the price (DEC-65)")
    extract.add_argument("--budget-usd", type=float,
                         help="llm: the ceiling of this run in USD (default: extract.llm.budget_usd, SPEC-L1-11)")

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

    represent = stage_parsers["represent"]
    represent.add_argument("--c2-run", required=True, help="the C2 run whose texts are represented")
    represent.add_argument("--representations", help="comma-separated (default: represent.representations)")
    represent.add_argument("--estimate", action="store_true", help="print what would be sent and its cost, and stop (SPEC-M1-04)")
    represent.add_argument("--confirm", action="store_true", help="confirm a run that sends texts to a hosted provider")
    represent.add_argument("--resume", help="an unfinished M1 run to continue")

    train = stage_parsers["train"]
    train.add_argument("--m1-run", required=True, help="the M1 run that holds the representations")
    train.add_argument("--l3-run", required=True, help="the L3 run whose tercile labels are the target")
    train.add_argument("--representations", help="comma-separated (default: train.representations)")
    train.add_argument("--classifiers", help="comma-separated (default: train.classifiers)")

    validate = stage_parsers["validate"]
    validate.add_argument("--manual-run", required=True, help="the L1 run of the manual (gold) import")
    validate.add_argument("--l3-run", required=True, help="the L3 run whose tercile cuts label the gold documents")
    validate.add_argument("--c1-run", required=True, help="the C1 run that holds the documents' programmes")
    validate.add_argument("--extraction-runs", default="", help="L1 runs of the automated extractors, comma-separated")
    validate.add_argument("--no-baseline", action="store_true", help="leave out the old regex baseline (sources.old_baseline)")

    evaluate = stage_parsers["evaluate"]
    evaluate.add_argument("--m2-run", required=True, help="the M2 run whose predictions are evaluated")

    report = stage_parsers["report"]
    report.add_argument("--e2-run", help="the E2 run of the chain (default: report.chain.e2_run)")
    report.add_argument("--e1-run", help="the E1 run of the chain (default: report.chain.e1_run)")
    report.add_argument("--l3-run", help="alone: a report without models (default: report.chain.l3_run)")
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


def _gold_doc_ids(pins: dict[str, str], gold_only: bool) -> list[str] | None:
    """The doc_ids of the pinned gold documents (DEC-34), or None for the whole run."""
    return sorted(sha[:16] for sha in pins.values()) if gold_only else None


def _extract(conn, cfg: config_mod.Config, args: argparse.Namespace) -> str:
    from grantrisk.corpus.acquire import load_import_config
    from grantrisk.extraction.manual import gold

    pins = load_import_config(cfg.resolve(cfg.values["acquire"]["import_config"]))["gold_pins"]
    if args.extractor == "llm":
        return _extract_llm(conn, cfg, args, pins)
    if args.extractor == "regex":
        from grantrisk.extraction.regex import run as regex

        return regex.run(
            conn, cfg.values, cfg.data_root, c2_run_id=args.c2_run, doc_ids=_gold_doc_ids(pins, args.gold_only),
            progress=lambda line: print(line, file=sys.stderr, flush=True),
        )
    return gold.run(
        conn, cfg.values, cfg.data_root, c1_run_id=args.c1_run, gold_csv=cfg.source("gold_csv"),
        gold_pins=pins, gold_set=cfg.values["extract"]["manual"]["gold_set"],
    )


def _extract_llm(conn, cfg: config_mod.Config, args: argparse.Namespace, pins: dict[str, str]) -> str:
    import json

    from grantrisk.extraction.llm import run as llm

    settings = llm.settings_from_config(cfg.values, cfg.resolve, args.model)
    if args.budget_usd is not None:
        import dataclasses

        settings = dataclasses.replace(settings, budget_usd=args.budget_usd)
    doc_ids = _gold_doc_ids(pins, args.gold_only)
    client = None
    if not args.reparse:
        import anthropic  # the API key comes from the environment, never from the configuration (SPEC-L1-13)

        client = anthropic.Anthropic(max_retries=cfg.values["extract"]["llm"].get("max_retries", 5))
    if args.estimate:
        estimate = llm.estimate(client, settings, llm.documents(conn, args.c2_run, doc_ids))
        if estimate["usd_estimate"] is not None:
            estimate["usd_estimate_batch"] = estimate["usd_estimate"] / 2  # DEC-65
        print(json.dumps(estimate, indent=2))
        return None  # an estimate is not a run
    if args.batch:
        from grantrisk.extraction.llm import batch

        return batch.run_batch(
            conn, cfg.values, cfg.data_root, client, settings, c2_run_id=args.c2_run, doc_ids=doc_ids,
            confirmed=args.confirm, resume_run_id=args.resume,
            progress=lambda line: print(line, file=sys.stderr, flush=True),
        )
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


def _names(value: str | None) -> list[str] | None:
    """A comma-separated option, or None to keep the configured default."""
    return [v for v in value.split(",") if v] if value else None


def _represent(conn, cfg: config_mod.Config, args: argparse.Namespace) -> str:
    import json

    from grantrisk.modelling import represent

    reps = _names(args.representations)
    if args.estimate:
        estimate = represent.estimate(conn, cfg.values, cfg.data_root, c2_run_id=args.c2_run, representations=reps)
        print(json.dumps(estimate, indent=2, default=str))
        return None  # an estimate is not a run
    return represent.run(
        conn, cfg.values, cfg.data_root, c2_run_id=args.c2_run, representations=reps, confirmed=args.confirm,
        resume_run_id=args.resume, progress=lambda line: print(line, file=sys.stderr, flush=True),
    )


def _train(conn, cfg: config_mod.Config, args: argparse.Namespace) -> str:
    from grantrisk.modelling import train

    return train.run(
        conn, cfg.values, cfg.data_root, m1_run_id=args.m1_run, l3_run_id=args.l3_run,
        representations=_names(args.representations), classifiers=_names(args.classifiers),
        progress=lambda line: print(line, file=sys.stderr, flush=True),
    )


def _evaluate(conn, cfg: config_mod.Config, args: argparse.Namespace) -> str:
    from grantrisk.evaluation import evaluate

    return evaluate.run(conn, cfg.values, cfg.data_root, m2_run_id=args.m2_run)


def _report(conn, cfg: config_mod.Config, args: argparse.Namespace) -> str:
    from grantrisk.reporting import report

    run_id = report.run(conn, cfg.values, cfg.data_root, e2_run_id=args.e2_run, e1_run_id=args.e1_run,
                        l3_run_id=args.l3_run)
    print(f"dashboard: {cfg.data_root / 'reports' / run_id / 'dashboard.html'}", file=sys.stderr)
    return run_id


def _validate(conn, cfg: config_mod.Config, args: argparse.Namespace) -> str:
    from grantrisk.evaluation import validate

    baseline = None if args.no_baseline else cfg.source("old_baseline")
    return validate.run(
        conn, cfg.values, cfg.data_root, manual_run_id=args.manual_run,
        extraction_run_ids=_names(args.extraction_runs) or [], l3_run_id=args.l3_run,
        c1_run_id=args.c1_run, old_baseline_csv=baseline,
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


def _with_preferences(values: dict, preferences: dict[str, str], e1_run_id: str | None) -> dict:
    """The configuration with the L2 preferences set, so the L2 run records the ones it used."""
    consolidate = {**(values.get("consolidate") or {}), "preferred_source": preferences,
                   "preferences_from_e1_run": e1_run_id}
    return {**values, "consolidate": consolidate}


def _run_all(cfg: config_mod.Config, args: argparse.Namespace) -> int:
    """ARC-07: every stage in order, each a new run that reads the runs before it.

    C1 and C2 can be reused by id, because C2 takes hours. The LLM extractor is never
    started here: it is paid and needs its own confirmation (SPEC-L1-11), so an existing
    run is included with --llm-run. The L2 preferences (SPEC-L2-03) come from the
    configuration when it names them. Otherwise, without an LLM run, the regex is the only
    automated source and is named for every factor, which has no other effect. With an
    LLM run, a first E1 run on the gold-only chain (DEC-35) suggests them (SPEC-E1-06).
    Each finished stage prints its code and run id on stdout.
    """
    import yaml

    from grantrisk.corpus import acquire, convert
    from grantrisk.corpus.acquire import load_import_config
    from grantrisk.evaluation import evaluate, validate
    from grantrisk.extraction.manual import gold
    from grantrisk.extraction.regex import run as regex
    from grantrisk.labelling import consolidate, label
    from grantrisk.labelling.scoring import FACTORS
    from grantrisk.modelling import represent, train
    from grantrisk.reporting import report
    from grantrisk.store import db, runs

    root, values = cfg.data_root, cfg.values
    progress = lambda line: print(line, file=sys.stderr, flush=True)  # noqa: E731
    current = ""

    def step(code: str, work) -> str:
        nonlocal current
        current = code
        run_id = work()
        print(f"{code} {run_id}", flush=True)
        return run_id

    def reuse(code: str, run_id: str, stage: str | None = None) -> str:
        nonlocal current
        current = code
        runs.require_complete(conn, run_id, stage or code)
        print(f"{code} {run_id} (reused)", flush=True)
        return run_id

    conn = db.connect(root)
    try:
        import_config = cfg.resolve(values["acquire"]["import_config"])
        if args.c1_run:
            c1 = reuse("C1", args.c1_run)
        else:
            sources = {k: cfg.source(k) for k in ("scraped_pdfs", "scrape_log", "old_database", "manual_zip")}
            c1 = step("C1", lambda: acquire.run(conn, values, root, sources, import_config))
        c2 = reuse("C2", args.c2_run) if args.c2_run else step(
            "C2", lambda: convert.run(conn, values, root, c1_run_id=c1, progress=progress))
        pins = load_import_config(import_config)["gold_pins"]
        manual = step("L1 manual", lambda: gold.run(
            conn, values, root, c1_run_id=c1, gold_csv=cfg.source("gold_csv"), gold_pins=pins,
            gold_set=values["extract"]["manual"]["gold_set"]))
        regex_run = step("L1 regex", lambda: regex.run(conn, values, root, c2_run_id=c2, progress=progress))
        llm_run = reuse("L1 llm", args.llm_run, "L1") if args.llm_run else None
        extraction_runs = [regex_run, *([llm_run] if llm_run else [])]
        baseline = None if args.no_baseline else cfg.source("old_baseline")

        labelling_values = values
        if not (values.get("consolidate") or {}).get("preferred_source"):
            if llm_run is None:
                labelling_values = _with_preferences(values, {f: "regex" for f in FACTORS}, None)
            else:
                gold_l2 = step("L2 gold-only", lambda: consolidate.run(
                    conn, values, root, manual_run_id=manual, gold_only=True))
                gold_l3 = step("L3 gold-only", lambda: label.run(conn, values, root, l2_run_id=gold_l2, c1_run_id=c1))
                e1_first = step("E1 preferences", lambda: validate.run(
                    conn, values, root, manual_run_id=manual, extraction_run_ids=extraction_runs,
                    l3_run_id=gold_l3, c1_run_id=c1, old_baseline_csv=baseline))
                suggestion = root / "reports" / e1_first / "l2_preferences.yaml"
                if not suggestion.exists():
                    raise ValueError(f"E1 run {e1_first} suggests no L2 preferences; see its report")
                preferences = yaml.safe_load(suggestion.read_text(encoding="utf-8"))["consolidate"]["preferred_source"]
                labelling_values = _with_preferences(values, preferences, e1_first)

        l2 = step("L2", lambda: consolidate.run(
            conn, labelling_values, root, manual_run_id=manual, regex_run_id=regex_run, llm_run_id=llm_run,
            c2_run_id=c2))
        l3 = step("L3", lambda: label.run(conn, values, root, l2_run_id=l2, c1_run_id=c1))
        e1 = step("E1", lambda: validate.run(
            conn, values, root, manual_run_id=manual, extraction_run_ids=extraction_runs, l3_run_id=l3,
            c1_run_id=c1, old_baseline_csv=baseline))
        reps = _names(args.representations)
        m1 = step("M1", lambda: represent.run(
            conn, values, root, c2_run_id=c2, representations=reps, confirmed=args.confirm, progress=progress))
        m2 = step("M2", lambda: train.run(
            conn, values, root, m1_run_id=m1, l3_run_id=l3, representations=reps,
            classifiers=_names(args.classifiers), progress=progress))
        e2 = step("E2", lambda: evaluate.run(conn, values, root, m2_run_id=m2))
        e3 = step("E3", lambda: report.run(conn, values, root, e2_run_id=e2, e1_run_id=e1))
        print(f"dashboard: {root / 'reports' / e3 / 'dashboard.html'}", file=sys.stderr)
    except (ValueError, RuntimeError) as exc:  # RuntimeError: M2, when every model run failed
        print(f"grantrisk run-all: {current}: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


EXTRACTOR_INPUT = {"manual": "c1_run", "regex": "c2_run", "llm": "c2_run"}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "extract" and not getattr(args, EXTRACTOR_INPUT[args.extractor]):
        option = "--" + EXTRACTOR_INPUT[args.extractor].replace("_", "-")
        parser.error(f"extract: the {args.extractor} extractor needs {option}")
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
    if args.command == "represent":
        return _run_stage("represent", _represent)(cfg, args)
    if args.command == "train":
        return _run_stage("train", _train)(cfg, args)
    if args.command == "validate":
        return _run_stage("validate", _validate)(cfg, args)
    if args.command == "evaluate":
        return _run_stage("evaluate", _evaluate)(cfg, args)
    if args.command == "report":
        return _run_stage("report", _report)(cfg, args)
    assert args.command == "run-all", args.command  # argparse accepts no other command
    return _run_all(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
