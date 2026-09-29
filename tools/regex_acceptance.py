"""SPEC-L1-06: per-factor gold agreement of the new regex rules against the old regex baseline.

Reads the database read-only and prints a Markdown table. With a C2 run that is still
converting, it compares on the gold documents converted so far; the acceptance test
(tests/test_regex_acceptance.py) needs a complete C2 run with all gold texts.

Usage: PYTHONPATH=src python tools/regex_acceptance.py [C2_RUN_ID]
Default: the newest complete C2 run that holds every gold document.
"""

import sys

from grantrisk import config as config_mod
from grantrisk.corpus.acquire import load_import_config
from grantrisk.extraction.regex import acceptance

cfg = config_mod.load()
pins = load_import_config(cfg.resolve(cfg.values["acquire"]["import_config"]))["gold_pins"]
conn = acceptance.open_read_only(cfg.data_root / "grantrisk.db")
c2 = sys.argv[1] if len(sys.argv) > 1 else acceptance.complete_c2_with(conn, [s[:16] for s in pins.values()])
if c2 is None:
    sys.exit("no complete C2 run holds every gold document; name a C2 run to compare on the texts it has")
validate = cfg.values["validate"]
docs = acceptance.gold_documents(
    conn, c1_run_id=acceptance.c1_of(conn, c2), c2_run_id=c2, gold_csv=cfg.source("gold_csv"), gold_pins=pins,
    old_baseline=cfg.resolve(validate["old_baseline"]), renames=validate.get("old_baseline_renamed"),
)
print(f"C2 run: {c2}\n")
print(acceptance.format_table(acceptance.compare(docs)))
