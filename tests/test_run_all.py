"""run-all (ARC-07): the whole chain through the CLI, on the six-call gold set of the E1 tests.

C1 and C2 are seeded and reused by id; every later stage runs for real.
"""

import json

import pytest
import yaml
from e1_helpers import GOLD, doc_id, identical_observations, seed_c1, seed_l1, sha, write_gold_csv
from modelling_helpers import seed_c2

from grantrisk import cli
from grantrisk import config as config_mod
from grantrisk.store import db, runs

# Terciles of the six calls (see e1_helpers): low {B, F}, medium {C, D}, high {A, E}; one word per class.
CLASS_WORD = {"A": "beruhazas", "E": "beruhazas", "C": "kepzes", "D": "kepzes", "B": "tanacsadas", "F": "tanacsadas"}
FILLER = "a felhívás célja a támogatás nyújtása pályázók számára"


def text(name):
    word = CLASS_WORD[name[0]]
    return f"<!-- page 1 -->\n# Felhívás {name}\n\n{FILLER} {word} {word} {FILLER}\n"


@pytest.fixture
def config_path(tmp_path):
    values = config_mod.load().values
    values["data_root"] = str(tmp_path / "data")
    values["sources"]["gold_csv"] = str(write_gold_csv(tmp_path / "gold.csv"))
    import_config = tmp_path / "import.yaml"
    import_config.write_text(yaml.safe_dump({"gold_pins": {n: sha(n) for n, *_ in GOLD}}), encoding="utf-8")
    values["acquire"]["import_config"] = str(import_config)
    values["consolidate"]["preferred_source"] = {}
    values["train"]["cv"] = {"n_splits": 2, "n_repeats": 1, "random_state": 42}
    values["represent"]["tfidf"]["min_df"] = 1  # three training documents per fold
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def conn(config_path):
    c = db.connect(config_mod.load(config_path).data_root)
    yield c
    c.close()


@pytest.fixture
def seeded(conn):
    return seed_c1(conn), seed_c2(conn, {doc_id(n): text(n) for n, *_ in GOLD})


def run_all(config_path, capsys, *argv):
    code = cli.main(["--config", str(config_path), "run-all", "--representations", "tfidf",
                     "--classifiers", "logreg,majority", "--no-baseline", *argv])
    out = capsys.readouterr()
    return code, dict(line.split(" (")[0].rsplit(" ", 1) for line in out.out.strip().splitlines()), out.err


def l2_report(config_path, conn, l2):
    return json.loads((config_mod.load(config_path).data_root / runs.get(conn, l2)["report_path"]).read_text("utf-8"))


def test_the_chain_runs_every_stage(config_path, conn, seeded, capsys):
    c1, c2 = seeded
    code, done, err = run_all(config_path, capsys, "--c1-run", c1, "--c2-run", c2)
    assert code == 0, err
    assert list(done) == ["C1", "C2", "L1 manual", "L1 regex", "L2", "L3", "E1", "M1", "M2", "E2", "E3"]
    assert (done["C1"], done["C2"]) == (c1, c2)
    for code_, run_id in done.items():
        assert runs.get(conn, run_id)["status"] == "complete", code_
    # Lineage: each stage reads the runs before it.
    assert set(runs.inputs(conn, done["L2"])) == {done["L1 manual"], done["L1 regex"], c2}
    assert set(runs.inputs(conn, done["M2"])) == {done["M1"], done["L3"]}
    assert set(runs.inputs(conn, done["E3"])) >= {done["E2"], done["E1"]}
    dashboard = config_mod.load(config_path).data_root / "reports" / done["E3"] / "dashboard.html"
    assert "http://" not in dashboard.read_text(encoding="utf-8").split("<body")[0]  # self-contained
    # Without an LLM run the regex is the only automated source, and no E1 run is named.
    report = l2_report(config_path, conn, done["L2"])
    assert set(report["preferred_source"].values()) == {"regex"}
    assert report["preferences_from_e1_run"] is None


def test_an_llm_run_takes_its_preferences_from_a_first_e1_run(config_path, conn, seeded, capsys):
    c1, c2 = seeded
    model = config_mod.load(config_path).values["extract"]["llm"]["model"]
    llm = seed_l1(conn, f"llm:{model}", identical_observations())
    code, done, err = run_all(config_path, capsys, "--c1-run", c1, "--c2-run", c2, "--llm-run", llm)
    assert code == 0, err
    assert list(done) == ["C1", "C2", "L1 manual", "L1 regex", "L1 llm", "L2 gold-only", "L3 gold-only",
                          "E1 preferences", "L2", "L3", "E1", "M1", "M2", "E2", "E3"]
    report = l2_report(config_path, conn, done["L2"])
    assert report["preferences_from_e1_run"] == done["E1 preferences"]
    assert set(runs.inputs(conn, done["E1"])) >= {done["L1 regex"], llm, done["L3"]}


def test_a_stage_error_names_the_stage(config_path, conn, seeded, capsys):
    c1, _ = seeded
    code, done, err = run_all(config_path, capsys, "--c1-run", c1, "--c2-run", "C2-none")
    assert code == 1
    assert list(done) == ["C1"]
    assert "grantrisk run-all: C2: run C2-none does not exist" in err
