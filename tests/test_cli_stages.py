"""The stage subcommands end to end through the CLI (ARC-07), on a small synthetic corpus."""

import pytest
import yaml
from modelling_helpers import seed_c2, seed_l3, synthetic_corpus

from grantrisk import cli
from grantrisk import config as config_mod
from grantrisk.store import db, runs

TEXTS, LABELS = synthetic_corpus(n_per_class=6)


@pytest.fixture
def config_path(tmp_path):
    """The default configuration with a temporary data root and a small, fast grid."""
    values = config_mod.load().values
    values["data_root"] = str(tmp_path / "data")
    values["represent"]["representations"] = ["tfidf"]
    values["train"]["classifiers"] = ["logreg", "majority"]
    values["train"]["cv"] = {"n_splits": 3, "n_repeats": 2, "random_state": 42}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def seeded(config_path):
    conn = db.connect(config_mod.load(config_path).data_root)
    try:
        yield seed_c2(conn, TEXTS), seed_l3(conn, LABELS)
    finally:
        conn.close()


def run_cli(config_path, capsys, *argv):
    code = cli.main(["--config", str(config_path), *argv])
    out = capsys.readouterr()
    assert code == 0, out.err
    return out.out.strip().splitlines()[-1]


def status(config_path, run_id):
    conn = db.connect(config_mod.load(config_path).data_root)
    try:
        return runs.get(conn, run_id)["status"], runs.inputs(conn, run_id)
    finally:
        conn.close()


def test_represent_then_train(config_path, seeded, capsys):
    c2, l3 = seeded
    m1 = run_cli(config_path, capsys, "represent", "--c2-run", c2)
    assert status(config_path, m1) == ("complete", [c2])
    m2 = run_cli(config_path, capsys, "train", "--m1-run", m1, "--l3-run", l3)
    state, inputs = status(config_path, m2)
    assert state == "complete"
    assert sorted(inputs) == sorted([m1, l3])


def test_represent_estimate_is_not_a_run(config_path, seeded, capsys):
    c2, _ = seeded
    assert cli.main(["--config", str(config_path), "represent", "--c2-run", c2, "--estimate"]) == 0
    conn = db.connect(config_mod.load(config_path).data_root)
    try:
        assert conn.execute("SELECT COUNT(*) FROM runs WHERE stage = 'M1'").fetchone()[0] == 0
    finally:
        conn.close()


def test_train_reports_bad_input_without_traceback(config_path, seeded, capsys):
    assert cli.main(["--config", str(config_path), "train", "--m1-run", "M1-none", "--l3-run", "L3-none"]) == 1
    assert "grantrisk train:" in capsys.readouterr().err


def test_extract_regex(config_path, seeded, capsys):
    c2, _ = seeded
    l1 = run_cli(config_path, capsys, "extract", "--extractor", "regex", "--c2-run", c2)
    assert status(config_path, l1) == ("complete", [c2])
    conn = db.connect(config_mod.load(config_path).data_root)
    try:
        n, sources = conn.execute(
            "SELECT COUNT(DISTINCT doc_id), GROUP_CONCAT(DISTINCT source) FROM factor_observations WHERE run_id = ?", (l1,)
        ).fetchone()
    finally:
        conn.close()
    assert (n, sources) == (len(TEXTS), "regex")
