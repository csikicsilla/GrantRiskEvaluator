"""The L3 run in the database (SPEC-L3-03, -12) and the ``label`` command."""

import json

import pytest
from l3_helpers import doc, seed_runs

from grantrisk import cli
from grantrisk.labelling import label
from grantrisk.store import db, runs

CONFIG = {"label": {"low_coverage_threshold": 5}}
DOCS = [doc(f"D{i}", idotartam=v) for i, v in enumerate([6, 15, 20, 30, 6, 15, 20, 30, 6])] + [
    doc("T1", "TOP", tam_osszeg=None)
]


@pytest.fixture
def data_root(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def conn(data_root):
    c = db.connect(data_root)
    yield c
    c.close()


def count(conn, table, run_id):
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)).fetchone()[0]


def last_run(conn, stage):
    return conn.execute("SELECT * FROM runs WHERE stage = ? ORDER BY rowid DESC LIMIT 1", (stage,)).fetchone()


def test_run_writes_labels_and_report(conn, data_root):
    c1, l2 = seed_runs(conn, DOCS)
    run_id = label.run(conn, CONFIG, data_root, l2_run_id=l2, c1_run_id=c1)
    row = runs.get(conn, run_id)
    assert row["status"] == "complete"
    assert runs.inputs(conn, run_id) == sorted([c1, l2])
    assert count(conn, "risk_labels", run_id) == len(DOCS)
    assert count(conn, "risk_label_factors", run_id) == 10 * len(DOCS)
    report = json.loads((data_root / row["report_path"]).read_text(encoding="utf-8"))
    assert report["input_runs"] == {"L2": l2, "C1": c1}
    assert report["n_documents"] == len(DOCS)
    t1 = conn.execute(
        "SELECT points_exact, origin FROM risk_label_factors WHERE run_id = ? AND doc_id = 'T1' AND factor = 'tam_osszeg'",
        (run_id,),
    ).fetchone()
    assert tuple(t1) == ("1", "top_rule")


def test_same_input_gives_same_output(conn, data_root):
    c1, l2 = seed_runs(conn, DOCS)
    a = label.run(conn, CONFIG, data_root, l2_run_id=l2, c1_run_id=c1)
    b = label.run(conn, CONFIG, data_root, l2_run_id=l2, c1_run_id=c1)

    def rows(run_id):
        return [
            tuple(r)[1:]
            for r in conn.execute("SELECT * FROM risk_labels WHERE run_id = ? ORDER BY doc_id", (run_id,))
        ]

    def report(run_id):
        return (data_root / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8")

    assert rows(a) == rows(b)
    assert report(a) == report(b)


def test_missing_document_record_fails_and_writes_nothing(conn, data_root):
    c1, l2 = seed_runs(conn, DOCS, skip_documents={"D3"})
    with pytest.raises(label.L3InputError, match="D3"):
        label.run(conn, CONFIG, data_root, l2_run_id=l2, c1_run_id=c1)
    failed = last_run(conn, "L3")
    assert failed["status"] == "failed"
    assert count(conn, "risk_labels", failed["run_id"]) == 0
    assert not (data_root / "reports" / failed["run_id"]).exists() or not any(
        (data_root / "reports" / failed["run_id"]).iterdir()
    )


def test_failure_after_the_first_document_leaves_nothing(conn, data_root, monkeypatch):
    c1, l2 = seed_runs(conn, DOCS)
    real_insert = label._insert_document
    calls = []

    def failing_insert(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected failure")
        real_insert(*args)

    monkeypatch.setattr(label, "_insert_document", failing_insert)
    with pytest.raises(RuntimeError, match="injected"):
        label.run(conn, CONFIG, data_root, l2_run_id=l2, c1_run_id=c1)
    failed = last_run(conn, "L3")
    assert failed["status"] == "failed"
    assert "injected failure" in failed["error"]
    assert count(conn, "risk_labels", failed["run_id"]) == 0
    assert count(conn, "risk_label_factors", failed["run_id"]) == 0
    assert not (data_root / "reports" / failed["run_id"] / "l3_report.json").exists()


def test_input_runs_must_be_complete_runs_of_the_right_stage(conn, data_root):
    c1, l2 = seed_runs(conn, DOCS)
    with pytest.raises(ValueError, match="not L2"):
        label.run(conn, CONFIG, data_root, l2_run_id=c1, c1_run_id=c1)
    assert last_run(conn, "L3") is None


def test_label_command(conn, data_root, tmp_path, capsys):
    c1, l2 = seed_runs(conn, DOCS)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"data_root: {data_root.as_posix()}\nlabel:\n  low_coverage_threshold: 5\n", encoding="utf-8"
    )
    assert cli.main(["--config", str(config_file), "label", "--l2-run", l2, "--c1-run", c1]) == 0
    run_id = capsys.readouterr().out.strip()
    assert runs.get(conn, run_id)["status"] == "complete"
    assert cli.main(["--config", str(config_file), "label", "--l2-run", "nope", "--c1-run", c1]) == 1
    assert "does not exist" in capsys.readouterr().err
