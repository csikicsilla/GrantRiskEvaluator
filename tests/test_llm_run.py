"""The LLM extractor as a run, against a fake API client (SPEC-L1-08, -10 … -14). Nothing is sent."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_gold_import import fake_sha, seed_c1

from grantrisk.extraction.llm import run as llm
from grantrisk.labelling.scoring import FACTORS
from grantrisk.store import db, runs
from grantrisk.store.db import transaction

PROMPT = Path(__file__).parents[1] / "prompts" / "P5_system_prompt.txt"
PRICES = {"input": 1.0, "output": 5.0}


def markdown(doc, pages=2):
    body = "".join(f"<!-- page {i} -->\nA {doc} felhívás {i}. oldala. " + "szöveg " * 50 + "\n" for i in range(1, pages))
    return body + f"<!-- page {pages} -->\nAz előleg mértéke 50%.\n"


class FakeClient:
    """Answers eloleg = 50 with a quote from the text; everything else null."""

    def __init__(self, input_limit=200_000, fail_for=()):
        self.created, self.fail_for = [], set(fail_for)
        self.messages = SimpleNamespace(create=self._create, count_tokens=self._count)
        self.models = SimpleNamespace(retrieve=lambda model: SimpleNamespace(max_input_tokens=input_limit))

    def _count(self, model, system, messages):
        return SimpleNamespace(input_tokens=(len(system) + len(messages[0]["content"])) // 4)

    def _create(self, **params):
        text = params["messages"][0]["content"]
        self.created.append(text)
        if any(marker in text for marker in self.fail_for):
            raise RuntimeError("overloaded after retries")
        data = {f: {"value": None, "evidence": ""} for f in FACTORS}
        if "előleg mértéke" in text:
            data["eloleg"] = {"value": 50, "evidence": "Az előleg mértéke 50%."}
        body = {"model": params["model"], "stop_reason": "end_turn",
                "content": [{"type": "text", "text": json.dumps(data)}],
                "usage": {"input_tokens": 1000, "output_tokens": 200}}
        return SimpleNamespace(to_dict=lambda: body)


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "data")
    yield c
    c.close()


def seed_c2(conn, docs):
    """A complete C2 run with the given {name: markdown} texts; returns (run id, doc ids)."""
    c1 = seed_c1(conn, [fake_sha(n) for n in docs])
    c2 = runs.start(conn, "C2", {"test": True}, inputs=[c1])
    ids = {}
    with transaction(conn):
        for name, md in docs.items():
            doc_id = fake_sha(name)[:16]
            ids[name] = doc_id
            conn.execute(
                "INSERT INTO document_texts (run_id, doc_id, status, markdown, converter, converter_version,"
                " settings_hash, cleaning_version) VALUES (?, ?, 'ok', ?, 'fake', '1', 'x', '1')",
                (c2, doc_id, md),
            )
        runs.complete(conn, c2)
    return c2, ids


def settings(**overrides):
    return llm.Settings(**{"model": "test-model", "prompt_path": PROMPT, "max_parallel_requests": 1,
                           "prices_per_mtok": PRICES} | overrides)


def rows(conn, run_id):
    return [tuple(r)[1:] for r in conn.execute(
        "SELECT * FROM factor_observations WHERE run_id = ? ORDER BY doc_id, factor", (run_id,))]


def test_run_on_selected_documents(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A"), "B": markdown("B")})
    client = FakeClient()
    run_id = llm.run(conn, {}, tmp_path / "data", client, settings(), c2_run_id=c2, doc_ids=[ids["A"]])
    obs = {r["factor"]: r for r in conn.execute("SELECT * FROM factor_observations WHERE run_id = ?", (run_id,))}
    assert len(obs) == 10
    assert (obs["eloleg"]["status"], json.loads(obs["eloleg"]["value_json"]), obs["eloleg"]["evidence_page"]) == ("found", 50, 2)
    assert obs["eloleg"]["source"] == "llm:test-model"
    assert obs["eloleg"]["prompt_version"] == "P5"
    assert (tmp_path / "data" / obs["eloleg"]["raw_response_path"]).exists()
    assert obs["fin_form"]["status"] == "not_found"
    report = json.loads((tmp_path / "data" / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))
    assert report["documents"] == 1 and report["cost_usd"] == pytest.approx(0.002)
    assert runs.inputs(conn, run_id) == [c2]


def test_prompt_is_sent_unchanged(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A")})
    sent = []
    client = FakeClient()
    create = client.messages.create
    client.messages.create = lambda **p: sent.append(p) or create(**p)
    llm.run(conn, {}, tmp_path / "data", client, settings(), c2_run_id=c2, doc_ids=[ids["A"]])
    assert sent[0]["system"] == PROMPT.read_text(encoding="utf-8")
    assert sent[0]["model"] == "test-model" and sent[0]["temperature"] == 0


def test_corpus_run_needs_confirmation(conn, tmp_path):
    c2, _ = seed_c2(conn, {"A": markdown("A")})
    with pytest.raises(ValueError, match="confirmation"):
        llm.run(conn, {}, tmp_path / "data", FakeClient(), settings(), c2_run_id=c2)
    assert llm.run(conn, {}, tmp_path / "data", FakeClient(), settings(), c2_run_id=c2, confirmed=True)


def test_reparse_without_the_api_gives_the_same_rows(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A"), "B": markdown("B")})
    first = llm.run(conn, {}, tmp_path / "data", FakeClient(), settings(), c2_run_id=c2, confirmed=True)
    again = llm.run(conn, {}, tmp_path / "data", None, settings(), c2_run_id=c2, confirmed=True)  # no client
    assert rows(conn, first) == rows(conn, again)


def test_budget_ceiling_stops_and_resume_continues(conn, tmp_path):
    c2, _ = seed_c2(conn, {n: markdown(n) for n in "ABC"})
    client = FakeClient()
    with pytest.raises(llm.BudgetReached):
        llm.run(conn, {}, tmp_path / "data", client, settings(budget_usd=0.001), c2_run_id=c2, confirmed=True)
    run_id = conn.execute("SELECT run_id FROM runs WHERE stage = 'L1'").fetchone()[0]
    assert runs.get(conn, run_id)["status"] == "failed"
    assert len(client.created) == 1  # 0.002 USD spent after the first document
    client = FakeClient()
    llm.run(conn, {}, tmp_path / "data", client, settings(budget_usd=1.0), c2_run_id=c2, confirmed=True, resume_run_id=run_id)
    assert len(client.created) == 2  # only the two missing documents
    assert runs.get(conn, run_id)["status"] == "complete"


def test_budget_needs_prices(conn, tmp_path):
    c2, _ = seed_c2(conn, {"A": markdown("A")})
    with pytest.raises(ValueError, match="prices"):
        llm.run(conn, {}, tmp_path / "data", FakeClient(), settings(budget_usd=1.0, prices_per_mtok=None),
                c2_run_id=c2, confirmed=True)


def test_a_failing_document_gets_error_observations(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A"), "B": markdown("B")})
    run_id = llm.run(conn, {}, tmp_path / "data", FakeClient(fail_for=["A felhívás"]), settings(),
                     c2_run_id=c2, confirmed=True)
    statuses = {r[0] for r in conn.execute(
        "SELECT status FROM factor_observations WHERE run_id = ? AND doc_id = ?", (run_id, ids["A"]))}
    assert statuses == {"error"}
    assert runs.get(conn, run_id)["status"] == "complete"


def test_long_document_is_split_and_merged(conn, tmp_path):
    long = markdown("L", pages=6)
    c2, ids = seed_c2(conn, {"L": long})
    limit = int((len(PROMPT.read_text(encoding="utf-8")) + len(long) // 2) // 4) + 8192  # about half the text
    client = FakeClient(input_limit=int(limit / llm.SAFETY) + 1)
    run_id = llm.run(conn, {}, tmp_path / "data", client, settings(), c2_run_id=c2, confirmed=True)
    assert len(client.created) >= 2
    eloleg = conn.execute("SELECT * FROM factor_observations WHERE run_id = ? AND factor = 'eloleg'", (run_id,)).fetchone()
    assert eloleg["status"] == "found" and "split_document" in json.loads(eloleg["warnings_json"])
    assert len(eloleg["raw_response_path"].split(";")) == len(client.created)


def test_settings_per_model():
    cfg = {"extract": {"llm": {"prompt": "p.txt", "model": "claude-haiku-4-5-20251001", "temperature": 0,
                               "prices_per_mtok": {"input": 1.0, "output": 5.0},
                               "model_settings": {"claude-sonnet-5": {"temperature": None, "thinking": "disabled",
                                                                      "prices_per_mtok": {"input": 2.0, "output": 10.0}}}}}}
    haiku = llm.settings_from_config(cfg, Path)
    sonnet = llm.settings_from_config(cfg, Path, "claude-sonnet-5")
    assert (haiku.model, haiku.temperature, haiku.thinking) == ("claude-haiku-4-5-20251001", 0, None)
    assert (sonnet.temperature, sonnet.thinking, sonnet.prices_per_mtok["input"]) == (None, "disabled", 2.0)
    assert sonnet.source == "llm:claude-sonnet-5"


def test_estimate(conn, tmp_path):
    c2, _ = seed_c2(conn, {"A": markdown("A"), "B": markdown("B")})
    client = FakeClient()
    result = llm.estimate(client, settings(), llm.documents(conn, c2, None))
    assert (result["documents"], result["requests"]) == (2, 2)
    assert result["usd_estimate"] > 0
    assert client.created == []  # counting only


# --- DEC-56: the budget over the whole run, retries, and stored answers ---------------------------------


def test_a_resumed_run_counts_what_it_spent_before(conn, tmp_path):
    c2, _ = seed_c2(conn, {n: markdown(n) for n in "ABC"})
    with pytest.raises(llm.BudgetReached):
        llm.run(conn, {}, tmp_path / "data", FakeClient(), settings(budget_usd=0.001), c2_run_id=c2, confirmed=True)
    run_id = conn.execute("SELECT run_id FROM runs WHERE stage = 'L1'").fetchone()[0]
    client = FakeClient()
    with pytest.raises(llm.BudgetReached):  # 0.002 USD spent before; one more document reaches 0.003
        llm.run(conn, {}, tmp_path / "data", client, settings(budget_usd=0.003), c2_run_id=c2, confirmed=True,
                resume_run_id=run_id)
    assert len(client.created) == 1
    assert [r[0] for r in conn.execute("SELECT resume FROM run_resumes WHERE run_id = ?", (run_id,))] == [1]


class Interrupt(BaseException):
    """Stands for Ctrl+C or a killed process."""


def test_a_resumed_run_asks_again_for_the_documents_that_failed(conn, tmp_path):
    c2, ids = seed_c2(conn, {n: markdown(n) for n in "ABC"})
    order = sorted(ids, key=ids.get)  # the run takes the documents in doc_id order
    first, last = order[0], order[-1]
    client = FakeClient(fail_for=[f"{first} felhívás"])
    create = client.messages.create

    def create_or_stop(**params):
        if f"{last} felhívás" in params["messages"][0]["content"]:
            raise Interrupt()
        return create(**params)

    client.messages.create = create_or_stop
    with pytest.raises(Interrupt):
        llm.run(conn, {}, tmp_path / "data", client, settings(), c2_run_id=c2, confirmed=True)
    run_id = conn.execute("SELECT run_id FROM runs WHERE stage = 'L1'").fetchone()[0]
    stored = {r[0] for r in conn.execute("SELECT DISTINCT doc_id FROM factor_observations WHERE run_id = ?", (run_id,))}
    assert ids[first] not in stored  # the failure waits for the end of the run
    client = FakeClient()
    llm.run(conn, {}, tmp_path / "data", client, settings(), c2_run_id=c2, confirmed=True, resume_run_id=run_id)
    assert len(client.created) == 2  # the failed and the interrupted document
    statuses = {r[0] for r in conn.execute(
        "SELECT status FROM factor_observations WHERE run_id = ? AND doc_id = ?", (run_id, ids[first]))}
    assert "error" not in statuses


def cut_off_first(client, times=1):
    """The first ``times`` answers stop at max_tokens."""
    create, calls = client.messages.create, []

    def create_cut(**params):
        answer = create(**params)
        calls.append(1)
        if len(calls) <= times:
            body = {**answer.to_dict(), "stop_reason": "max_tokens"}
            return SimpleNamespace(to_dict=lambda: body)
        return answer

    client.messages.create = create_cut
    return client


def test_an_unusable_answer_is_asked_once_more(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A")})
    client = cut_off_first(FakeClient())
    run_id = llm.run(conn, {}, tmp_path / "data", client, settings(), c2_run_id=c2, doc_ids=[ids["A"]])
    eloleg = conn.execute("SELECT * FROM factor_observations WHERE run_id = ? AND factor = 'eloleg'", (run_id,)).fetchone()
    assert eloleg["status"] == "found" and eloleg["raw_response_path"].endswith("-a2.json")
    assert len(client.created) == 2
    assert (tmp_path / "data" / eloleg["raw_response_path"].replace("-a2.json", "-a1.json")).exists()  # kept


def test_an_answer_is_asked_at_most_twice(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A")})
    client = cut_off_first(FakeClient(), times=5)
    first = llm.run(conn, {}, tmp_path / "data", client, settings(), c2_run_id=c2, doc_ids=[ids["A"]])
    assert len(client.created) == 2
    assert {r[0] for r in conn.execute("SELECT status FROM factor_observations WHERE run_id = ?", (first,))} == {"error"}
    client = FakeClient()
    llm.run(conn, {}, tmp_path / "data", client, settings(), c2_run_id=c2, doc_ids=[ids["A"]])
    assert client.created == []  # both attempts are used up; a later run does not ask again


def test_changed_request_settings_never_reuse_an_answer(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A")})
    llm.run(conn, {}, tmp_path / "data", FakeClient(), settings(), c2_run_id=c2, doc_ids=[ids["A"]])
    same, more_tokens = FakeClient(), FakeClient()
    llm.run(conn, {}, tmp_path / "data", same, settings(), c2_run_id=c2, doc_ids=[ids["A"]])
    llm.run(conn, {}, tmp_path / "data", more_tokens, settings(max_tokens=16000), c2_run_id=c2, doc_ids=[ids["A"]])
    assert (len(same.created), len(more_tokens.created)) == (0, 1)


def test_the_report_names_what_the_run_paid_for_and_what_it_skipped(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A")})
    llm.run(conn, {}, tmp_path / "data", FakeClient(), settings(), c2_run_id=c2, doc_ids=[ids["A"]])
    again = llm.run(conn, {}, tmp_path / "data", FakeClient(), settings(), c2_run_id=c2, doc_ids=[ids["A"], "0" * 16])
    report = json.loads((tmp_path / "data" / runs.get(conn, again)["report_path"]).read_text(encoding="utf-8"))
    assert report["cost_usd"] == pytest.approx(0.002) and report["cost_usd_paid_by_this_run"] == 0
    assert report["skipped"] == [{"doc_id": "0" * 16, "reason": "not in the C2 run"}]
    assert report["models_that_answered"] == ["test-model"]


def test_the_run_records_its_budget_ceiling(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A")})
    run_id = llm.run(conn, {}, tmp_path / "data", FakeClient(), settings(budget_usd=4.0), c2_run_id=c2, doc_ids=[ids["A"]])
    assert json.loads(runs.get(conn, run_id)["config_json"])["llm_run"]["budget_usd"] == 4.0
    report = json.loads((tmp_path / "data" / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))
    assert report["budget_usd"] == 4.0
