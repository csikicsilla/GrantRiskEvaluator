"""The LLM extractor through the Message Batches API (DEC-65), against a fake client. Nothing is sent."""

import json
from types import SimpleNamespace

import pytest
from test_llm_run import FakeClient, markdown, seed_c2, settings

from grantrisk.extraction.llm import batch
from grantrisk.extraction.llm import run as llm
from grantrisk.store import db, runs


class FakeBatchClient(FakeClient):
    """FakeClient with messages.batches: each batch ends after ``polls`` polls; answers come from FakeClient."""

    def __init__(self, polls=1, fail_once=(), cut_off_once=(), **kwargs):
        super().__init__(**kwargs)
        self.batches, self.polls, self.fail_once, self.cut_off_once = {}, polls, set(fail_once), set(cut_off_once)
        self.failed, self.cut = set(), set()
        self.messages.batches = SimpleNamespace(create=self._create_batch, retrieve=self._retrieve, results=self._results)

    def _create_batch(self, requests):
        batch_id = f"msgbatch_{len(self.batches) + 1}"
        self.batches[batch_id] = {"requests": list(requests), "polls": self.polls}
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    def _retrieve(self, batch_id):
        b = self.batches[batch_id]
        counts = SimpleNamespace(processing=len(b["requests"]), succeeded=0, errored=0)
        if b["polls"] > 0:
            b["polls"] -= 1
            return SimpleNamespace(processing_status="in_progress", request_counts=counts)
        return SimpleNamespace(processing_status="ended", request_counts=counts)

    def _results(self, batch_id):
        for r in self.batches[batch_id]["requests"]:
            text = r["params"]["messages"][0]["content"]
            marker = next((m for m in self.fail_once if m in text), None)
            if marker and marker not in self.failed:
                self.failed.add(marker)
                yield SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type="errored"))
                continue
            body = self._create(**r["params"]).to_dict()
            marker = next((m for m in self.cut_off_once if m in text), None)
            if marker and marker not in self.cut:
                self.cut.add(marker)
                body = {**body, "stop_reason": "max_tokens"}
            yield SimpleNamespace(custom_id=r["custom_id"],
                                  result=SimpleNamespace(type="succeeded", message=SimpleNamespace(to_dict=lambda b=body: b)))

    def requests_sent(self):
        return [r for b in self.batches.values() for r in b["requests"]]


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "data")
    yield c
    c.close()


def run_batch(conn, tmp_path, client, s=None, **kwargs):
    return batch.run_batch(conn, {}, tmp_path / "data", client, s or settings(), sleep=lambda seconds: None, **kwargs)


def test_answers_are_stored_like_those_of_a_direct_run_at_half_the_price(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A"), "B": markdown("B")})
    client = FakeBatchClient()
    run_id = run_batch(conn, tmp_path, client, c2_run_id=c2, doc_ids=list(ids.values()))
    sent = client.requests_sent()
    assert len(client.batches) == 1 and len(sent) == 2
    assert sent[0]["params"]["temperature"] == 0 and "extra_body" not in sent[0]["params"]  # in the request itself
    assert sent[0]["params"]["system"] == settings().prompt
    eloleg = conn.execute("SELECT * FROM factor_observations WHERE run_id = ? AND factor = 'eloleg'", (run_id,)).fetchall()
    assert [(r["status"], json.loads(r["value_json"])) for r in eloleg] == [("found", 50), ("found", 50)]
    record = json.loads((tmp_path / "data" / eloleg[0]["raw_response_path"]).read_text(encoding="utf-8"))
    assert (record["service"], record["batch_id"], record["run_id"]) == ("batch", "msgbatch_1", run_id)
    report = json.loads((tmp_path / "data" / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))
    assert report["cost_usd"] == pytest.approx(0.002)  # 2 × 0.002 USD at the direct price, halved
    assert report["batch_outcomes"] == {"succeeded": 2}
    assert runs.get(conn, run_id)["status"] == "complete"


def test_failed_requests_go_into_a_second_batch(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A"), "B": markdown("B")})
    client = FakeBatchClient(fail_once=["A felhívás"])
    run_id = run_batch(conn, tmp_path, client, c2_run_id=c2, doc_ids=list(ids.values()))
    assert [len(b["requests"]) for b in client.batches.values()] == [2, 1]
    statuses = {r[0] for r in conn.execute(
        "SELECT status FROM factor_observations WHERE run_id = ? AND doc_id = ?", (run_id, ids["A"]))}
    assert "error" not in statuses


def test_an_unusable_answer_is_asked_again_in_the_next_batch(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A")})
    client = FakeBatchClient(cut_off_once=["A felhívás"])
    run_id = run_batch(conn, tmp_path, client, c2_run_id=c2, doc_ids=[ids["A"]])
    assert [r["custom_id"] for r in client.requests_sent()] == [f"{ids['A']}-p1of1-a1", f"{ids['A']}-p1of1-a2"]
    eloleg = conn.execute("SELECT * FROM factor_observations WHERE run_id = ? AND factor = 'eloleg'", (run_id,)).fetchone()
    assert eloleg["status"] == "found" and eloleg["raw_response_path"].endswith("-a2.json")


def test_the_ceiling_is_checked_against_the_upper_bound_before_submitting(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A")})
    client = FakeBatchClient()
    with pytest.raises(llm.BudgetReached, match="Nothing was submitted"):
        run_batch(conn, tmp_path, client, settings(budget_usd=0.01), c2_run_id=c2, doc_ids=[ids["A"]])
    assert client.batches == {}  # 8192 output tokens at 5 USD per million, halved, is above 0.01 USD


class Interrupt(BaseException):
    pass


def test_a_resumed_run_collects_its_batches_instead_of_submitting_them_again(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A"), "B": markdown("B")})
    client = FakeBatchClient(polls=3)

    def stop(seconds):
        raise Interrupt()

    with pytest.raises(Interrupt):
        batch.run_batch(conn, {}, tmp_path / "data", client, settings(), c2_run_id=c2, doc_ids=list(ids.values()),
                        sleep=stop)
    run_id = conn.execute("SELECT run_id FROM runs WHERE stage = 'L1'").fetchone()[0]
    assert runs.get(conn, run_id)["status"] == "failed" and len(client.batches) == 1
    run_batch(conn, tmp_path, client, c2_run_id=c2, doc_ids=list(ids.values()), resume_run_id=run_id)
    assert len(client.batches) == 1  # collected, not submitted again
    assert runs.get(conn, run_id)["status"] == "complete"
    assert conn.execute("SELECT COUNT(*) FROM factor_observations WHERE run_id = ? AND status = 'error'",
                        (run_id,)).fetchone()[0] == 0


def test_answers_of_a_direct_run_are_reused(conn, tmp_path):
    c2, ids = seed_c2(conn, {"A": markdown("A")})
    llm.run(conn, {}, tmp_path / "data", FakeClient(), settings(), c2_run_id=c2, doc_ids=[ids["A"]])
    client = FakeBatchClient()
    run_id = run_batch(conn, tmp_path, client, c2_run_id=c2, doc_ids=[ids["A"]])
    assert client.batches == {}
    assert conn.execute("SELECT status FROM factor_observations WHERE run_id = ? AND factor = 'eloleg'",
                        (run_id,)).fetchone()[0] == "found"


def test_a_corpus_run_needs_confirmation(conn, tmp_path):
    c2, _ = seed_c2(conn, {"A": markdown("A")})
    with pytest.raises(ValueError, match="confirmation"):
        run_batch(conn, tmp_path, FakeBatchClient(), c2_run_id=c2)


def test_batch_requests_fit_the_installed_sdk():
    """The request of a part, as the SDK sends it inside a batch (anthropic 1.x)."""
    anthropic = pytest.importorskip("anthropic")
    from anthropic.types.messages.batch_create_params import Request  # noqa: F401  (the type exists)

    params = batch.batch_params(settings(), markdown("A"))
    assert params["temperature"] == 0 and params["output_config"]["format"]["type"] == "json_schema"
    assert batch.custom_id("0123456789abcdef", 1, 2, 1) == "0123456789abcdef-p1of2-a1"
