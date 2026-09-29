"""The M1 run in the database (SPEC-M1-01, -04, -06, -07, -08), with fake embedders only."""

import json

import numpy as np
import pytest
from modelling_helpers import FakeEmbedder, FakeTokenizer, seed_c2, synthetic_corpus, word_vector

from grantrisk.modelling import embeddings, hosted, represent
from grantrisk.store import db, runs


@pytest.fixture
def data_root(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def conn(data_root):
    c = db.connect(data_root)
    yield c
    c.close()


TEXTS, LABELS = synthetic_corpus(n_per_class=2)
DOCS = sorted(TEXTS)
CONFIG = {"represent": {"representations": ["tfidf", "hubert"], "cache_index_batch": 32}}


def report(conn, data_root, run_id):
    return json.loads((data_root / runs.get(conn, run_id)["report_path"]).read_text(encoding="utf-8"))


def features(conn, run_id, rep):
    return {r["doc_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM features WHERE run_id = ? AND representation = ?", (run_id, rep))}


def test_run_writes_features_with_one_text_hash_per_document(conn, data_root):
    """SPEC-M1-01: every representation shows the same text hash for a document."""
    c2 = seed_c2(conn, TEXTS)
    run_id = represent.run(conn, CONFIG, data_root, c2_run_id=c2, embedders={"hubert": FakeEmbedder()})
    assert runs.get(conn, run_id)["status"] == "complete"
    assert runs.inputs(conn, run_id) == [c2]
    tf, hb = features(conn, run_id, "tfidf"), features(conn, run_id, "hubert")
    assert sorted(tf) == sorted(hb) == DOCS
    assert all(tf[d]["text_hash"] == hb[d]["text_hash"] for d in DOCS)
    assert all(tf[d]["cache_key"] is None and hb[d]["cache_key"] for d in DOCS)
    r = report(conn, data_root, run_id)
    assert r["input_runs"] == {"C2": c2}
    assert r["representations"]["hubert"]["computed"] == len(DOCS)
    assert r["representations"]["hubert"]["chunks_max"] >= 1
    assert r["incomplete"] == []


def test_failed_conversions_are_not_in_the_document_set(conn, data_root):
    c2 = seed_c2(conn, TEXTS, failed={DOCS[0]})
    run_id = represent.run(conn, CONFIG, data_root, c2_run_id=c2, embedders={"hubert": FakeEmbedder()})
    assert DOCS[0] not in features(conn, run_id, "hubert")


def test_second_run_embeds_nothing(conn, data_root):
    """SPEC-M1-06: the cache is reused, and its paths are relative."""
    c2 = seed_c2(conn, TEXTS)
    emb = FakeEmbedder()
    first = represent.run(conn, CONFIG, data_root, c2_run_id=c2, embedders={"hubert": emb})
    calls = emb.calls
    second = represent.run(conn, CONFIG, data_root, c2_run_id=c2, embedders={"hubert": emb})
    assert emb.calls == calls
    assert report(conn, data_root, second)["representations"]["hubert"]["reused"] == len(DOCS)
    assert features(conn, first, "hubert") .keys() == features(conn, second, "hubert").keys()
    paths = [r[0] for r in conn.execute("SELECT vector_path FROM feature_cache")]
    assert paths and all(not p.startswith("/") and ":" not in p for p in paths)


def test_a_changed_chunking_is_a_new_cache_entry(conn, data_root):
    c2 = seed_c2(conn, TEXTS)
    represent.run(conn, CONFIG, data_root, c2_run_id=c2, embedders={"hubert": FakeEmbedder(budget=20)})
    other = FakeEmbedder(budget=5)
    represent.run(conn, CONFIG, data_root, c2_run_id=c2, embedders={"hubert": other})
    assert other.calls > 0


def test_cache_index_is_written_in_batches(conn, data_root, monkeypatch):
    c2 = seed_c2(conn, TEXTS)
    written = []
    original = represent.CacheIndex.flush

    def spy(self):
        if self.pending:
            written.append(len(self.pending))
        original(self)

    monkeypatch.setattr(represent.CacheIndex, "flush", spy)
    config = {"represent": {**CONFIG["represent"], "cache_index_batch": 4}}
    represent.run(conn, config, data_root, c2_run_id=c2, embedders={"hubert": FakeEmbedder()})
    assert written == [4, 2]


def test_load_features_in_the_requested_order(conn, data_root):
    c2 = seed_c2(conn, TEXTS)
    run_id = represent.run(conn, CONFIG, data_root, c2_run_id=c2, embedders={"hubert": FakeEmbedder()})
    order = DOCS[::-1]
    texts = represent.load_features(conn, data_root, run_id, "tfidf", order)
    assert texts[0].startswith("Felhívás") and "|" not in texts[0] and "<!--" not in texts[0]
    matrix = represent.load_features(conn, data_root, run_id, "hubert", order)
    assert matrix.shape == (len(DOCS), 5)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    with pytest.raises(ValueError, match="lacks 1 documents"):
        represent.load_features(conn, data_root, run_id, "hubert", [*order, "nosuchdoc"])


def test_a_hosted_model_without_provider_stops_before_the_run(conn, data_root):
    c2 = seed_c2(conn, TEXTS)
    config = {"represent": {"representations": ["e5"], "models": {"e5": {"model_id": "x", "location": "hosted", "provider": None}}}}
    with pytest.raises(embeddings.ProviderNotConfigured):
        represent.run(conn, config, data_root, c2_run_id=c2)
    assert conn.execute("SELECT COUNT(*) FROM runs WHERE stage = 'M1'").fetchone()[0] == 0


# --- Hosted computation (SPEC-M1-04, -07, -08) ---------------------------------------------------


class Transport:
    """Answers with a word vector of each input; fails for inputs that contain ``fail_on``."""

    def __init__(self, fail_on=None):
        self.fail_on = fail_on
        self.calls = 0

    def __call__(self, url, headers, payload, timeout):
        self.calls += 1
        if self.fail_on and any(self.fail_on in t for t in payload["input"]):
            return 400, "bad input"
        return 200, {"data": [{"index": i, "embedding": list(word_vector(t.split()))} for i, t in enumerate(payload["input"])]}


def hosted_e5(transport, price=1.0):
    model = {"model_id": "intfloat/multilingual-e5-large", "revision": "r1", "prefix": "passage: ", "input_limit": 30,
             "prices_per_mtok": {"input": price}}
    return hosted.HostedEmbedder("e5", model, "prov", {"url": "https://example.invalid", "margin_tokens": 2},
                                 transport=transport, tokenizer=FakeTokenizer(), sleep=lambda s: None)


HOSTED = {"represent": {"representations": ["e5"], "equivalence_check": {"enabled": False}}}


def test_estimate_sends_nothing(conn, data_root):
    c2 = seed_c2(conn, TEXTS)
    transport = Transport()
    est = represent.estimate(conn, HOSTED, data_root, c2_run_id=c2, embedders={"e5": hosted_e5(transport, price=2.0)})
    assert transport.calls == 0
    e5 = est["representations"]["e5"]
    assert e5["to_compute"] == len(DOCS) and e5["cached"] == 0 and e5["tokens_sent"] > e5["tokens"]
    assert e5["usd_estimate"] == pytest.approx(e5["tokens_sent"] * 2.0 / 1e6)


def test_hosted_run_needs_confirmation_and_records_the_provider(conn, data_root):
    c2 = seed_c2(conn, TEXTS)
    transport = Transport()
    with pytest.raises(ValueError, match="confirm"):
        represent.run(conn, HOSTED, data_root, c2_run_id=c2, embedders={"e5": hosted_e5(transport)})
    assert transport.calls == 0
    run_id = represent.run(conn, HOSTED, data_root, c2_run_id=c2, embedders={"e5": hosted_e5(transport)}, confirmed=True)
    rows = conn.execute("SELECT provider, location FROM feature_cache").fetchall()
    assert {tuple(r) for r in rows} == {("prov", "hosted")}
    assert report(conn, data_root, run_id)["representations"]["e5"]["cost_usd"] > 0
    # everything cached: a new run sends nothing and needs no confirmation
    calls = transport.calls
    represent.run(conn, HOSTED, data_root, c2_run_id=c2, embedders={"e5": hosted_e5(transport)})
    assert transport.calls == calls


def test_budget_ceiling_stops_the_run_and_resume_finishes_it(conn, data_root):
    c2 = seed_c2(conn, TEXTS)
    config = {"represent": {**HOSTED["represent"], "budget_usd": 1e-9}}
    transport = Transport()
    with pytest.raises(represent.BudgetReached):
        represent.run(conn, config, data_root, c2_run_id=c2, embedders={"e5": hosted_e5(transport)}, confirmed=True)
    failed = conn.execute("SELECT run_id, status FROM runs WHERE stage = 'M1'").fetchone()
    assert failed["status"] == "failed"
    assert conn.execute("SELECT COUNT(*) FROM feature_cache").fetchone()[0] == 1  # the vector paid for is kept
    assert conn.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 0
    run_id = represent.run(conn, HOSTED, data_root, c2_run_id=c2, embedders={"e5": hosted_e5(transport)},
                           confirmed=True, resume_run_id=failed["run_id"])
    assert run_id == failed["run_id"] and runs.get(conn, run_id)["status"] == "complete"
    assert transport.calls == len(DOCS)  # the first document was not sent again


def test_a_budget_needs_prices(conn, data_root):
    c2 = seed_c2(conn, TEXTS)
    config = {"represent": {**HOSTED["represent"], "budget_usd": 5}}
    with pytest.raises(ValueError, match="prices"):
        represent.run(conn, config, data_root, c2_run_id=c2, embedders={"e5": hosted_e5(Transport(), price=None)}, confirmed=True)


def test_a_missing_vector_makes_the_representation_incomplete(conn, data_root):
    """SPEC-M1-08."""
    victim = DOCS[1]
    c2 = seed_c2(conn, {**TEXTS, victim: TEXTS[victim] + " HIBÁS"})
    transport = Transport(fail_on="HIBÁS")
    run_id = represent.run(conn, HOSTED, data_root, c2_run_id=c2, embedders={"e5": hosted_e5(transport)}, confirmed=True)
    rows = features(conn, run_id, "e5")
    assert rows[victim]["status"] == "missing" and "400" in rows[victim]["error"]
    r = report(conn, data_root, run_id)
    assert r["incomplete"] == ["e5"]
    assert [m["doc_id"] for m in r["representations"]["e5"]["missing"]] == [victim]
    with pytest.raises(ValueError, match="lacks 1 documents"):
        represent.load_features(conn, data_root, run_id, "e5", DOCS)


def test_equivalence_check(conn, data_root):
    """SPEC-M1-07: cosine per document of the hosted and the local vectors."""
    c2 = seed_c2(conn, TEXTS)
    config = {"represent": {"representations": ["e5"], "equivalence_check": {"n_documents": 3, "min_cosine": 0.99}}}
    local = FakeEmbedder(key="e5", model_id="intfloat/multilingual-e5-large")
    run_id = represent.run(conn, config, data_root, c2_run_id=c2, embedders={"e5": hosted_e5(Transport())},
                           equivalence_embedder=local, confirmed=True)
    check = report(conn, data_root, run_id)["equivalence_check"]
    assert len(check["documents"]) == 3 and check["provider"] == "prov"
    assert check["passed"] and check["min_cosine"] >= 0.99

    class Different(FakeEmbedder):
        def embed_chunks(self, text, chunks):
            return np.stack([np.array([0, 0, 1.0, 0, 0]) for _ in chunks])

    run_id = represent.run(conn, config, data_root, c2_run_id=c2, embedders={"e5": hosted_e5(Transport())},
                           equivalence_embedder=Different(key="e5", model_id="other"))
    assert report(conn, data_root, run_id)["equivalence_check"]["passed"] is False


def test_sample_is_spread_over_the_documents():
    assert represent._sample(list("abcdefghi"), 3) == ["a", "e", "i"]
    assert represent._sample(list("ab"), 5) == ["a", "b"]
