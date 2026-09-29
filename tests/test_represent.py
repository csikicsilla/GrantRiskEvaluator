"""M1 building blocks: plain text, chunking, pooling, embedders and TF-IDF (SPEC-M1-01 … -05)."""

import numpy as np
import pytest
from modelling_helpers import FakeEmbedder, FakeTokenizer

from grantrisk.modelling import embeddings, hosted, tfidf
from grantrisk.modelling.embeddings import Chunk, LocalTransformerEmbedder, embed_document, plan_chunks, pool_chunks
from grantrisk.modelling.text import plain_text

MARKDOWN = """<!-- page 1 -->
# Felhívás

## 1. A támogatás összege

A **vissza nem térítendő** támogatás 50 millió Ft.

| Tevékenység | Összeg (Ft) |
|---|---:|
| eszközbeszerzés | 10 000 000 |

<!-- page 2 -->
### 2. Határidő
A benyújtás 2024.01.31-ig.
"""


def test_plain_text_removes_markers_separators_pipes_and_heading_marks():
    """SPEC-M1-01."""
    assert plain_text(MARKDOWN) == (
        "Felhívás\n\n1. A támogatás összege\n\nA **vissza nem térítendő** támogatás 50 millió Ft.\n\n"
        "Tevékenység Összeg (Ft)\neszközbeszerzés 10 000 000\n\n2. Határidő\nA benyújtás 2024.01.31-ig."
    )


def test_plain_text_keeps_a_hash_sign_inside_a_line():
    assert plain_text("<!-- page 1 -->\nA #3 pont szerint") == "A #3 pont szerint"


@pytest.mark.parametrize("n_tokens", [0, 1, 7, 20, 21, 45])
def test_chunks_reassemble_the_tokens_and_the_text(n_tokens):
    """SPEC-M1-02: no token and no character is lost, and no chunk exceeds the budget."""
    text = " ".join(f"szó{i}" for i in range(n_tokens))
    tok = FakeTokenizer()
    enc = tok(text, return_offsets_mapping=True)
    chunks = plan_chunks(enc["input_ids"], enc["offset_mapping"], len(text), budget=7)
    assert [i for c in chunks for i in c.ids] == enc["input_ids"]
    assert "".join(text[c.start : c.end] for c in chunks) == text
    assert all(len(c.ids) <= 7 for c in chunks)
    assert len(chunks) == max(1, -(-n_tokens // 7))  # a short document is one chunk


def test_chunk_budget_must_leave_room():
    with pytest.raises(ValueError):
        plan_chunks([1, 2], None, 3, budget=0)


def test_pooling_weights_chunks_by_their_tokens():
    """SPEC-M1-03: one and a half chunks give a 2:1 weighting towards the full chunk."""
    v = pool_chunks(np.array([[1.0, 0.0], [0.0, 1.0]]), [20, 10])
    assert np.allclose(v, np.array([2.0, 1.0]) / np.sqrt(5))
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_embed_document_one_and_a_half_chunks():
    emb = FakeEmbedder(budget=4)
    text = "alacsony alacsony alacsony alacsony magas magas"
    dv = embed_document(emb, text)
    assert (dv.n_chunks, dv.n_tokens) == (2, 6)
    first = np.array([4, 0, 0, 4, 1.0])  # word_vector of the full chunk
    second = np.array([0, 0, 2, 2, 1.0])
    expected = (4 * first + 2 * second) / 6
    assert np.allclose(dv.vector, expected / np.linalg.norm(expected))


def test_token_pooling():
    hidden = np.array([[[1.0, 0], [3.0, 2], [9.0, 9]]])  # the third token is padding
    mask = np.array([[1, 1, 0]])
    assert np.allclose(embeddings.pool_tokens(hidden, mask, "mean"), [[2.0, 1.0]])
    assert np.allclose(embeddings.pool_tokens(hidden, mask, "cls"), [[1.0, 0.0]])
    assert np.allclose(embeddings.pool_tokens(hidden, mask, "last_token"), [[3.0, 2.0]])


def local_embedder(prefix="", **kw):
    seen = []

    def forward(ids, mask):
        seen.append(ids.copy())
        return np.repeat(mask[..., None].astype(float), 3, axis=2) * ids[..., None]

    emb = LocalTransformerEmbedder("e5", "fake/e5", prefix=prefix, input_limit=8, tokenizer=FakeTokenizer(),
                                   forward=forward, batch_size=2, **kw)
    return emb, seen


def test_local_embedder_adds_prefix_and_special_tokens_to_every_chunk():
    """SPEC-M1-03: e5 inputs start with 'passage: '; SPEC-M1-02: the limit counts specials and prefix."""
    emb, seen = local_embedder(prefix="passage: ")
    prefix_ids = FakeTokenizer()("passage:")["input_ids"]
    assert emb.chunk_budget() == 8 - 2 - len(prefix_ids)
    text = " ".join(f"w{i}" for i in range(12))
    dv = embed_document(emb, text)
    assert dv.n_chunks == 3  # 12 tokens in chunks of 5
    rows = [row for batch in seen for row in batch]
    for row in rows:
        assert list(row[: 1 + len(prefix_ids)]) == [1, *prefix_ids]
        assert len([t for t in row if t != 0]) <= 8
    assert np.isclose(np.linalg.norm(dv.vector), 1.0)


def test_local_embedder_batches_bound_the_memory():
    emb, seen = local_embedder()
    embed_document(emb, " ".join(f"w{i}" for i in range(30)))  # 5 chunks of 6 tokens, batches of 2
    assert [len(b) for b in seen] == [2, 2, 1]


def test_unknown_pooling_is_refused():
    with pytest.raises(ValueError):
        LocalTransformerEmbedder("x", "fake", pooling="max")


# --- Hosted embedder (SPEC-M1-04) ------------------------------------------------------------


class FakeTransport:
    def __init__(self, statuses=()):
        self.statuses = list(statuses)
        self.payloads = []

    def __call__(self, url, headers, payload, timeout):
        self.payloads.append(payload)
        if self.statuses:
            status = self.statuses.pop(0)
            if status != 200:
                return status, "busy"
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(payload["input"])]
        return 200, {"data": data[::-1], "usage": {"prompt_tokens": 7}}


def hosted_embedder(transport, **provider):
    model = {"model_id": "intfloat/multilingual-e5-large", "prefix": "passage: ", "input_limit": 12,
             "prices_per_mtok": {"input": 2.0}}
    settings = {"url": "https://example.invalid/v1/embeddings", "margin_tokens": 1, "max_chunks_per_request": 2, **provider}
    return hosted.HostedEmbedder("e5", model, "someprovider", settings, max_retries=3, transport=transport,
                                 tokenizer=FakeTokenizer(), sleep=lambda s: None)


def test_hosted_embedder_sends_prefixed_chunk_texts_and_names_the_provider():
    transport = FakeTransport()
    emb = hosted_embedder(transport)
    text = " ".join(f"w{i}" for i in range(20))
    dv = embed_document(emb, text)
    sent = [t for p in transport.payloads for t in p["input"]]
    assert all(t.startswith("passage: ") for t in sent)
    assert "".join(t[len("passage: "):] for t in sent) == text
    assert emb.provenance()["provider"] == "someprovider"
    assert emb.provenance()["location"] == "hosted"
    assert dv.n_tokens == 20
    assert emb.tokens_billed == 7 * len(transport.payloads)


def test_hosted_limit_is_the_tighter_one():
    emb = hosted_embedder(FakeTransport(), max_input_tokens=10)
    assert emb.input_limit == 10
    assert emb.chunk_budget() == 10 - 2 - 1 - 1  # specials, "passage:", margin


def test_hosted_embedder_retries_temporary_failures():
    transport = FakeTransport([429, 503, 200])
    emb = hosted_embedder(transport)
    embed_document(emb, "egy kettő")
    assert len(transport.payloads) == 3


def test_hosted_embedder_gives_up_on_permanent_errors():
    emb = hosted_embedder(FakeTransport([400]))
    with pytest.raises(hosted.HostedError, match="400"):
        embed_document(emb, "egy kettő")
    emb = hosted_embedder(FakeTransport([503] * 10))
    with pytest.raises(hosted.HostedError, match="gave up"):
        embed_document(emb, "egy kettő")


def test_api_key_comes_from_the_environment(monkeypatch):
    emb = hosted_embedder(FakeTransport(), api_key_env="GRANTRISK_TEST_KEY")
    monkeypatch.delenv("GRANTRISK_TEST_KEY", raising=False)
    with pytest.raises(hosted.HostedError, match="GRANTRISK_TEST_KEY"):
        embed_document(emb, "egy")
    monkeypatch.setenv("GRANTRISK_TEST_KEY", "k")
    assert embed_document(emb, "egy").n_chunks == 1


def test_a_hosted_model_without_provider_is_refused():
    config = {"represent": {"models": {"e5": {"model_id": "x", "location": "hosted", "provider": None}}}}
    with pytest.raises(embeddings.ProviderNotConfigured, match="provider is not set"):
        embeddings.create(config, "e5")
    config["represent"]["models"]["e5"]["provider"] = "nowhere"
    with pytest.raises(embeddings.ProviderNotConfigured, match="nowhere"):
        embeddings.check_configured(config, "e5")
    # the same weights locally, for the equivalence check, need no provider
    assert embeddings.create(config, "e5", force_local=True).location == "local"


def test_cost():
    assert hosted.cost_usd(2_000_000, {"input": 0.5}) == 1.0
    assert hosted.cost_usd(10, {"input": None}) is None


# --- TF-IDF (SPEC-M1-05) ------------------------------------------------------------------


def test_stop_words_are_cleaned():
    """ISS-43: no two-word entry, no duplicates."""
    assert all(" " not in w for w in tfidf.HU_STOP_WORDS)
    assert len(set(tfidf.HU_STOP_WORDS)) == len(tfidf.HU_STOP_WORDS)
    assert "az" in tfidf.HU_STOP_WORDS and "az a" not in tfidf.HU_STOP_WORDS


def test_tfidf_settings_and_hungarian_tokens():
    s = tfidf.settings({"min_df": 1})
    assert s["min_df"] == 1 and s["max_df"] == 0.9 and s["sublinear_tf"] and s["max_features"] == 200_000
    vec = tfidf.vectorizer({"min_df": 1, "max_df": 1.0, "ngram_range": [1, 1]})
    vec.fit(["Az őrzés és a biztosíték", "ŐRZÉS díja"])
    assert set(vec.vocabulary_) == {"őrzés", "biztosíték", "díja"}  # lower case, accents kept, stop words out


def test_special_tokens_are_found_by_probing_the_tokenizer():
    assert embeddings.special_tokens(FakeTokenizer()) == ([1], [2])


def test_default_configuration_is_consistent():
    """Every configured representation has a model or is tfidf; a named provider is described."""
    from grantrisk import config

    values = config.load().values
    for key in values["represent"]["representations"]:
        if key == "tfidf":
            continue
        m = embeddings.model_settings(values, key)
        assert m["pooling"] in embeddings.POOLINGS
        if m["location"] == "hosted" and m.get("provider"):
            embeddings.check_configured(values, key)
    assert values["represent"]["models"]["e5"]["prefix"] == "passage: "
