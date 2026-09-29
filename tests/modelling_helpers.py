"""Fakes and synthetic runs for the M1 and M2 tests: no model is downloaded or loaded."""

import random
import re
import zlib

import numpy as np

from grantrisk.modelling.embeddings import Embedder
from grantrisk.store import runs
from grantrisk.store.db import transaction

WORD = re.compile(r"\S+")
CLASS_WORDS = {"low": "alacsony", "medium": "közepes", "high": "magas"}
FILLER = "pályázat támogatás összeg határidő kedvezményezett projekt előleg biztosíték konzorcium".split()


class FakeTokenizer:
    """Word-level ids with character offsets; one start and one end token around a sequence."""

    is_fast = True
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, verbose=False):
        spans = [m.span() for m in WORD.finditer(text)]
        ids = [10 + zlib.crc32(text[a:b].encode("utf-8")) % 997 for a, b in spans]
        out = {"input_ids": [1, *ids, 2] if add_special_tokens else ids}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out



def word_vector(words):
    """A small vector from the class words, so that the classes are learnable."""
    return np.array([sum(w == cw for w in words) for cw in CLASS_WORDS.values()] + [len(words), 1.0])


class FakeEmbedder(Embedder):
    """A deterministic embedder over FakeTokenizer; counts the chunks it embeds."""

    def __init__(self, key="hubert", location="local", provider="local", budget=20, model_id="fake/model"):
        self.key, self.location, self.provider, self.budget, self.model_id = key, location, provider, budget, model_id
        self.tokenizer = FakeTokenizer()
        self.calls = 0
        self.tokens_billed = 0
        self.prices_per_mtok = {"input": 1.0}

    def provenance(self):
        return {"representation": self.key, "model_id": self.model_id, "revision": "r1", "provider": self.provider,
                "location": self.location, "budget": self.budget}

    def tokenize(self, text):
        enc = self.tokenizer(text, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    def chunk_budget(self):
        return self.budget

    def embed_chunks(self, text, chunks):
        self.calls += len(chunks)
        self.tokens_billed += sum(len(c.ids) for c in chunks)
        return np.stack([word_vector(text[c.start : c.end].split()) for c in chunks])


def synthetic_corpus(n_per_class=10, seed=0):
    """(doc_id → Markdown, doc_id → tercile label): each class has its own word."""
    rng = random.Random(seed)
    texts, labels = {}, {}
    i = 0
    for label, word in CLASS_WORDS.items():
        for _ in range(n_per_class):
            doc_id = f"{zlib.crc32(str(i).encode()):08x}{i:08x}"
            body = " ".join(rng.choice(FILLER) for _ in range(30)) + f" {word} {word} " + " ".join(
                rng.choice(FILLER) for _ in range(10))
            texts[doc_id] = f"<!-- page 1 -->\n# Felhívás {i}\n\n{body}\n\n| a | b |\n|---|---|\n| {word} | {i} |\n"
            labels[doc_id] = label
            i += 1
    return texts, labels


def seed_c2(conn, texts, failed=()):
    """A complete C2 run with these Markdown texts; ``failed`` documents have status failed."""
    c2 = runs.start(conn, "C2", {"test": True})
    with transaction(conn):
        for doc_id, markdown in texts.items():
            status = "failed" if doc_id in failed else "ok"
            conn.execute(
                "INSERT INTO document_texts (run_id, doc_id, status, markdown, converter, converter_version,"
                " settings_hash, cleaning_version) VALUES (?, ?, ?, ?, 'fake', '1', 'x', '1')",
                (c2, doc_id, status, None if status == "failed" else markdown),
            )
        runs.complete(conn, c2)
    return c2


def seed_l3(conn, labels, series=None):
    """Complete C1, L2 and L3 runs; the L3 run holds the given tercile labels."""
    c1 = runs.start(conn, "C1", {"test": True})
    with transaction(conn):
        for doc_id in labels:
            conn.execute(
                "INSERT INTO documents (run_id, doc_id, call_code, call_series, programme, period,"
                " doc_type, source, original_name, file_path, sha256)"
                " VALUES (?, ?, ?, ?, 'GINOP_PLUSZ', '2021-2027', 'main_call', 'scraped', 'x.pdf', 'pdf/x.pdf', ?)",
                (c1, doc_id, doc_id, (series or {}).get(doc_id, doc_id), doc_id),
            )
        runs.complete(conn, c1)
    l2 = runs.start(conn, "L2", {"test": True}, inputs=[c1])
    with transaction(conn):
        runs.complete(conn, l2)
    l3 = runs.start(conn, "L3", {"test": True}, inputs=[l2, c1])
    with transaction(conn):
        for doc_id, label in labels.items():
            conn.execute(
                "INSERT INTO risk_labels VALUES (?, ?, 10, 0, 1.0, '1', 5.0, '5', ?, ?)", (l3, doc_id, label, label)
            )
        runs.complete(conn, l3)
    return l3
