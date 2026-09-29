"""Chunking, pooling and the embedders of M1 (SPEC-M1-02, -03, -04).

An embedder turns the plain text of a document into chunk vectors. ``embed_document``
does the rest the same way for every model: chunks cut between tokens of the
model's own tokenizer, and a token-weighted mean of the chunk vectors, normalised to
unit length. ``LocalTransformerEmbedder`` runs a model on this machine;
``hosted.HostedEmbedder`` sends the chunks to a provider (DEC-24).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

POOLINGS = ("mean", "cls", "last_token")
WEIGHTING = "token_count"


class ProviderNotConfigured(ValueError):
    """A hosted representation is requested, but its provider is not in the configuration."""


@dataclass(frozen=True)
class Chunk:
    """Consecutive content tokens of the text; ``start``/``end`` are character offsets, if known."""

    ids: tuple[int, ...]
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class DocumentVector:
    vector: np.ndarray
    n_chunks: int
    n_tokens: int


def plan_chunks(
    ids: Sequence[int], offsets: Sequence[tuple[int, int]] | None, text_length: int, budget: int
) -> list[Chunk]:
    """SPEC-M1-02: split the token sequence into consecutive chunks of at most ``budget`` tokens.

    The split falls between tokens; nothing is decoded or encoded again. With the
    tokenizer's offsets, each chunk also gets its character span, and the spans cover
    the whole text: a chunk runs from its first token to the next chunk's first token.
    """
    if budget < 1:
        raise ValueError(f"no room for content tokens (budget {budget})")
    starts = list(range(0, len(ids), budget)) or [0]
    chunks = []
    for i, first in enumerate(starts):
        piece = tuple(ids[first : first + budget])
        if offsets is None:
            chunks.append(Chunk(piece))
            continue
        start = 0 if i == 0 else offsets[first][0]
        end = offsets[starts[i + 1]][0] if i + 1 < len(starts) else text_length
        chunks.append(Chunk(piece, start, end))
    return chunks


def pool_tokens(hidden: np.ndarray, mask: np.ndarray, pooling: str) -> np.ndarray:
    """One vector per sequence from token vectors (batch × tokens × dim) and the attention mask."""
    mask = mask.astype(hidden.dtype)
    if pooling == "mean":  # the attention-masked mean (huBERT, e5)
        return (hidden * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    if pooling == "cls":  # the first token (bge-m3)
        return hidden[:, 0]
    if pooling == "last_token":  # the last real token, with right padding (Qwen3-Embedding)
        last = mask.sum(axis=1).astype(int) - 1
        return hidden[np.arange(hidden.shape[0]), np.maximum(last, 0)]
    raise ValueError(f"unknown pooling {pooling!r}")


def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm > 0, v / np.where(norm > 0, norm, 1), v)


def pool_chunks(vectors: np.ndarray, token_counts: Sequence[int]) -> np.ndarray:
    """SPEC-M1-03: the mean of the chunk vectors weighted by their token counts, at unit length."""
    weights = np.asarray(token_counts, dtype=float)
    if weights.sum() == 0:  # an empty text: one empty chunk
        weights = np.ones_like(weights)
    return unit((np.asarray(vectors, dtype=float) * weights[:, None]).sum(axis=0) / weights.sum())


class Embedder:
    """What M1 needs from a model: its provenance, its tokenizer and its chunk vectors."""

    key: str
    location: str  # local or hosted
    provider: str

    def provenance(self) -> dict[str, Any]:
        """Everything that determines the vector of a text; hashed into the cache key (SPEC-M1-06)."""
        raise NotImplementedError

    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
        raise NotImplementedError

    def chunk_budget(self) -> int:
        """Content tokens per chunk: the input limit minus special tokens and prefix."""
        raise NotImplementedError

    def embed_chunks(self, text: str, chunks: Sequence[Chunk]) -> np.ndarray:
        raise NotImplementedError

    def resolved_revision(self) -> str | None:
        """The model revision actually used, when the embedder knows it."""
        return None


def embed_document(embedder: Embedder, text: str) -> DocumentVector:
    """Chunk, embed and pool one plain text (SPEC-M1-02, -03)."""
    ids, offsets = embedder.tokenize(text)
    chunks = plan_chunks(ids, offsets, len(text), embedder.chunk_budget())
    vectors = np.asarray(embedder.embed_chunks(text, chunks), dtype=float)
    if vectors.shape[0] != len(chunks):
        raise ValueError(f"{embedder.key}: {vectors.shape[0]} vectors for {len(chunks)} chunks")
    return DocumentVector(pool_chunks(vectors, [len(c.ids) for c in chunks]), len(chunks), len(ids))


class LocalTransformerEmbedder(Embedder):
    """A Hugging Face encoder run on this machine (huBERT; e5 for the equivalence check).

    The tokenizer and the model are loaded on first use. ``tokenizer`` and ``forward``
    can be given instead, for the tests.
    """

    location = "local"
    provider = "local"

    def __init__(
        self,
        key: str,
        model_id: str,
        *,
        revision: str | None = None,
        pooling: str = "mean",
        prefix: str = "",
        input_limit: int = 512,
        normalize_chunks: bool = False,
        batch_size: int = 8,
        tokenizer: Any = None,
        forward: Any = None,
    ) -> None:
        if pooling not in POOLINGS:
            raise ValueError(f"{key}: unknown pooling {pooling!r}")
        self.key, self.model_id, self.revision = key, model_id, revision
        self.pooling, self.prefix, self.input_limit = pooling, prefix, input_limit
        self.normalize_chunks, self.batch_size = normalize_chunks, batch_size
        self._tokenizer, self._forward, self._model = tokenizer, forward, None

    def provenance(self) -> dict[str, Any]:
        return {
            "representation": self.key, "model_id": self.model_id, "revision": self.revision,
            "provider": self.provider, "location": self.location, "pooling": self.pooling, "prefix": self.prefix,
            "input_limit": self.input_limit, "normalize_chunks": self.normalize_chunks, "weighting": WEIGHTING,
        }

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        return self._tokenizer

    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
        return tokenize_with(self.tokenizer, text)

    def _prefix_ids(self) -> list[int]:
        return tokenize_with(self.tokenizer, self.prefix)[0] if self.prefix else []

    def chunk_budget(self) -> int:
        head, tail = special_tokens(self.tokenizer)
        return self.input_limit - len(head) - len(tail) - len(self._prefix_ids())

    def model_inputs(self, chunks: Sequence[Chunk]) -> list[list[int]]:
        """The ids the model sees per chunk: special tokens around the prefix and the chunk."""
        head, tail = special_tokens(self.tokenizer)
        prefix = self._prefix_ids()
        return [head + prefix + list(c.ids) + tail for c in chunks]

    def _load(self) -> None:
        import torch
        from transformers import AutoModel

        self._model = AutoModel.from_pretrained(self.model_id, revision=self.revision)
        self._model.eval()

        def forward(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                out = self._model(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(mask))
            return out.last_hidden_state.float().numpy()

        self._forward = forward

    def resolved_revision(self) -> str | None:
        return getattr(getattr(self._model, "config", None), "_commit_hash", None) or self.revision

    def embed_chunks(self, text: str, chunks: Sequence[Chunk]) -> np.ndarray:
        if self._forward is None:
            self._load()
        inputs = self.model_inputs(chunks)
        pad = self.tokenizer.pad_token_id or 0
        out = []
        for i in range(0, len(inputs), self.batch_size):  # small batches bound the memory (ISS-33)
            batch = inputs[i : i + self.batch_size]
            width = max(len(x) for x in batch)
            ids = np.full((len(batch), width), pad, dtype=np.int64)
            mask = np.zeros((len(batch), width), dtype=np.int64)
            for row, x in enumerate(batch):
                ids[row, : len(x)] = x
                mask[row, : len(x)] = 1
            pooled = pool_tokens(np.asarray(self._forward(ids, mask)), mask, self.pooling)
            out.append(unit(pooled) if self.normalize_chunks else pooled)
        return np.concatenate(out, axis=0)


def special_tokens(tokenizer: Any) -> tuple[list[int], list[int]]:
    """The special tokens the tokenizer puts before and after a single sequence, e.g. [CLS] … [SEP].

    Found by encoding one word with and without them, which works for every tokenizer.
    """
    plain = tokenizer("a", add_special_tokens=False)["input_ids"]
    full = tokenizer("a", add_special_tokens=True)["input_ids"]
    for i in range(len(full) - len(plain) + 1):
        if full[i : i + len(plain)] == plain:
            return list(full[:i]), list(full[i + len(plain) :])
    raise ValueError("cannot locate the special tokens of the tokenizer")


def tokenize_with(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
    """Token ids without special tokens, and character offsets where the tokenizer gives them."""
    fast = getattr(tokenizer, "is_fast", False)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=fast, verbose=False)
    offsets = [tuple(o) for o in enc["offset_mapping"]] if fast else None
    return list(enc["input_ids"]), offsets


# --- Configuration --------------------------------------------------------------------------


def model_settings(config_values: Mapping[str, Any], key: str) -> dict[str, Any]:
    models = (config_values.get("represent") or {}).get("models") or {}
    if key not in models:
        raise ValueError(f"representation {key!r} is not in represent.models")
    return dict(models[key])


def check_configured(config_values: Mapping[str, Any], key: str) -> None:
    """Raise ProviderNotConfigured if a hosted representation has no provider (SPEC-M1-04)."""
    m = model_settings(config_values, key)
    if m.get("location", "local") != "hosted":
        return
    provider = m.get("provider")
    providers = (config_values.get("represent") or {}).get("providers") or {}
    if not provider:
        raise ProviderNotConfigured(
            f"{key} is computed by hosted inference (DEC-24), but represent.models.{key}.provider is not set;"
            " choose a provider and describe it under represent.providers"
        )
    if provider not in providers:
        raise ProviderNotConfigured(f"{key}: provider {provider!r} is not described under represent.providers")


def create(config_values: Mapping[str, Any], key: str, *, force_local: bool = False, transport: Any = None) -> Embedder:
    """The embedder of a representation, as configured under ``represent.models``.

    ``force_local`` runs a hosted model's open weights locally, for the equivalence
    check (SPEC-M1-07).
    """
    m = model_settings(config_values, key)
    if m.get("location", "local") == "local" or force_local:
        return LocalTransformerEmbedder(
            key, m["model_id"], revision=m.get("revision"), pooling=m.get("pooling", "mean"),
            prefix=m.get("prefix", ""), input_limit=m.get("input_limit", 512),
            normalize_chunks=m.get("normalize_chunks", False), batch_size=m.get("batch_size", 8),
        )
    check_configured(config_values, key)
    from grantrisk.modelling import hosted

    represent = config_values["represent"]
    return hosted.HostedEmbedder(
        key, m, m["provider"], represent["providers"][m["provider"]],
        max_retries=represent.get("max_retries", 5), transport=transport,
    )
