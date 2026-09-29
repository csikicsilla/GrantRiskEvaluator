"""Hosted inference of open-weight embedding models, for the thesis experiments only (DEC-24).

Only corpus documents, which are public calls, are sent. The provider is configuration
(``represent.providers``); an adapter per request format keeps the embedder
independent of it. The chunks are planned with the model's own tokenizer, run
locally; the provider receives their text (SPEC-M1-02, -04).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from grantrisk.modelling.embeddings import WEIGHTING, Chunk, Embedder, special_tokens, tokenize_with, unit

TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
DEFAULT_MARGIN = 8  # tokens kept free per chunk: the provider tokenises the chunk's text again


class HostedError(RuntimeError):
    """A request failed for good: a permanent error, or a temporary one after every retry."""


class TransientError(RuntimeError):
    pass


Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], tuple[int, Any]]


def http_post(url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float) -> tuple[int, Any]:
    """POST a JSON body; return the status and the decoded answer."""
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={**headers, "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")[:500]
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise TransientError(f"{type(exc).__name__}: {exc}") from exc


def _openai_compatible(model: str, texts: Sequence[str]) -> dict[str, Any]:
    return {"model": model, "input": list(texts), "encoding_format": "float"}


def _parse_openai_compatible(answer: Any, n: int) -> tuple[np.ndarray, int | None]:
    data = sorted(answer["data"], key=lambda d: d["index"])
    if len(data) != n:
        raise HostedError(f"{len(data)} embeddings for {n} inputs")
    usage = answer.get("usage") or {}
    return np.asarray([d["embedding"] for d in data], dtype=float), usage.get("prompt_tokens")


# Request formats. DeepInfra and Fireworks, used by the old code line, both accept the OpenAI format.
APIS = {"openai_compatible": (_openai_compatible, _parse_openai_compatible)}


def cost_usd(tokens: int, prices_per_mtok: Mapping[str, float | None] | None) -> float | None:
    price = (prices_per_mtok or {}).get("input")
    return None if price is None else tokens * price / 1_000_000


class HostedEmbedder(Embedder):
    location = "hosted"

    def __init__(
        self,
        key: str,
        model: Mapping[str, Any],
        provider: str,
        provider_settings: Mapping[str, Any],
        *,
        max_retries: int = 5,
        transport: Transport | None = None,
        tokenizer: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        api = provider_settings.get("api", "openai_compatible")
        if api not in APIS:
            raise ValueError(f"{key}: unknown provider API {api!r}")
        self.key, self.provider, self.api = key, provider, api
        self.model_id = model["model_id"]
        self.revision = model.get("revision")
        self.provider_model_id = model.get("provider_model_id") or self.model_id
        self.pooling = model.get("pooling", "provider")
        self.prefix = model.get("prefix", "")
        self.normalize_chunks = model.get("normalize_chunks", True)
        provider_limit = provider_settings.get("max_input_tokens")
        self.input_limit = min(model.get("input_limit", 512), provider_limit or 10**9)  # the tighter limit applies
        self.margin = provider_settings.get("margin_tokens", DEFAULT_MARGIN)
        self.max_chunks_per_request = provider_settings.get("max_chunks_per_request", 16)
        self.url = provider_settings["url"]
        self.api_key_env = provider_settings.get("api_key_env")
        self.timeout = provider_settings.get("timeout_s", 120)
        self.prices_per_mtok = model.get("prices_per_mtok")
        self.max_retries = max_retries
        self._transport = transport or http_post
        self._tokenizer = tokenizer
        self._sleep = sleep
        self.tokens_billed = 0  # over the life of this embedder, for the cost

    def provenance(self) -> dict[str, Any]:
        return {
            "representation": self.key, "model_id": self.model_id, "revision": self.revision,
            "provider": self.provider, "provider_model_id": self.provider_model_id, "location": self.location,
            "pooling": self.pooling, "prefix": self.prefix, "input_limit": self.input_limit,
            "margin_tokens": self.margin, "normalize_chunks": self.normalize_chunks, "weighting": WEIGHTING,
        }

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # only the tokenizer; the model runs at the provider

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        return self._tokenizer

    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
        ids, offsets = tokenize_with(self.tokenizer, text)
        if offsets is None:
            raise ValueError(f"{self.key}: the tokenizer gives no offsets, so the chunks cannot be cut from the text")
        return ids, offsets

    def prefix_tokens(self) -> int:
        return len(tokenize_with(self.tokenizer, self.prefix)[0]) if self.prefix else 0

    def special_count(self) -> int:
        head, tail = special_tokens(self.tokenizer)
        return len(head) + len(tail)

    def chunk_budget(self) -> int:
        special = self.special_count()
        return self.input_limit - special - self.prefix_tokens() - self.margin

    def chunk_texts(self, text: str, chunks: Sequence[Chunk]) -> list[str]:
        return [self.prefix + text[c.start : c.end] for c in chunks]

    def billed_tokens(self, chunks: Sequence[Chunk]) -> int:
        """The tokens sent for these chunks, as counted locally (for the estimate)."""
        special = self.special_count()
        return sum(len(c.ids) + special + self.prefix_tokens() for c in chunks)

    def _headers(self) -> dict[str, str]:
        if not self.api_key_env:
            return {}
        key = os.environ.get(self.api_key_env)
        if not key:
            raise HostedError(f"the environment variable {self.api_key_env} (API key of {self.provider}) is not set")
        return {"Authorization": f"Bearer {key}"}

    def _request(self, texts: Sequence[str]) -> tuple[np.ndarray, int | None]:
        """One request, retried with exponential backoff on temporary failures (as SPEC-L1-12)."""
        build, parse = APIS[self.api]
        payload = build(self.provider_model_id, texts)
        headers = self._headers()
        for attempt in range(self.max_retries + 1):
            try:
                status, answer = self._transport(self.url, headers, payload, self.timeout)
                if status == 200:
                    return parse(answer, len(texts))
                if status not in TRANSIENT_STATUS:
                    raise HostedError(f"{self.provider} answered {status}: {str(answer)[:300]}")
                failure = f"status {status}"
            except TransientError as exc:
                failure = str(exc)
            if attempt < self.max_retries:
                self._sleep(min(2**attempt, 60))
        raise HostedError(f"{self.provider}: gave up after {self.max_retries + 1} attempts ({failure})")

    def embed_chunks(self, text: str, chunks: Sequence[Chunk]) -> np.ndarray:
        texts = self.chunk_texts(text, chunks)
        out = []
        for i in range(0, len(texts), self.max_chunks_per_request):
            batch = texts[i : i + self.max_chunks_per_request]
            vectors, tokens = self._request(batch)
            self.tokens_billed += tokens if tokens is not None else self.billed_tokens(chunks[i : i + len(batch)])
            out.append(unit(vectors) if self.normalize_chunks else vectors)
        return np.concatenate(out, axis=0)
