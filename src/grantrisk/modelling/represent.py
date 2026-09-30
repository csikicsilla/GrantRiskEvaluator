"""Stage M1: represent (Spec_M1_Represent.md).

Every representation reads the same plain text of a C2 run (SPEC-M1-01). The
embeddings are computed once per text, model, provider and chunking, and kept in a
cache that later runs reuse (SPEC-M1-06). For ``tfidf`` the feature is the plain text
itself, because the vectoriser is fitted inside each training fold in M2 (SPEC-M1-05).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from grantrisk.modelling import embeddings, hosted
from grantrisk.modelling.embeddings import Embedder, embed_document, plan_chunks
from grantrisk.modelling.text import PLAIN_TEXT_VERSION, plain_text, text_hash
from grantrisk.modelling.tfidf import settings as tfidf_settings
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

TFIDF = "tfidf"
REPRESENTATIONS = ("tfidf", "hubert", "e5", "bge_m3", "qwen3_8b")


class BudgetReached(RuntimeError):
    pass


def cache_key(text_sha256: str, provenance: Mapping[str, Any]) -> str:
    """SPEC-M1-06: the text hash, the model and revision, the provider and the chunking parameters."""
    canonical = json.dumps({"text_hash": text_sha256, **provenance}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def vector_path(representation: str, key: str) -> str:
    return f"features/{representation}/{key[:2]}/{key}.npy"


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class CacheIndex:
    """The ``feature_cache`` table; new rows are written in batches, not once per document."""

    def __init__(self, conn: sqlite3.Connection, data_root: Path, run_id: str, batch_size: int = 32) -> None:
        self.conn, self.data_root, self.run_id, self.batch_size = conn, data_root, run_id, batch_size
        self.pending: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        row = self.pending.get(key)
        if row is None:
            found = self.conn.execute("SELECT * FROM feature_cache WHERE cache_key = ?", (key,)).fetchone()
            row = dict(found) if found else None
        if row is None or not (self.data_root / row["vector_path"]).exists():
            return None
        return row

    def load(self, key: str) -> np.ndarray:
        row = self.get(key)
        if row is None:
            raise KeyError(key)
        return np.load(self.data_root / row["vector_path"])

    def put(self, key: str, representation: str, provenance: Mapping[str, Any], revision: str | None,
            text_sha256: str, vector: np.ndarray, n_chunks: int, n_tokens: int) -> dict[str, Any]:
        path = vector_path(representation, key)
        target = self.data_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp.npy")
        np.save(tmp, np.asarray(vector, dtype=np.float32))
        tmp.replace(target)  # content-addressed: the same key always holds the same vector
        params = {k: v for k, v in provenance.items() if k not in ("representation", "model_id", "revision", "provider", "location")}
        row = {
            "cache_key": key, "representation": representation, "model_id": provenance["model_id"],
            "revision": revision, "provider": provenance["provider"], "location": provenance["location"],
            "params_json": json.dumps(params, sort_keys=True, ensure_ascii=False), "text_hash": text_sha256,
            "vector_path": path, "dim": int(np.asarray(vector).shape[-1]), "n_chunks": n_chunks, "n_tokens": n_tokens,
            "created_at": _now(), "created_by_run": self.run_id,
        }
        self.pending[key] = row
        if len(self.pending) >= self.batch_size:
            self.flush()
        return row

    def flush(self) -> None:
        if not self.pending:
            return
        columns = list(next(iter(self.pending.values())))
        update = ", ".join(f"{c} = excluded.{c}" for c in columns if c != "cache_key")
        with transaction(self.conn):  # a row is replaced only when its vector file had gone missing
            self.conn.executemany(
                f"INSERT INTO feature_cache ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})"
                f" ON CONFLICT (cache_key) DO UPDATE SET {update}",
                [tuple(r[c] for c in columns) for r in self.pending.values()],
            )
        self.pending.clear()


def representations_of(settings: Mapping[str, Any], requested: Sequence[str] | None) -> list[str]:
    reps = list(requested or settings.get("representations") or REPRESENTATIONS)
    unknown = [r for r in reps if r not in REPRESENTATIONS]
    if unknown:
        raise ValueError(f"unknown representations {unknown}; known: {list(REPRESENTATIONS)}")
    return list(dict.fromkeys(reps))


def document_ids(conn: sqlite3.Connection, c2_run_id: str) -> list[str]:
    """The document set: the documents with text in the C2 run, sorted (edge case: a failed conversion has no features)."""
    return [r[0] for r in conn.execute(
        "SELECT doc_id FROM document_texts WHERE run_id = ? AND status = 'ok' ORDER BY doc_id", (c2_run_id,))]


def document_text(conn: sqlite3.Connection, c2_run_id: str, doc_id: str) -> str:
    row = conn.execute(
        "SELECT markdown FROM document_texts WHERE run_id = ? AND doc_id = ?", (c2_run_id, doc_id)
    ).fetchone()
    return plain_text(row[0] or "")


def text_hashes(conn: sqlite3.Connection, c2_run_id: str, doc_ids: Sequence[str]) -> dict[str, str]:
    return {d: text_hash(document_text(conn, c2_run_id, d)) for d in doc_ids}


def _embedders(config_values: Mapping[str, Any], reps: Sequence[str], given: Mapping[str, Embedder] | None) -> dict[str, Embedder]:
    out = dict(given or {})
    for rep in reps:
        if rep != TFIDF and rep not in out:
            out[rep] = embeddings.create(config_values, rep)  # a hosted model without a provider stops here
    return out


def _cached(index: CacheIndex, emb: Embedder, hashes: Mapping[str, str]) -> set[str]:
    prov = emb.provenance()
    return {d for d, h in hashes.items() if index.get(cache_key(h, prov))}


def estimate(
    conn: sqlite3.Connection,
    config_values: Mapping[str, Any],
    data_root: Path,
    *,
    c2_run_id: str,
    representations: Sequence[str] | None = None,
    embedders: Mapping[str, Embedder] | None = None,
) -> dict[str, Any]:
    """SPEC-M1-04: what a run would compute and send, and its cost. Nothing is sent."""
    runs.require_complete(conn, c2_run_id, "C2")
    reps = representations_of(config_values.get("represent") or {}, representations)
    embs = _embedders(config_values, reps, embedders)
    doc_ids = document_ids(conn, c2_run_id)
    hashes = text_hashes(conn, c2_run_id, doc_ids)
    index = CacheIndex(conn, data_root, "estimate")
    out: dict[str, Any] = {"c2_run": c2_run_id, "documents": len(doc_ids), "representations": {}}
    total_usd = 0.0
    for rep in reps:
        if rep == TFIDF:
            out["representations"][rep] = {"location": "local", "to_compute": 0}
            continue
        emb = embs[rep]
        cached = _cached(index, emb, hashes)
        todo = [d for d in doc_ids if d not in cached]
        chunks = tokens = billed = requests = 0
        for d in todo:
            text = document_text(conn, c2_run_id, d)
            ids, offsets = emb.tokenize(text)
            planned = plan_chunks(ids, offsets, len(text), emb.chunk_budget())
            chunks += len(planned)
            tokens += len(ids)
            if emb.location == "hosted":
                billed += emb.billed_tokens(planned)
                requests += -(-len(planned) // emb.max_chunks_per_request)  # each document is sent on its own
        entry = {"location": emb.location, "provider": emb.provider, "model_id": emb.provenance()["model_id"],
                 "cached": len(cached), "to_compute": len(todo), "chunks": chunks, "tokens": tokens}
        if emb.location == "hosted":
            usd = hosted.cost_usd(billed, emb.prices_per_mtok)
            entry.update({"tokens_sent": billed, "requests": requests, "usd_estimate": usd,
                          "prices_per_mtok": emb.prices_per_mtok})
            total_usd = None if usd is None or total_usd is None else total_usd + usd
        out["representations"][rep] = entry
    out["usd_estimate_hosted"] = total_usd
    return out


def _sample(doc_ids: Sequence[str], n: int) -> list[str]:
    """``n`` documents spread evenly over the sorted document set (deterministic)."""
    if len(doc_ids) <= n:
        return list(doc_ids)
    return [doc_ids[round(i * (len(doc_ids) - 1) / (n - 1))] for i in range(n)] if n > 1 else [doc_ids[0]]


def equivalence_check(
    conn: sqlite3.Connection,
    c2_run_id: str,
    hashes: Mapping[str, str],
    hosted_embedder: Embedder,
    local_embedder: Embedder,
    index: CacheIndex,
    *,
    n_documents: int = 5,
    min_cosine: float = 0.99,
) -> dict[str, Any]:
    """SPEC-M1-07: cosine similarity of the hosted and the locally computed vectors of a few documents.

    The local vectors are computed from the same chunks as the hosted ones (DEC-58): the
    hosted chunks keep a margin below the input limit (DEC-45), and different chunk
    boundaries would add their own difference to that of the weights' precision.
    """
    results = []
    budget = hosted_embedder.chunk_budget()
    hosted_prov = hosted_embedder.provenance()
    local_prov = {**local_embedder.provenance(), "chunk_budget": budget}
    for d in _sample(sorted(hashes), n_documents):
        h = hashes[d]
        hosted_key = cache_key(h, hosted_prov)
        if not index.get(hosted_key):
            results.append({"doc_id": d, "cosine": None, "note": "no hosted vector"})
            continue
        local_key = cache_key(h, local_prov)
        if not index.get(local_key):
            dv = embed_document(local_embedder, document_text(conn, c2_run_id, d), budget)
            index.put(local_key, local_embedder.key, local_prov, local_embedder.resolved_revision(), h,
                      dv.vector, dv.n_chunks, dv.n_tokens)
        a, b = index.load(hosted_key), index.load(local_key)
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        results.append({"doc_id": d, "cosine": round(cosine, 6)})
    compared = [r["cosine"] for r in results if r["cosine"] is not None]
    return {
        "representation": hosted_embedder.key,
        "provider": hosted_embedder.provider,
        "model_id": hosted_prov["model_id"],
        "chunk_budget": budget,
        "min_cosine_required": min_cosine,
        "documents": results,
        "min_cosine": min(compared) if compared else None,
        "passed": bool(compared) and min(compared) >= min_cosine,
    }


def run(
    conn: sqlite3.Connection,
    config_values: Mapping[str, Any],
    data_root: Path,
    *,
    c2_run_id: str,
    representations: Sequence[str] | None = None,
    embedders: Mapping[str, Embedder] | None = None,
    equivalence_embedder: Embedder | None = None,
    confirmed: bool = False,
    resume_run_id: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Compute the representations of a C2 run's documents. Returns the run id.

    Hosted computation needs ``confirmed`` whenever something would be sent, and stops
    at the budget ceiling (SPEC-M1-04). A vector that cannot be computed by the provider
    is recorded as missing, and the representation as incomplete (SPEC-M1-08). The
    finished vectors stay in the cache, so a failed or interrupted run can be resumed.
    """
    runs.require_complete(conn, c2_run_id, "C2")
    settings = config_values.get("represent") or {}
    reps = representations_of(settings, representations)
    embs = _embedders(config_values, reps, embedders)
    doc_ids = document_ids(conn, c2_run_id)
    if not doc_ids:
        raise ValueError(f"C2 run {c2_run_id} has no document with text")
    hashes = text_hashes(conn, c2_run_id, doc_ids)
    budget = settings.get("budget_usd")
    hosted_reps = [r for r in reps if r != TFIDF and embs[r].location == "hosted"]
    probe = CacheIndex(conn, data_root, "probe")
    to_send = {r: len(doc_ids) - len(_cached(probe, embs[r], hashes)) for r in hosted_reps}
    if any(to_send.values()) and not confirmed:
        raise ValueError(
            f"hosted inference would compute {to_send} vectors; run the estimate first and confirm the run (SPEC-M1-04)"
        )
    if budget is not None:
        unpriced = [r for r in hosted_reps if hosted.cost_usd(1, embs[r].prices_per_mtok) is None]
        if unpriced:
            raise ValueError(f"a budget ceiling needs the prices of {unpriced} (represent.models.<key>.prices_per_mtok)")

    if resume_run_id:
        row = runs.get(conn, resume_run_id)
        if row is None or row["stage"] != "M1" or row["status"] == "complete":
            raise ValueError(f"run {resume_run_id} is not an unfinished M1 run")
        run_id = resume_run_id
        runs.resume(conn, run_id, "M1")  # records the code version that continues it (DEC-57)
    else:
        snapshot = {**config_values, "m1_run": {"representations": reps, "plain_text_version": PLAIN_TEXT_VERSION}}
        run_id = runs.start(conn, "M1", snapshot, inputs=[c2_run_id])
    index = CacheIndex(conn, data_root, run_id, settings.get("cache_index_batch", 32))
    rows: list[tuple] = []
    stats: dict[str, dict[str, Any]] = {}
    spent = 0.0
    report_path = None
    try:
        for rep in reps:
            started = time.perf_counter()
            if rep == TFIDF:
                rows += [(run_id, rep, d, "ok", hashes[d], None, None, None, None) for d in doc_ids]
                stats[rep] = {"location": "local", "documents": len(doc_ids), "settings": tfidf_settings(settings.get("tfidf")),
                              "missing": [], "complete": True}
                continue
            emb = embs[rep]
            prov = emb.provenance()
            computed = reused = 0
            missing, chunks, tokens, cost = [], [], [], 0.0
            for d in doc_ids:
                key = cache_key(hashes[d], prov)
                hit = index.get(key)
                if hit is None:
                    if emb.location == "hosted" and budget is not None and spent >= budget:
                        raise BudgetReached(f"budget ceiling of {budget} USD reached after {spent:.2f} USD")
                    billed_before = getattr(emb, "tokens_billed", 0)
                    try:
                        dv = embed_document(emb, document_text(conn, c2_run_id, d))
                    except hosted.HostedError as exc:
                        missing.append({"doc_id": d, "error": str(exc)})
                        rows.append((run_id, rep, d, "missing", hashes[d], None, None, None, str(exc)))
                        continue
                    hit = index.put(key, rep, prov, emb.resolved_revision(), hashes[d], dv.vector, dv.n_chunks, dv.n_tokens)
                    computed += 1
                    if emb.location == "hosted":
                        c = hosted.cost_usd(emb.tokens_billed - billed_before, emb.prices_per_mtok) or 0.0
                        cost, spent = cost + c, spent + c
                    if progress:
                        progress(f"{rep} {d} {dv.n_chunks} chunks")
                else:
                    reused += 1
                rows.append((run_id, rep, d, "ok", hashes[d], key, hit["n_chunks"], hit["n_tokens"], None))
                chunks.append(hit["n_chunks"])
                tokens.append(hit["n_tokens"])
            stats[rep] = {
                "location": emb.location, "provider": emb.provider, "model_id": prov["model_id"],
                "revision": emb.resolved_revision(), "documents": len(doc_ids), "computed": computed, "reused": reused,
                "missing": missing, "complete": not missing, "chunks_total": sum(chunks), "chunks_max": max(chunks, default=0),
                "tokens_total": sum(tokens), "seconds": round(time.perf_counter() - started, 1),
                "cost_usd": round(cost, 4) if emb.location == "hosted" else None,
            }
        index.flush()
        check = settings.get("equivalence_check") or {}
        check_rep = check.get("representation", "e5")
        equivalence = None
        if check.get("enabled", True) and check_rep in hosted_reps:
            local = equivalence_embedder or embeddings.create(config_values, check_rep, force_local=True)
            equivalence = equivalence_check(
                conn, c2_run_id, hashes, embs[check_rep], local, index,
                n_documents=check.get("n_documents", 5), min_cosine=check.get("min_cosine", 0.99),
            )
            index.flush()
        report = {
            "input_runs": {"C2": c2_run_id},
            "plain_text_version": PLAIN_TEXT_VERSION,
            "documents": len(doc_ids),
            "representations": stats,
            "incomplete": sorted(r for r, s in stats.items() if not s["complete"]),
            "cost_usd_this_invocation": round(spent, 4),
            "equivalence_check": equivalence,
            "warnings": [f"{r}: the model revision is not pinned" for r in reps
                         if r != TFIDF and embs[r].provenance().get("revision") is None],
        }
        report_path = files.write_json(data_root, f"reports/{run_id}/m1_report.json", report)
        with transaction(conn):
            conn.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            runs.complete(conn, run_id, report_path)
    except BaseException as exc:  # budget, interruption: the cached vectors stay for --resume
        if report_path:
            files.remove(data_root, report_path)
        try:
            index.flush()
        finally:
            runs.fail(conn, run_id, f"{type(exc).__name__}: {exc}")
        raise
    return run_id


# --- Reading features (for M2) ------------------------------------------------------------


def load_features(
    conn: sqlite3.Connection, data_root: Path, m1_run_id: str, representation: str, doc_ids: Sequence[str]
) -> list[str] | np.ndarray:
    """The features of ``doc_ids`` in this order: plain texts for tfidf, a matrix otherwise.

    Raises ValueError listing the documents without a feature (SPEC-M1-08, SPEC-M2-01).
    """
    rows = {
        r["doc_id"]: r for r in conn.execute(
            "SELECT f.doc_id, f.status, f.text_hash, c.vector_path FROM features f"
            " LEFT JOIN feature_cache c ON c.cache_key = f.cache_key"
            " WHERE f.run_id = ? AND f.representation = ?", (m1_run_id, representation))
    }
    missing = [d for d in doc_ids if d not in rows or rows[d]["status"] != "ok"]
    if missing:
        raise ValueError(f"representation {representation} of M1 run {m1_run_id} lacks {len(missing)} documents: {missing[:20]}")
    if representation == TFIDF:
        c2 = [r for r in runs.inputs(conn, m1_run_id) if runs.get(conn, r)["stage"] == "C2"]
        if len(c2) != 1:
            raise ValueError(f"M1 run {m1_run_id} has no single C2 input")
        texts = [document_text(conn, c2[0], d) for d in doc_ids]
        changed = [d for d, t in zip(doc_ids, texts) if text_hash(t) != rows[d]["text_hash"]]
        if changed:
            raise ValueError(f"the plain text of {changed[:20]} no longer matches M1 run {m1_run_id}; run M1 again")
        return texts
    return np.stack([np.load(data_root / rows[d]["vector_path"]).astype(float) for d in doc_ids])
