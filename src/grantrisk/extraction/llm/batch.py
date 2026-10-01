"""The L1 LLM extractor through the Message Batches API (DEC-65): the same requests at half the price.

The requests of a run are sent together as batches and processed by the provider in the background,
usually within an hour and at most within 24 hours. Each answer is stored exactly like an answer of a
direct run (SPEC-L1-08), marked with the batch it came from, so parsing, re-parsing and E1 do not change.

A run goes in rounds:
1. collect the batches the run has submitted and not yet collected;
2. plan what is still missing: every part without a usable stored answer, and a part whose answer
   cannot be used is asked once more (DEC-56);
3. check the ceiling against an upper bound of the cost (counted input tokens and `max_tokens` of output),
   because nothing is billed before a batch runs;
4. submit, and go back to 1.

The run's state (the batches it submitted) is kept in its own file, so a run that is interrupted while
waiting is resumed with `--resume` and collects its batches instead of submitting them again.
"""

from __future__ import annotations

import collections
import datetime
import json
import os
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from grantrisk.extraction import documents as document_set
from grantrisk.extraction.llm import extractor
from grantrisk.extraction.llm.run import (
    MAX_ATTEMPTS,
    BudgetReached,
    Settings,
    _complete_plan,
    _count,
    _insert,
    _limit,
    _md_hash,
    _report,
    _spans,
    _stored,
    extract_document,
    paid_by_run,
    plan_document,
    response_path,
)
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

BATCH_DISCOUNT = 0.5  # the Message Batches API bills every token at half the price
BATCH_SIZE = 100  # requests per batch: about 25 MB of text, well below the API's 100,000 requests and 256 MB
MAX_ROUNDS = 3  # the first submission and at most two follow-ups for failed or unusable answers


def state_path(run_id: str) -> str:
    return f"llm/batches/{run_id}.json"


def _read_state(data_root: Path, run_id: str) -> dict[str, Any]:
    path = data_root / state_path(run_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"batches": []}


def _write_state(data_root: Path, run_id: str, state: Mapping[str, Any]) -> None:
    """The run's own working state; replaced as the run goes on (it is not another run's output)."""
    path = data_root / state_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def custom_id(doc_id: str, part: int, parts: int, attempt: int) -> str:
    return f"{doc_id}-p{part}of{parts}-a{attempt}"


def batch_params(s: Settings, text: str) -> dict[str, Any]:
    """The request of one part, as in a direct run; the body fields that the SDK takes as `extra_body`
    in a direct call are part of the request itself in a batch."""
    params = extractor.build_request(s.model, s.prompt, text, max_tokens=s.max_tokens, temperature=s.temperature,
                                     thinking=s.thinking)
    params.update(params.pop("extra_body", {}))
    return params


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _plan(client: Any, s: Settings, data_root: Path, limit: int, docs: Sequence[tuple[str, str]],
          done: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """The requests still needed, and the documents given up because a part has used every attempt."""
    requests, given_up = [], []
    for doc_id, markdown in docs:
        if doc_id in done:
            continue
        stored = _stored(s, data_root, doc_id, markdown)
        n = _complete_plan(stored)
        if n is not None:
            spans = [tuple(json.loads(stored[n][i][-1].read_text(encoding="utf-8"))["request"]["span"])
                     for i in range(1, n + 1)]
            counts = [None] * n
        else:
            parts, total = plan_document(client, s, markdown, limit)
            n, spans = len(parts), _spans(parts)
            counts = [total] if n == 1 else [None] * n
        doc_requests, stuck = [], False
        for i, span in enumerate(spans, start=1):
            attempts = stored.get(n, {}).get(i, [])
            if attempts:
                try:
                    extractor.parse_response(json.loads(attempts[-1].read_text(encoding="utf-8"))["response"])
                    continue  # a usable answer is stored
                except extractor.ResponseError:
                    pass
            if len(attempts) >= MAX_ATTEMPTS:
                stuck = True
                break
            text = markdown[span[0] : span[1]]
            tokens = counts[i - 1] if counts[i - 1] is not None else _count(client, s, text)
            doc_requests.append({"custom_id": custom_id(doc_id, i, n, len(attempts) + 1), "doc_id": doc_id,
                                 "part": i, "parts": n, "span": list(span), "attempt": len(attempts) + 1,
                                 "input_tokens": tokens})
        if stuck:
            given_up.append(doc_id)
        else:
            requests += doc_requests
    return requests, given_up


def upper_bound_usd(s: Settings, requests: Sequence[Mapping[str, Any]]) -> float:
    """What the requests can cost at most: their counted input, and `max_tokens` of output each."""
    usage = {"input_tokens": sum(r["input_tokens"] for r in requests), "output_tokens": len(requests) * s.max_tokens}
    return extractor.cost(usage, s.prices_per_mtok) * BATCH_DISCOUNT


def _submit(client: Any, s: Settings, data_root: Path, run_id: str, state: dict[str, Any],
            requests: Sequence[Mapping[str, Any]], texts: Mapping[str, str],
            progress: Callable[[str], None] | None) -> None:
    for start in range(0, len(requests), BATCH_SIZE):
        chunk = requests[start : start + BATCH_SIZE]
        batch = client.messages.batches.create(requests=[
            {"custom_id": r["custom_id"],
             "params": batch_params(s, texts[r["doc_id"]][r["span"][0] : r["span"][1]])} for r in chunk])
        state["batches"].append({"batch_id": batch.id, "submitted_at": _now(), "collected": False,
                                 "requests": {r["custom_id"]: {k: r[k] for k in ("doc_id", "part", "parts", "span", "attempt")}
                                              for r in chunk}})
        _write_state(data_root, run_id, state)  # at once: an interrupted run must not submit it again
        if progress:
            progress(f"submitted batch {batch.id} with {len(chunk)} requests")


def _collect(client: Any, s: Settings, data_root: Path, run_id: str, state: dict[str, Any],
             texts: Mapping[str, str], poll_s: float, sleep: Callable[[float], None],
             progress: Callable[[str], None] | None) -> collections.Counter:
    """Wait for every submitted batch to end, and store its answers like the answers of a direct run."""
    outcomes: collections.Counter = collections.Counter()
    for entry in state["batches"]:
        if entry["collected"]:
            continue
        while True:
            batch = client.messages.batches.retrieve(entry["batch_id"])
            if batch.processing_status == "ended":
                break
            if progress:
                counts = batch.request_counts
                progress(f"batch {entry['batch_id']}: {batch.processing_status}, {counts.processing} processing, "
                         f"{counts.succeeded} succeeded, {counts.errored} errored")
            sleep(poll_s)
        for result in client.messages.batches.results(entry["batch_id"]):
            request = entry["requests"].get(result.custom_id)
            outcomes[result.result.type] += 1
            if request is None or result.result.type != "succeeded":
                continue  # errored, canceled or expired: planned again in the next round
            markdown = texts[request["doc_id"]]
            text = markdown[request["span"][0] : request["span"][1]]
            path = response_path(s, request["doc_id"], markdown, request["part"], request["parts"], request["attempt"])
            if (data_root / path).exists():
                continue  # stored before an interruption
            record = {
                "run_id": run_id,  # the run that paid for this answer (DEC-56)
                "service": "batch", "batch_id": entry["batch_id"],
                "request": {
                    "model": s.model, "prompt_version": s.prompt_version, "prompt_sha256": s.prompt_sha256,
                    "request_sha256": s.request_sha256, "doc_id": request["doc_id"],
                    "markdown_sha256": _md_hash(markdown), "part": request["part"], "parts": request["parts"],
                    "span": request["span"], "part_sha256": _md_hash(text), "attempt": request["attempt"],
                    "max_tokens": s.max_tokens, "temperature": s.temperature, "thinking": s.thinking,
                },
                "response": result.result.message.to_dict(),
                "received_at": _now(),
            }
            files.write_json(data_root, path, record)
        entry["collected"] = True
        entry["ended_at"] = _now()
        _write_state(data_root, run_id, state)
        if progress:
            progress(f"collected batch {entry['batch_id']}")
    return outcomes


def run_batch(
    conn: sqlite3.Connection,
    config_values: Mapping[str, Any],
    data_root: Path,
    client: Any,
    s: Settings,
    *,
    c2_run_id: str,
    doc_ids: Sequence[str] | None = None,
    confirmed: bool = False,
    resume_run_id: str | None = None,
    progress: Callable[[str], None] | None = None,
    poll_s: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Extract the factors of a C2 run's documents through batches. Returns the run id.

    The command waits until every batch has ended. A run over the whole corpus needs
    ``confirmed`` (SPEC-L1-11); the ceiling is checked before every submission against an
    upper bound of its cost.
    """
    runs.require_complete(conn, c2_run_id, "C2")
    if doc_ids is None and not confirmed:
        raise ValueError("a run over the whole corpus needs an explicit confirmation (SPEC-L1-11)")
    if s.budget_usd is not None and not s.priced:
        raise ValueError("a budget ceiling needs the prices in extract.llm.prices_per_mtok")
    docs, skipped = document_set.of_c2_run(conn, c2_run_id, doc_ids)
    texts = dict(docs)
    if resume_run_id:
        runs.resume(conn, resume_run_id, "L1")
        run_id = resume_run_id
    else:
        snapshot = {**config_values, "llm_run": {"model": s.model, "prompt_version": s.prompt_version,
                                                  "prompt_sha256": s.prompt_sha256, "request_sha256": s.request_sha256,
                                                  "budget_usd": s.budget_usd, "doc_filter": doc_ids is not None,
                                                  "service": "batch"}}
        run_id = runs.start(conn, "L1", snapshot, inputs=[c2_run_id])
    try:
        state = _read_state(data_root, run_id)
        spent_before = paid_by_run(s, data_root, run_id)[1] or 0.0
        outcomes = _collect(client, s, data_root, run_id, state, texts, poll_s, sleep, progress)
        done = {r[0] for r in conn.execute("SELECT DISTINCT doc_id FROM factor_observations WHERE run_id = ?", (run_id,))}
        limit = _limit(client, s)
        given_up: list[str] = []
        for _ in range(MAX_ROUNDS):
            requests, given_up = _plan(client, s, data_root, limit, docs, done)
            if not requests:
                break
            spent = paid_by_run(s, data_root, run_id)[1] or 0.0
            bound = upper_bound_usd(s, requests)
            if s.budget_usd is not None and spent + bound > s.budget_usd:
                raise BudgetReached(
                    f"{len(requests)} requests can cost up to {bound:.2f} USD (with {s.max_tokens} output tokens each), "
                    f"{spent:.2f} USD is spent: above the ceiling of {s.budget_usd} USD. Nothing was submitted.")
            _submit(client, s, data_root, run_id, state, requests, texts, progress)
            outcomes += _collect(client, s, data_root, run_id, state, texts, poll_s, sleep, progress)
        # Parse every document from its stored answers, as a re-parse does (SPEC-L1-08).
        failed = []
        for doc_id, markdown in docs:
            if doc_id in done:
                continue
            result = extract_document(None, s, data_root, doc_id, markdown, None, run_id=run_id)
            if result["error"]:
                failed.append(result)
            else:
                with transaction(conn):
                    _insert(conn, run_id, s, result)
        with transaction(conn):
            for result in failed:
                _insert(conn, run_id, s, result)
        spent_now = (paid_by_run(s, data_root, run_id)[1] or 0.0) - spent_before
        report = _report(conn, run_id, c2_run_id, s, data_root, spent_now, skipped)
        report["batches"] = [{k: e[k] for k in ("batch_id", "submitted_at", "collected")} | {"requests": len(e["requests"])}
                             for e in state["batches"]]
        report["batch_outcomes"] = dict(sorted(outcomes.items()))
        report["given_up"] = given_up
        report_path = files.write_json(data_root, f"reports/{run_id}/l1_llm_report.json", report)
        with transaction(conn):
            runs.complete(conn, run_id, report_path)
    except BaseException as exc:  # budget, interruption: the state file keeps the submitted batches
        runs.fail(conn, run_id, f"{type(exc).__name__}: {exc}")
        raise
    return run_id
