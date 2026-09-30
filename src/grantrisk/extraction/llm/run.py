"""The L1 LLM extractor as a stage run (Spec_L1_ExtractFactors.md §3.4).

Every response is stored in full before it is parsed (SPEC-L1-08). A rerun with the
same model, prompt, request settings and text reuses the stored responses, so parsing
can be repeated without calling the API, and an interrupted run resumes where it
stopped. A stored answer that cannot be used is asked again, at most once (DEC-56).
"""

from __future__ import annotations

import collections
import datetime
import hashlib
import json
import re
import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grantrisk.extraction import documents as document_set
from grantrisk.extraction import evidence
from grantrisk.extraction.llm import extractor
from grantrisk.extraction.llm.extractor import Observation
from grantrisk.labelling.scoring import FACTORS
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

OUTPUT_TOKENS_ESTIMATE = 2500  # per request, for the cost estimate before a run
SAFETY = 0.9  # parts are planned to use at most 90% of the input limit
MAX_ATTEMPTS = 2  # a part whose answer cannot be used is asked once more, never more (DEC-56)
_RESPONSE_NAME = re.compile(r"-p(\d+)of(\d+)-a(\d+)\.json$")


class BudgetReached(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    model: str
    prompt_path: Path
    max_tokens: int = 8192
    temperature: float | None = 0
    thinking: str | None = None
    max_parallel_requests: int = 4
    budget_usd: float | None = None
    prices_per_mtok: dict[str, float] | None = None

    @property
    def prompt(self) -> str:
        return self.prompt_path.read_text(encoding="utf-8")

    @property
    def prompt_version(self) -> str:
        m = re.match(r"(P\d+)", self.prompt_path.name)
        return m.group(1) if m else self.prompt_path.stem

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt_path.read_bytes()).hexdigest()

    @property
    def request_sha256(self) -> str:
        """Everything besides the model, the prompt and the text that shapes an answer (DEC-56)."""
        request = {"max_tokens": self.max_tokens, "temperature": self.temperature, "thinking": self.thinking,
                   "output_schema": extractor.output_schema()}
        return hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()

    @property
    def priced(self) -> bool:
        return bool(self.prices_per_mtok) and None not in self.prices_per_mtok.values()

    @property
    def source(self) -> str:
        return f"llm:{self.model}"


def settings_from_config(config_values: Mapping[str, Any], resolve: Callable[[str], Path], model: str | None = None) -> Settings:
    """The LLM settings of the configuration, with per-model overrides (extract.llm.model_settings)."""
    llm = dict(config_values["extract"]["llm"])
    model = model or llm["model"]
    llm.update((llm.get("model_settings") or {}).get(model, {}))
    return Settings(
        model=model,
        prompt_path=resolve(llm["prompt"]),
        max_tokens=llm.get("max_tokens", 8192),
        temperature=llm.get("temperature"),
        thinking=llm.get("thinking"),
        max_parallel_requests=llm.get("max_parallel_requests", 4),
        budget_usd=llm.get("budget_usd"),
        prices_per_mtok=llm.get("prices_per_mtok"),
    )


class Budget:
    """The cost of the run so far, shared by the worker threads (SPEC-L1-11, DEC-56).

    The ceiling is checked before every request; the requests already under way when it
    is reached still finish, so a run may pass the ceiling by at most their cost.
    """

    def __init__(self, limit: float | None = None, prices: Mapping[str, float] | None = None, spent: float = 0.0) -> None:
        self.limit, self.prices, self.spent = limit, prices, spent
        self._lock = threading.Lock()

    @classmethod
    def of(cls, s: Settings, spent: float = 0.0) -> Budget:
        return cls(s.budget_usd, s.prices_per_mtok if s.priced else None, spent)

    def check(self) -> None:
        with self._lock:
            if self.limit is not None and self.spent >= self.limit:
                raise BudgetReached(f"budget ceiling of {self.limit} USD reached after {self.spent:.4f} USD")

    def add(self, usage: Mapping[str, Any]) -> float:
        cost = extractor.cost(usage, self.prices) if self.prices else 0.0
        with self._lock:
            self.spent += cost
        return cost


def _md_hash(markdown: str) -> str:
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()


def response_folder(s: Settings) -> str:
    """One folder per model, prompt and request settings, so a changed setting never reuses an old answer."""
    safe_model = re.sub(r"[^A-Za-z0-9.-]+", "_", s.model)
    return f"llm/{safe_model}/{s.prompt_sha256[:12]}/{s.request_sha256[:12]}"


def response_path(s: Settings, doc_id: str, markdown: str, part: int, parts: int, attempt: int = 1) -> str:
    """Where one answer is kept: per text, part of the plan and attempt."""
    return f"{response_folder(s)}/{doc_id}-{_md_hash(markdown)[:12]}-p{part}of{parts}-a{attempt}.json"


def _limit(client: Any, s: Settings) -> int:
    """The tokens one request may send: the model's input limit minus the answer, with a margin."""
    return int(client.models.retrieve(s.model).max_input_tokens * SAFETY) - s.max_tokens


def _count(client: Any, s: Settings, markdown: str) -> int:
    params = extractor.build_request(s.model, s.prompt, markdown, max_tokens=s.max_tokens, temperature=None, thinking=None)
    return client.messages.count_tokens(model=s.model, system=params["system"], messages=params["messages"]).input_tokens


def plan_document(client: Any, s: Settings, markdown: str, limit: int) -> tuple[list[str], int]:
    """The parts of a document (one unless it is too long) and its token count (SPEC-L1-10)."""
    total = _count(client, s, markdown)
    if total <= limit:
        return [markdown], total
    ratio = total / len(markdown)  # tokens per character, measured on this document
    safety = 1.0
    while True:
        parts = extractor.plan_parts(markdown, lambda t: int(len(t) * ratio / safety), limit)
        if all(_count(client, s, p) <= limit for p in parts):
            return parts, total
        safety *= 0.85


def _call(client: Any, s: Settings, markdown: str) -> dict[str, Any]:
    params = extractor.build_request(
        s.model, s.prompt, markdown, max_tokens=s.max_tokens, temperature=s.temperature, thinking=s.thinking
    )
    return client.messages.create(**params).to_dict()


def _stored(s: Settings, data_root: Path, doc_id: str, markdown: str) -> dict[int, dict[int, list[Path]]]:
    """The stored answers of a document: plan size → part → the attempts, in order."""
    folder = data_root / response_folder(s)
    found: dict[int, dict[int, list[tuple[int, Path]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in folder.glob(f"{doc_id}-{_md_hash(markdown)[:12]}-p*of*-a*.json"):
        m = _RESPONSE_NAME.search(p.name)
        found[int(m.group(2))][int(m.group(1))].append((int(m.group(3)), p))
    return {n: {i: [p for _, p in sorted(a)] for i, a in parts.items()} for n, parts in found.items()}


def _complete_plan(stored: Mapping[int, Mapping[int, list[Path]]]) -> int | None:
    """The size of a plan whose every part has a stored answer, if there is one."""
    complete = [n for n, parts in stored.items() if set(parts) == set(range(1, n + 1))]
    return min(complete) if complete else None


def _answer_part(
    client: Any, s: Settings, data_root: Path, run_id: str | None, budget: Budget, doc_id: str, markdown: str,
    span: tuple[int, int], part: int, parts: int, attempts: Sequence[Path], usage: collections.Counter,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], bool]:
    """(record, answers, asked) for one part: the latest stored attempt if it can be used, otherwise a new one.

    A part is asked at most MAX_ATTEMPTS times in all, over every run; without API access
    (``client`` None) only stored answers are read.
    """
    attempts = list(attempts)
    failure: Exception | None = None
    if attempts:
        record = json.loads(attempts[-1].read_text(encoding="utf-8"))
        try:
            return record, extractor.parse_response(record["response"]), False
        except extractor.ResponseError as exc:
            failure = exc
    text = markdown[span[0] : span[1]]
    while True:
        if client is None:
            raise failure or extractor.ResponseError("no stored response and no API access (re-parse only)")
        if len(attempts) >= MAX_ATTEMPTS:
            raise failure
        budget.check()
        attempt = len(attempts) + 1
        record = {
            "run_id": run_id,  # the run that paid for this answer (DEC-56)
            "request": {
                "model": s.model, "prompt_version": s.prompt_version, "prompt_sha256": s.prompt_sha256,
                "request_sha256": s.request_sha256, "doc_id": doc_id, "markdown_sha256": _md_hash(markdown),
                "part": part, "parts": parts, "span": list(span), "part_sha256": _md_hash(text), "attempt": attempt,
                "max_tokens": s.max_tokens, "temperature": s.temperature, "thinking": s.thinking,
            },
            "response": _call(client, s, text),
            "received_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        path = files.write_json(data_root, response_path(s, doc_id, markdown, part, parts, attempt), record)
        attempts.append(data_root / path)
        new_usage = {k: v or 0 for k, v in record["response"].get("usage", {}).items() if isinstance(v, int)}
        budget.add(new_usage)
        usage.update(new_usage)
        try:
            return record, extractor.parse_response(record["response"]), True
        except extractor.ResponseError as exc:
            failure = exc


def _spans(parts: Sequence[str]) -> list[tuple[int, int]]:
    """The character spans of consecutive parts that together make up the text."""
    out, start = [], 0
    for p in parts:
        out.append((start, start + len(p)))
        start += len(p)
    return out


def extract_document(
    client: Any, s: Settings, data_root: Path, doc_id: str, markdown: str, limit: int | None,
    budget: Budget | None = None, run_id: str | None = None,
) -> dict[str, Any]:
    """Answer one document: reuse the stored answers or call the API; runs in a worker thread.

    ``usage`` counts only the tokens of the answers asked now. BudgetReached stops the run;
    every other failure gives the document 10 error observations.
    """
    budget = budget or Budget()
    usage: collections.Counter = collections.Counter()
    records: list[dict[str, Any]] = []
    asked = False
    try:
        stored = _stored(s, data_root, doc_id, markdown)
        n = _complete_plan(stored)
        if n is not None:  # every part answered before: the plan is read from the stored answers
            spans = [tuple(json.loads(stored[n][i][-1].read_text(encoding="utf-8"))["request"]["span"])
                     for i in range(1, n + 1)]
        elif limit is None:
            raise extractor.ResponseError("no stored response and no API access (re-parse only)")
        else:
            parts, _ = plan_document(client, s, markdown, limit)
            n, spans = len(parts), _spans(parts)
        answers = []
        for i, span in enumerate(spans, start=1):
            record, answer, new = _answer_part(client, s, data_root, run_id, budget, doc_id, markdown, span, i, n,
                                               stored.get(n, {}).get(i, []), usage)
            records.append(record)
            answers.append(answer)
            asked = asked or new
        page_texts = evidence.pages(markdown)
        observed = [extractor.observe(a, page_texts) for a in answers]
        result = observed[0] if len(observed) == 1 else extractor.merge_parts(observed)
        error = None
    except BudgetReached:
        raise
    except Exception as exc:  # the API's errors after its own retries, refusals, broken answers
        result, error = extractor.error_observations(f"{type(exc).__name__}: {exc}"), str(exc)
        records = []
    paths = [response_path(s, doc_id, markdown, r["request"]["part"], r["request"]["parts"], r["request"]["attempt"])
             for r in records]
    return {
        "doc_id": doc_id, "observations": result, "usage": dict(usage), "reused": not asked, "error": error,
        "parts": len(records), "response_paths": paths,
        "served_by": sorted({r["response"].get("model") for r in records if r["response"].get("model")}),
    }


def _insert(conn: sqlite3.Connection, run_id: str, s: Settings, doc: dict[str, Any]) -> None:
    for f, o in doc["observations"].items():
        conn.execute(
            "INSERT INTO factor_observations (run_id, doc_id, factor, source, value_json, evidence, evidence_page,"
            " status, warnings_json, prompt_version, raw_response_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, doc["doc_id"], f, s.source, None if o.value is None else json.dumps(o.value, ensure_ascii=False),
                o.evidence or None, o.evidence_page, o.status, json.dumps(o.warnings), s.prompt_version,
                ";".join(doc["response_paths"]) or None,
            ),
        )


def documents(conn: sqlite3.Connection, c2_run_id: str, doc_ids: Sequence[str] | None) -> list[tuple[str, str]]:
    """(doc_id, Markdown) of the documents with text in a C2 run, optionally filtered (SPEC-L1-03, -14)."""
    return document_set.of_c2_run(conn, c2_run_id, doc_ids)[0]


def estimate(client: Any, s: Settings, docs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """SPEC-L1-11: the expected cost of a run, from the token counts and the configured prices."""
    limit = _limit(client, s)
    prompt_tokens = _count(client, s, ".")
    tokens, requests = 0, 0
    for _, markdown in docs:
        parts, total = plan_document(client, s, markdown, limit)
        tokens += total + (len(parts) - 1) * prompt_tokens  # the prompt is sent again with every further part
        requests += len(parts)
    output = requests * OUTPUT_TOKENS_ESTIMATE
    prices = s.prices_per_mtok
    usd = extractor.cost({"input_tokens": tokens, "output_tokens": output}, prices) if s.priced else None
    return {"model": s.model, "documents": len(docs), "requests": requests, "input_tokens": tokens,
            "output_tokens_estimate": output, "usd_estimate": usd, "prices_per_mtok": prices}


def paid_by_run(s: Settings, data_root: Path, run_id: str) -> tuple[collections.Counter, float | None]:
    """The token usage and cost of every answer this run asked for, found by the run id in the stored answers."""
    usage: collections.Counter = collections.Counter()
    for p in (data_root / response_folder(s)).glob("*.json"):
        record = json.loads(p.read_text(encoding="utf-8"))
        if record.get("run_id") == run_id:
            usage.update({k: v or 0 for k, v in record["response"].get("usage", {}).items() if isinstance(v, int)})
    return usage, (extractor.cost(usage, s.prices_per_mtok) if s.priced else None)


def run(
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
) -> str:
    """Extract the factors of a C2 run's documents with the LLM. Returns the run id.

    A run over the whole corpus needs ``confirmed`` (SPEC-L1-11). The budget ceiling
    covers the whole run, resumed parts included: when it is reached the run stops, the
    finished documents are kept, and the run can be resumed with a higher ceiling. A
    document that failed gets its error observations only when the run completes, so a
    resumed run asks for it again (SPEC-L1-12, DEC-56).
    """
    runs.require_complete(conn, c2_run_id, "C2")
    if doc_ids is None and not confirmed:
        raise ValueError("a run over the whole corpus needs an explicit confirmation (SPEC-L1-11)")
    if s.budget_usd is not None and not s.priced:
        raise ValueError("a budget ceiling needs the prices in extract.llm.prices_per_mtok")
    docs, skipped = document_set.of_c2_run(conn, c2_run_id, doc_ids)
    if resume_run_id:
        runs.resume(conn, resume_run_id, "L1")
        run_id = resume_run_id
    else:
        snapshot = {**config_values, "llm_run": {"model": s.model, "prompt_version": s.prompt_version,
                                                  "prompt_sha256": s.prompt_sha256, "request_sha256": s.request_sha256,
                                                  "doc_filter": doc_ids is not None}}
        run_id = runs.start(conn, "L1", snapshot, inputs=[c2_run_id])
    try:
        done = {r[0] for r in conn.execute("SELECT DISTINCT doc_id FROM factor_observations WHERE run_id = ?", (run_id,))}
        todo = [d for d in docs if d[0] not in done]
        limit = _limit(client, s) if client is not None else None
        budget = Budget.of(s, (paid_by_run(s, data_root, run_id)[1] or 0.0) if resume_run_id else 0.0)
        start_spent = budget.spent
        failed: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, s.max_parallel_requests)) as pool:
            pending = list(todo)
            while pending:
                batch, pending = pending[: s.max_parallel_requests], pending[s.max_parallel_requests :]
                results = pool.map(lambda d: extract_document(client, s, data_root, d[0], d[1], limit, budget, run_id), batch)
                for result in results:
                    if result["error"]:
                        failed.append(result)  # written when the run completes; a resumed run asks again
                    else:
                        with transaction(conn):
                            _insert(conn, run_id, s, result)
                    if progress:
                        progress(f"{result['doc_id']} {'reused' if result['reused'] else 'answered'}"
                                 f"{' ERROR ' + result['error'] if result['error'] else ''}"
                                 f" (spent {budget.spent - start_spent:.2f} USD now, {budget.spent:.2f} USD in the run)")
        with transaction(conn):
            for result in failed:
                _insert(conn, run_id, s, result)
        report = _report(conn, run_id, c2_run_id, s, data_root, budget.spent - start_spent, skipped)
        report_path = files.write_json(data_root, f"reports/{run_id}/l1_llm_report.json", report)
        with transaction(conn):
            runs.complete(conn, run_id, report_path)
    except BaseException as exc:  # budget, interruption: the stored documents stay for --resume
        runs.fail(conn, run_id, f"{type(exc).__name__}: {exc}")
        raise
    return run_id


def _used_responses(conn: sqlite3.Connection, run_id: str, data_root: Path) -> list[dict[str, Any]]:
    paths = {p for (joined,) in conn.execute(
        "SELECT DISTINCT raw_response_path FROM factor_observations WHERE run_id = ? AND raw_response_path IS NOT NULL",
        (run_id,)) for p in joined.split(";")}
    return [json.loads((data_root / p).read_text(encoding="utf-8")) for p in sorted(paths)]


def _report(conn, run_id, c2_run_id, s: Settings, data_root: Path, spent_now: float,
            skipped: list[dict[str, Any]]) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT factor, status, warnings_json FROM factor_observations WHERE run_id = ?", (run_id,)
    ).fetchall()
    by_factor = {f: collections.Counter() for f in FACTORS}
    warnings = {f: collections.Counter() for f in FACTORS}
    for r in rows:
        by_factor[r["factor"]][r["status"]] += 1
        warnings[r["factor"]].update(w.split(":")[0] for w in json.loads(r["warnings_json"] or "[]"))
    used = _used_responses(conn, run_id, data_root)
    usage = collections.Counter()
    for record in used:
        usage.update({k: v or 0 for k, v in record["response"].get("usage", {}).items() if isinstance(v, int)})
    paid_usage, paid_cost = paid_by_run(s, data_root, run_id)
    n_docs = len({r[0] for r in conn.execute("SELECT DISTINCT doc_id FROM factor_observations WHERE run_id = ?", (run_id,))})
    failed = sorted(r[0] for r in conn.execute(
        "SELECT doc_id FROM factor_observations WHERE run_id = ? GROUP BY doc_id HAVING SUM(status != 'error') = 0",
        (run_id,)))
    return {
        "input_runs": {"C2": c2_run_id},
        "model": s.model,
        "models_that_answered": sorted({r["response"].get("model") for r in used if r["response"].get("model")}),
        "prompt_version": s.prompt_version,
        "prompt_sha256": s.prompt_sha256,
        "request_settings": {"max_tokens": s.max_tokens, "temperature": s.temperature, "thinking": s.thinking,
                             "request_sha256": s.request_sha256},
        "documents": n_docs,
        "failed_documents": failed,
        "skipped": skipped,
        "status_by_factor": {f: dict(sorted(c.items())) for f, c in by_factor.items()},
        "warnings_by_factor": {f: dict(sorted(c.items())) for f, c in warnings.items() if c},
        "usage": dict(usage),  # every answer behind the observations, also those an earlier run paid for
        "cost_usd": extractor.cost(usage, s.prices_per_mtok) if s.priced else None,
        "usage_paid_by_this_run": dict(paid_usage),
        "cost_usd_paid_by_this_run": paid_cost,
        "cost_usd_this_invocation": round(spent_now, 4),
    }
