"""The L1 LLM extractor as a stage run (Spec_L1_ExtractFactors.md §3.4).

Every response is stored in full before it is parsed (SPEC-L1-08). A rerun with the
same model, prompt and text reuses the stored responses, so parsing can be repeated
without calling the API, and an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import collections
import datetime
import hashlib
import json
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grantrisk.extraction import evidence
from grantrisk.extraction.llm import extractor
from grantrisk.extraction.llm.extractor import Observation
from grantrisk.labelling.scoring import FACTORS
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

OUTPUT_TOKENS_ESTIMATE = 2500  # per request, for the cost estimate before a run
SAFETY = 0.9  # parts are planned to use at most 90% of the input limit


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


def _md_hash(markdown: str) -> str:
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()


def response_path(s: Settings, doc_id: str, markdown: str, part: int, parts: int) -> str:
    """Where a response is kept: per model, prompt and text, so a changed text never reuses an old answer."""
    safe_model = re.sub(r"[^A-Za-z0-9.-]+", "_", s.model)
    return f"llm/{safe_model}/{s.prompt_sha256[:12]}/{doc_id}-{_md_hash(markdown)[:12]}-p{part}of{parts}.json"


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


def _stored_set(s: Settings, data_root: Path, doc_id: str, markdown: str) -> list[Path] | None:
    """The stored responses of a document if every part of one plan is there; otherwise None."""
    folder = data_root / Path(response_path(s, doc_id, markdown, 1, 1)).parent
    by_total = collections.defaultdict(dict)
    for p in folder.glob(f"{doc_id}-{_md_hash(markdown)[:12]}-p*of*.json"):
        m = re.search(r"-p(\d+)of(\d+)\.json$", p.name)
        by_total[int(m.group(2))][int(m.group(1))] = p
    for total, found in by_total.items():
        if set(found) == set(range(1, total + 1)):
            return [found[i] for i in range(1, total + 1)]
    return None


def extract_document(client: Any, s: Settings, data_root: Path, doc_id: str, markdown: str, limit: int | None) -> dict[str, Any]:
    """Answer one document: reuse the stored responses or call the API; runs in a worker thread."""
    usage = collections.Counter()
    try:
        stored = _stored_set(s, data_root, doc_id, markdown)
        if stored:
            records = [json.loads(p.read_text(encoding="utf-8")) for p in stored]
            reused = True
        else:
            if limit is None:
                raise extractor.ResponseError("no stored response and no API access (re-parse only)")
            parts, _ = plan_document(client, s, markdown, limit)
            records, reused = [], False
            for i, part in enumerate(parts, start=1):
                path = response_path(s, doc_id, markdown, i, len(parts))
                if (data_root / path).exists():  # stored before an interruption
                    records.append(json.loads((data_root / path).read_text(encoding="utf-8")))
                    continue
                record = {
                    "request": {
                        "model": s.model, "prompt_version": s.prompt_version, "prompt_sha256": s.prompt_sha256,
                        "doc_id": doc_id, "markdown_sha256": _md_hash(markdown), "part": i, "parts": len(parts),
                        "part_sha256": _md_hash(part), "max_tokens": s.max_tokens, "temperature": s.temperature,
                        "thinking": s.thinking,
                    },
                    "response": _call(client, s, part),
                    "received_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                }
                files.write_json(data_root, path, record)
                records.append(record)
        for r in records:
            usage.update({k: v or 0 for k, v in r["response"].get("usage", {}).items() if isinstance(v, int)})
        page_texts = evidence.pages(markdown)
        observed = [extractor.observe(extractor.parse_response(r["response"]), page_texts) for r in records]
        result = observed[0] if len(observed) == 1 else extractor.merge_parts(observed)
        error = None
    except Exception as exc:  # the API's errors after its own retries, refusals, broken answers
        result, reused, error = extractor.error_observations(f"{type(exc).__name__}: {exc}"), False, str(exc)
        records = []
    paths = [response_path(s, doc_id, markdown, r["request"]["part"], r["request"]["parts"]) for r in records]
    return {
        "doc_id": doc_id, "observations": result, "usage": dict(usage), "reused": reused, "error": error,
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
    rows = conn.execute(
        "SELECT doc_id, markdown FROM document_texts WHERE run_id = ? AND status = 'ok' ORDER BY doc_id", (c2_run_id,)
    ).fetchall()
    if doc_ids is not None:
        wanted = set(doc_ids)
        rows = [r for r in rows if r[0] in wanted]
    return [(r[0], r[1]) for r in rows]


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
    usd = extractor.cost({"input_tokens": tokens, "output_tokens": output}, prices) if prices and None not in prices.values() else None
    return {"model": s.model, "documents": len(docs), "requests": requests, "input_tokens": tokens,
            "output_tokens_estimate": output, "usd_estimate": usd, "prices_per_mtok": prices}


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

    A run over the whole corpus needs ``confirmed`` (SPEC-L1-11). When the budget
    ceiling is reached the run stops; the finished documents are kept, and the run can
    be resumed with a higher ceiling.
    """
    runs.require_complete(conn, c2_run_id, "C2")
    if doc_ids is None and not confirmed:
        raise ValueError("a run over the whole corpus needs an explicit confirmation (SPEC-L1-11)")
    if s.budget_usd is not None and (not s.prices_per_mtok or None in s.prices_per_mtok.values()):
        raise ValueError("a budget ceiling needs the prices in extract.llm.prices_per_mtok")
    docs = documents(conn, c2_run_id, doc_ids)
    if resume_run_id:
        row = runs.get(conn, resume_run_id)
        if row is None or row["stage"] != "L1" or row["status"] == "complete":
            raise ValueError(f"run {resume_run_id} is not an unfinished L1 run")
        run_id = resume_run_id
        conn.execute("UPDATE runs SET status = 'running', error = NULL WHERE run_id = ?", (run_id,))
    else:
        snapshot = {**config_values, "llm_run": {"model": s.model, "prompt_version": s.prompt_version,
                                                  "prompt_sha256": s.prompt_sha256, "doc_filter": doc_ids is not None}}
        run_id = runs.start(conn, "L1", snapshot, inputs=[c2_run_id])
    try:
        done = {r[0] for r in conn.execute("SELECT DISTINCT doc_id FROM factor_observations WHERE run_id = ?", (run_id,))}
        todo = [d for d in docs if d[0] not in done]
        limit = _limit(client, s) if client is not None else None
        spent = 0.0  # the ceiling limits what one invocation spends; the report gives the run's total cost
        with ThreadPoolExecutor(max_workers=max(1, s.max_parallel_requests)) as pool:
            pending = list(todo)
            while pending:
                if s.budget_usd is not None and spent >= s.budget_usd:
                    raise BudgetReached(f"budget ceiling of {s.budget_usd} USD reached after {spent:.2f} USD")
                batch, pending = pending[: s.max_parallel_requests], pending[s.max_parallel_requests :]
                for result in pool.map(lambda d: extract_document(client, s, data_root, d[0], d[1], limit), batch):
                    with transaction(conn):
                        _insert(conn, run_id, s, result)
                    if not result["reused"] and s.prices_per_mtok and None not in s.prices_per_mtok.values():
                        spent += extractor.cost(result["usage"], s.prices_per_mtok)
                    if progress:
                        progress(f"{result['doc_id']} {'reused' if result['reused'] else 'answered'}"
                                 f"{' ERROR ' + result['error'] if result['error'] else ''} (spent {spent:.2f} USD)")
        report = _report(conn, run_id, c2_run_id, s, data_root, spent)
        report_path = files.write_json(data_root, f"reports/{run_id}/l1_llm_report.json", report)
        with transaction(conn):
            runs.complete(conn, run_id, report_path)
    except BaseException as exc:  # budget, interruption: the stored documents stay for --resume
        runs.fail(conn, run_id, f"{type(exc).__name__}: {exc}")
        raise
    return run_id


def _stored_usage(conn: sqlite3.Connection, run_id: str, data_root: Path) -> collections.Counter:
    usage = collections.Counter()
    paths = {p for (joined,) in conn.execute(
        "SELECT DISTINCT raw_response_path FROM factor_observations WHERE run_id = ? AND raw_response_path IS NOT NULL",
        (run_id,)) for p in joined.split(";")}
    for p in paths:
        response = json.loads((data_root / p).read_text(encoding="utf-8"))["response"]
        usage.update({k: v or 0 for k, v in response.get("usage", {}).items() if isinstance(v, int)})
    return usage


def _report(conn, run_id, c2_run_id, s: Settings, data_root: Path, spent_now: float) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT factor, status, warnings_json FROM factor_observations WHERE run_id = ?", (run_id,)
    ).fetchall()
    by_factor = {f: collections.Counter() for f in FACTORS}
    warnings = {f: collections.Counter() for f in FACTORS}
    for r in rows:
        by_factor[r["factor"]][r["status"]] += 1
        warnings[r["factor"]].update(w.split(":")[0] for w in json.loads(r["warnings_json"] or "[]"))
    usage = _stored_usage(conn, run_id, data_root)
    prices = s.prices_per_mtok
    total_cost = extractor.cost(usage, prices) if prices and None not in prices.values() else None
    n_docs = len({r[0] for r in conn.execute("SELECT DISTINCT doc_id FROM factor_observations WHERE run_id = ?", (run_id,))})
    return {
        "input_runs": {"C2": c2_run_id},
        "model": s.model,
        "prompt_version": s.prompt_version,
        "prompt_sha256": s.prompt_sha256,
        "request_settings": {"max_tokens": s.max_tokens, "temperature": s.temperature, "thinking": s.thinking},
        "documents": n_docs,
        "status_by_factor": {f: dict(sorted(c.items())) for f, c in by_factor.items()},
        "warnings_by_factor": {f: dict(sorted(c.items())) for f, c in warnings.items() if c},
        "usage": dict(usage),
        "cost_usd": total_cost,
        "cost_usd_this_invocation": round(spent_now, 4),
    }
