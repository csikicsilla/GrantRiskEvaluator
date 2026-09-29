"""The LLM extractor's logic without the network (SPEC-L1-07, -09, -10).

The request asks for structured JSON: one {value, evidence} object per factor. The
response is parsed, each value is checked against its domain, and each quote against
the text (SPEC-L1-02). Documents too long for one request are split at page
markers, and the parts are merged per factor.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from grantrisk.extraction import evidence
from grantrisk.labelling import scoring
from grantrisk.labelling.scoring import FACTORS, UNTIL_FUNDS_RUN_OUT

ACTIVITIES = list(scoring.ACTIVITY_POINTS)


class ResponseError(ValueError):
    """The response cannot be used: a refusal, a truncated or malformed answer (SPEC-L1-09)."""


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def output_schema() -> dict[str, Any]:
    """The JSON schema of the answer: the domain of each factor (Spec_L3 §2.1) or null."""
    flag = _nullable({"type": "integer", "enum": [0, 1]})
    values = {
        "fin_form": _nullable({"type": "string", "enum": list(scoring.FIN_FORM_POINTS)}),
        "tam_osszeg": _nullable({"type": "integer"}),
        "konzorcium": flag,
        "bead_napok": {"anyOf": [{"type": "integer"}, {"type": "string", "enum": [UNTIL_FUNDS_RUN_OUT]}, {"type": "null"}]},
        "max_tam_int": _nullable({"type": "number"}),
        "eloleg": _nullable({"type": "number"}),
        "idotartam": _nullable({"type": "number"}),
        "tam_tevekenyseg": _nullable({"type": "array", "items": {"type": "string", "enum": ACTIVITIES}}),
        "egysz_elszam": flag,
        "biztositek": flag,
    }
    properties = {
        f: {
            "type": "object",
            "properties": {"value": values[f], "evidence": {"type": "string"}},
            "required": ["value", "evidence"],
            "additionalProperties": False,
        }
        for f in FACTORS
    }
    return {"type": "object", "properties": properties, "required": list(FACTORS), "additionalProperties": False}


def build_request(
    model: str,
    system_prompt: str,
    markdown: str,
    *,
    max_tokens: int,
    temperature: float | None,
    thinking: str | None,
) -> dict[str, Any]:
    """The Messages API parameters: the prompt unchanged, the document as the user message."""
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": markdown}],
        "output_config": {"format": {"type": "json_schema", "schema": output_schema()}},
    }
    if temperature is not None:  # some models reject sampling parameters
        params["temperature"] = temperature
    if thinking == "disabled":
        params["thinking"] = {"type": "disabled"}
    return params


def parse_response(response: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The factor answers of a stored response; raises ResponseError if the response is unusable."""
    stop = response.get("stop_reason")
    if stop == "refusal":
        raise ResponseError(f"refusal: {response.get('stop_details')}")
    if stop == "max_tokens":
        raise ResponseError("the answer was cut off at max_tokens")
    text = next((b.get("text") for b in response.get("content", []) if b.get("type") == "text"), None)
    if text is None:
        raise ResponseError("the response has no text block")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResponseError(f"the answer is not valid JSON: {exc}") from exc
    missing = [f for f in FACTORS if not isinstance(data.get(f), dict) or not {"value", "evidence"} <= set(data[f])]
    if missing:
        raise ResponseError(f"the answer does not match the schema for {missing}")
    return {f: data[f] for f in FACTORS}


@dataclass
class Observation:
    value: Any = None
    evidence: str = ""
    evidence_page: int | None = None
    status: str = "not_found"  # found, not_found, ambiguous or error
    warnings: list[str] = field(default_factory=list)


def observe(answers: dict[str, dict[str, Any]], page_texts: list[tuple[int, str]]) -> dict[str, Observation]:
    """SPEC-L1-01, -02, -09: one observation per factor, checked against its domain and the text."""
    result = {}
    for f in FACTORS:
        value, quote = answers[f]["value"], answers[f]["evidence"] or ""
        if value is None:
            result[f] = Observation()
            continue
        empty_list = f == "tam_tevekenyseg" and value == []  # stored as it is; L2 treats it as not found (DEC-33)
        error = None if empty_list else scoring.domain_error(f, value)
        if error:
            result[f] = Observation(status="error", warnings=[f"out_of_domain: {value!r}"])
            continue
        page = evidence.find_page(page_texts, quote)
        result[f] = Observation(
            value=value, evidence=quote, evidence_page=page, status="found",
            warnings=[] if page is not None else ["evidence_not_in_text"],
        )
    return result


def error_observations(message: str) -> dict[str, Observation]:
    """SPEC-L1-12: a document that failed gets 10 observations with status error."""
    return {f: Observation(status="error", warnings=[message]) for f in FACTORS}


# --- Long documents (SPEC-L1-10) ---------------------------------------------------------


def split_pages(markdown: str) -> list[str]:
    """The Markdown cut before each page marker; each piece starts with its marker."""
    starts = [m.start() for m in evidence.PAGE_MARKER.finditer(markdown)] or [0]
    starts[0] = 0
    return [markdown[a:b] for a, b in zip(starts, starts[1:] + [len(markdown)])]


def plan_parts(markdown: str, count: Callable[[str], int], limit: int) -> list[str]:
    """Consecutive runs of pages whose token count (``count``) stays within ``limit``."""
    if count(markdown) <= limit:
        return [markdown]
    parts: list[str] = []
    current = ""
    for page in split_pages(markdown):
        if current and count(current + page) > limit:
            parts.append(current)
            current = page
        else:
            current += page
    if current:
        parts.append(current)
    too_long = [i for i, p in enumerate(parts) if count(p) > limit]
    if too_long:
        raise ResponseError(f"a single page is longer than the input limit (part {too_long[0] + 1})")
    return parts


def _merge_values(factor: str, values: list[Any]) -> tuple[Any, bool]:
    """The merged value and whether a factor rule decided it; (None, False) means ambiguous."""
    distinct = []
    for v in values:
        if v not in distinct:
            distinct.append(v)
    if len(distinct) == 1:
        return distinct[0], True
    if factor == "bead_napok":
        numbers = [v for v in distinct if v != UNTIL_FUNDS_RUN_OUT]
        return min(numbers), True  # the shortest submission period
    if factor in ("tam_osszeg", "max_tam_int", "eloleg"):
        return max(distinct), True
    if factor == "tam_tevekenyseg":
        return sorted({a for v in distinct for a in v}), True
    if factor == "biztositek":
        return 1, True
    if factor == "fin_form" and set(distinct) == {"grant", "conditional_grant"}:
        return "conditional_grant", True
    return None, False


def merge_parts(parts: Sequence[dict[str, Observation]]) -> dict[str, Observation]:
    """SPEC-L1-10: one observation per factor from the observations of the parts."""
    merged = {}
    for f in FACTORS:
        found = [p[f] for p in parts if p[f].status == "found"]
        if not found:
            status = "error" if any(p[f].status == "error" for p in parts) else "not_found"
            merged[f] = Observation(status=status, warnings=["split_document"])
            continue
        value, decided = _merge_values(f, [o.value for o in found])
        if not decided:
            merged[f] = Observation(status="ambiguous", warnings=["split_document", "parts_disagree"])
            continue
        source = next((o for o in found if o.value == value), found[0])
        merged[f] = Observation(
            value=value, evidence=source.evidence, evidence_page=source.evidence_page, status="found",
            warnings=["split_document", *source.warnings],
        )
    return merged


def cost(usage: dict[str, Any], prices_per_mtok: dict[str, float]) -> float:
    """The cost of one response from its token usage and the configured prices (SPEC-L1-11)."""
    return (
        (usage.get("input_tokens") or 0) * prices_per_mtok["input"]
        + (usage.get("output_tokens") or 0) * prices_per_mtok["output"]
    ) / 1_000_000
