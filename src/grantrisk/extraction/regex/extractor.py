"""The regex extractor for one document (SPEC-L1-01, -02, -05).

Runs the rule of every factor, checks each value against the factor's domain, and
cuts a verbatim evidence quote from the Markdown, with its page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from grantrisk.extraction import evidence
from grantrisk.extraction.regex.rules import RULES, Finding
from grantrisk.extraction.regex.text import Document
from grantrisk.labelling import scoring
from grantrisk.labelling.scoring import FACTORS


@dataclass
class Observation:
    value: Any = None
    evidence: str = ""
    evidence_page: int | None = None
    status: str = "not_found"  # found, not_found, ambiguous or error
    warnings: list[str] = field(default_factory=list)


def _plain(value: Any) -> Any:
    """A whole percentage as an int (100.0 → 100), so values read like the text."""
    return int(value) if isinstance(value, float) and value.is_integer() else value


def _observe(doc: Document, factor: str, finding: Finding) -> Observation:
    finding.value = _plain(finding.value)
    if finding.value is None:
        status = finding.status if finding.status in ("not_found", "ambiguous") else "not_found"
        return Observation(status=status, warnings=list(finding.warnings))
    error = scoring.domain_error(factor, finding.value)
    if error:
        return Observation(status="error", warnings=[f"out_of_domain: {finding.value!r}", *finding.warnings])
    if finding.span is None:  # biztositek without any mention: 0 by definition, nothing to quote
        return Observation(value=finding.value, status="found", warnings=list(finding.warnings))
    quote, start = doc.quote(*finding.span)
    warnings = list(finding.warnings)
    if not quote or evidence.find_page(doc.page_texts(), quote) is None:
        warnings.append("evidence_not_in_text")
    return Observation(
        value=finding.value, evidence=quote, evidence_page=doc.page_of(start), status="found", warnings=warnings
    )


def extract(markdown: str) -> dict[str, Observation]:
    """SPEC-L1-01: exactly one observation per factor; a rule that fails gives status error."""
    doc = Document(markdown)
    result = {}
    for f in FACTORS:
        try:
            result[f] = _observe(doc, f, RULES[f](doc))
        except Exception as exc:  # a fault in one rule must not lose the other factors
            result[f] = Observation(status="error", warnings=[f"{type(exc).__name__}: {exc}"])
    return result
