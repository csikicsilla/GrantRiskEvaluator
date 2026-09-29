"""Stage C1: acquire (Spec_C1_Acquire.md).

Builds the corpus from the frozen snapshot: one PDF per call, either the main call
or the product description. ``plan`` decides the fate of every input file;
``run`` copies the chosen files into the file store and records the documents.
"""

from __future__ import annotations

import collections
import hashlib
import io
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber
import yaml

from grantrisk.corpus import callcode, doctype, snapshot, versions
from grantrisk.corpus.snapshot import InputFile
from grantrisk.store import files, runs
from grantrisk.store.db import transaction

CANDIDATE_TYPES = (doctype.MAIN_CALL, doctype.PRODUCT_DESCRIPTION)


class ImportConfigError(ValueError):
    """The import configuration contradicts the snapshot."""


@dataclass
class Candidate:
    file: InputFile
    sha256: str
    call_code: str
    programme: str
    period: str
    doc_type: str | None
    typed_by: str  # override, name or first_page

    @property
    def doc_id(self) -> str:
        return self.sha256[:16]


@dataclass
class Chosen:
    candidate: Candidate
    chosen_by: str  # only, version, gold_pin or override
    page_count: int


@dataclass
class Plan:
    documents: list[Chosen] = field(default_factory=list)
    fates: list[dict[str, Any]] = field(default_factory=list)
    county_suffixes: list[dict[str, str]] = field(default_factory=list)
    programme_mismatches: list[dict[str, str]] = field(default_factory=list)
    review_doc_type: list[dict[str, Any]] = field(default_factory=list)
    review_version: list[dict[str, Any]] = field(default_factory=list)
    calls_seen: set[str] = field(default_factory=set)


def _fate(f: InputFile, fate: str, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "location": f.location,
        "source": f.source,
        "original_name": f.original_name,
        "fate": fate,
        "reason": reason,
        **extra,
    }


def first_page_text(data: bytes) -> str | None:
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return pdf.pages[0].extract_text() if pdf.pages else None
    except Exception:  # an unreadable PDF simply gives no text here
        return None


def count_pages(data: bytes) -> int:
    """SPEC-C1-09: the page count; raises if the PDF cannot be opened or has no pages."""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        n = len(pdf.pages)
    if n == 0:
        raise ValueError("the PDF has no pages")
    return n


def load_import_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for key in ("manual_files", "doc_type_overrides", "version_overrides", "gold_pins"):
        cfg[key] = cfg.get(key) or {}
    bad = {k: v for k, v in cfg["doc_type_overrides"].items() if v not in (*CANDIDATE_TYPES, doctype.OTHER)}
    if bad:
        raise ImportConfigError(f"doc_type_overrides with an unknown type: {bad}")
    return cfg


def plan(inputs: Sequence[InputFile], import_cfg: Mapping[str, Any]) -> Plan:
    """Decide the fate of every input file (SPEC-C1-02 … -09)."""
    p = Plan()
    type_overrides = import_cfg["doc_type_overrides"]
    seen: dict[str, Candidate] = {}
    by_call: dict[str, list[Candidate]] = collections.defaultdict(list)
    unknown: list[str] = []

    # SPEC-C1-03: a stable order, scraped files first, so a duplicate keeps its source URL.
    for f in sorted(inputs, key=lambda f: (f.source != "scraped", f.original_name, f.location)):
        data = f.read()
        if not data.startswith(b"%PDF"):
            p.fates.append(_fate(f, "excluded", "not_pdf"))
            continue
        sha = hashlib.sha256(data).hexdigest()
        code, county = callcode.normalise(f.raw_call_code or "")
        if sha in seen:
            p.fates.append(_fate(f, "excluded", "duplicate", call_code=code, duplicate_of=seen[sha].doc_id))
            continue
        try:
            programme, period = callcode.programme_and_period(code)
        except callcode.UnknownProgrammeError as exc:
            unknown.append(f"{f.location}: {exc}")
            continue
        p.calls_seen.add(code)
        if county:
            p.county_suffixes.append({"call_code": code, "suffix": county, "original_name": f.original_name})
        if f.scrape_programme and callcode.programme_family(f.scrape_programme) != programme:
            p.programme_mismatches.append(
                {"call_code": code, "derived": programme, "scrape_log": f.scrape_programme, "location": f.location}
            )

        doc_id = sha[:16]
        if doc_id in type_overrides:
            kind, typed_by = type_overrides[doc_id], "override"
        elif (kind := doctype.by_name(f.original_name)) is not None:
            typed_by = "name"
        else:
            text = first_page_text(data)
            kind, typed_by = (doctype.by_first_page(text) if text else None), "first_page"
        c = Candidate(f, sha, code, programme, period, kind, typed_by)
        seen[sha] = c
        extra = {"call_code": code, "doc_id": doc_id, "typed_by": typed_by}
        if kind == doctype.OTHER:
            p.fates.append(_fate(f, "excluded", "other_type", **extra))
        elif kind is None:
            p.fates.append(_fate(f, "needs_review", "doc_type_unknown", **extra))
            p.review_doc_type.append({"doc_id": doc_id, "original_name": f.original_name, "call_code": code, "location": f.location})
        else:
            by_call[code].append(c)

    if unknown:
        raise callcode.UnknownProgrammeError("unknown programme prefixes:\n  " + "\n  ".join(unknown))
    _choose_versions(p, by_call, import_cfg)
    return p


def _choose_versions(p: Plan, by_call: Mapping[str, list[Candidate]], import_cfg: Mapping[str, Any]) -> None:
    """SPEC-C1-07 … -09: one readable document per call; gold pins first."""
    gold_by_sha = {sha: name for name, sha in import_cfg["gold_pins"].items()}
    version_overrides = import_cfg["version_overrides"]
    chosen_shas: set[str] = set()

    for code in sorted(by_call):
        candidates = sorted(by_call[code], key=lambda c: (c.file.source != "scraped", c.file.original_name))
        while candidates:
            pinned = [c for c in candidates if c.sha256 in gold_by_sha]
            if len(pinned) > 1:
                raise ImportConfigError(f"{code}: more than one gold pin ({[gold_by_sha[c.sha256] for c in pinned]})")
            if pinned:
                choice, chosen_by = pinned[0], "gold_pin"
            elif code in version_overrides:
                matches = [c for c in candidates if c.doc_id == version_overrides[code]]
                if not matches:
                    raise ImportConfigError(f"{code}: version override {version_overrides[code]} is not a candidate")
                choice, chosen_by = matches[0], "override"
            else:
                i = versions.choose([c.file.original_name for c in candidates])
                if i is None:
                    for c in candidates:
                        p.fates.append(_fate(c.file, "needs_review", "version_unclear", call_code=code, doc_id=c.doc_id))
                    p.review_version.append(
                        {
                            "call_code": code,
                            "candidates": [
                                {"doc_id": c.doc_id, "original_name": c.file.original_name,
                                 "marker": str(versions.marker(c.file.original_name))}
                                for c in candidates
                            ],
                        }
                    )
                    break
                choice, chosen_by = candidates[i], ("only" if len(candidates) == 1 else "version")
            try:
                pages = count_pages(choice.file.read())
            except Exception as exc:
                if chosen_by == "gold_pin":
                    raise ImportConfigError(f"{code}: the gold-pinned file cannot be read ({exc})") from exc
                p.fates.append(_fate(choice.file, "excluded", "unreadable", call_code=code, doc_id=choice.doc_id, error=str(exc)))
                candidates = [c for c in candidates if c is not choice]
                continue
            p.documents.append(Chosen(choice, chosen_by, pages))
            chosen_shas.add(choice.sha256)
            p.fates.append(
                _fate(choice.file, "included", None, call_code=code, doc_id=choice.doc_id,
                      typed_by=choice.typed_by, chosen_by=chosen_by)
            )
            other_reason = "not_the_gold_version" if chosen_by == "gold_pin" else "older_version"
            for c in candidates:
                if c is not choice:
                    p.fates.append(_fate(c.file, "excluded", other_reason, call_code=code, doc_id=c.doc_id))
            break

    # SPEC-C1-08: every gold call is in the corpus with the scored document.
    missing = sorted(name for sha, name in gold_by_sha.items() if sha not in chosen_shas)
    if missing:
        raise ImportConfigError(f"gold calls without their pinned document in the corpus: {missing}")


def report(p: Plan, n_inputs: Mapping[str, int]) -> dict[str, Any]:
    docs = [c.candidate for c in p.documents]
    chosen_codes = {c.call_code for c in docs}
    fates = collections.Counter((f["fate"], f["reason"]) for f in p.fates)
    return {
        "inputs": dict(n_inputs),
        "fates": {
            fate: {str(reason): n for (f2, reason), n in sorted(fates.items(), key=str) if f2 == fate}
            for fate in ("included", "excluded", "needs_review")
        },
        "documents": {
            "total": len(docs),
            "by_programme": dict(sorted(collections.Counter(c.programme for c in docs).items())),
            "by_period": dict(sorted(collections.Counter(c.period for c in docs).items())),
            "by_doc_type": dict(sorted(collections.Counter(c.doc_type for c in docs).items())),
            "typed_by": dict(sorted(collections.Counter(c.typed_by for c in docs).items())),
            "chosen_by": dict(sorted(collections.Counter(c.chosen_by for c in p.documents).items())),
        },
        "calls_without_document": sorted(p.calls_seen - chosen_codes),
        "county_suffixes_removed": sorted(p.county_suffixes, key=lambda x: x["call_code"]),
        "programme_mismatches": p.programme_mismatches,
        "review": {"doc_type": p.review_doc_type, "version": p.review_version},
        "files": sorted(p.fates, key=lambda f: (f["source"], f["location"])),
    }


def _store_pdf(data_root: Path, chosen: Chosen, created: list[str]) -> str:
    """SPEC-C1-10: copy the file to pdf/<call code>_<original name>; never overwrite other content."""
    c = chosen.candidate
    rel = f"pdf/{c.call_code.replace('/', '_')}_{c.file.original_name}"
    dest = data_root / rel
    data = c.file.read()
    if dest.exists():
        if hashlib.sha256(dest.read_bytes()).hexdigest() != c.sha256:
            raise FileExistsError(f"{rel} exists with different content")
        return rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    created.append(rel)
    return rel


def _insert(conn: sqlite3.Connection, run_id: str, chosen: Chosen, rel: str, prov: Mapping[str, tuple]) -> None:
    c = chosen.candidate
    url, downloaded_at = prov.get(c.sha256, (None, None))
    conn.execute(
        "INSERT INTO documents (run_id, doc_id, call_code, call_series, programme, period, doc_type, source,"
        " source_url, original_name, file_path, sha256, page_count, downloaded_at, tender_status)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id, c.doc_id, c.call_code, callcode.series(c.call_code), c.programme, c.period, c.doc_type,
            c.file.source, url if c.file.source == "scraped" else None, c.file.original_name, rel, c.sha256,
            chosen.page_count, downloaded_at, c.file.tender_status,
        ),
    )


def run(conn: sqlite3.Connection, config_values: Mapping[str, Any], data_root: Path, sources: Mapping[str, Path],
        import_config_path: Path) -> str:
    """Build the corpus from the snapshot. Returns the new run id.

    On any error the run is marked failed, and it leaves neither rows nor files behind.
    """
    import_cfg = load_import_config(import_config_path)
    run_id = runs.start(conn, "C1", {**config_values, "import_config": import_cfg})
    created: list[str] = []
    try:
        inputs = snapshot.scraped_files(sources["scraped_pdfs"], sources["scrape_log"])
        inputs += snapshot.manual_files(sources["manual_zip"], import_cfg["manual_files"])
        p = plan(inputs, import_cfg)
        prov = snapshot.provenance(sources["old_database"])
        paths = {c.candidate.sha256: _store_pdf(data_root, c, created) for c in p.documents}
        n_inputs = collections.Counter(f.source for f in inputs)
        rep = report(p, n_inputs)
        report_path = files.write_json(data_root, f"reports/{run_id}/c1_report.json", rep)
        created.append(report_path)
        with transaction(conn):
            for c in p.documents:
                _insert(conn, run_id, c, paths[c.candidate.sha256], prov)
            runs.complete(conn, run_id, report_path)
    except Exception as exc:
        for rel in created:
            files.remove(data_root, rel)
        runs.fail(conn, run_id, f"{type(exc).__name__}: {exc}")
        raise
    return run_id
