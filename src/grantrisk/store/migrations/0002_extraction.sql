-- L1 output (Spec_L1_ExtractFactors.md §2.2). One run holds the observations of one extractor.
CREATE TABLE factor_observations (
    run_id             TEXT NOT NULL REFERENCES runs (run_id),
    doc_id             TEXT NOT NULL,
    factor             TEXT NOT NULL,
    source             TEXT NOT NULL,      -- manual, regex or llm:<model id>
    value_json         TEXT,               -- the value as JSON; NULL if not determined
    points             INTEGER,            -- the expert's points (manual source only)
    evidence           TEXT,
    evidence_page      INTEGER,
    status             TEXT NOT NULL CHECK (status IN ('found', 'not_found', 'ambiguous', 'error')),
    warnings_json      TEXT,
    prompt_version     TEXT,               -- LLM only
    raw_response_path  TEXT,               -- LLM only; relative to the data root
    PRIMARY KEY (run_id, doc_id, factor)
);

-- The gold set, written by the manual import (SPEC-L1-04).
CREATE TABLE gold_records (
    run_id     TEXT NOT NULL REFERENCES runs (run_id),
    gold_set   TEXT NOT NULL,
    gold_name  TEXT NOT NULL,              -- the call as written in the gold file
    doc_id     TEXT NOT NULL,
    call_code  TEXT NOT NULL,
    factor     TEXT NOT NULL,
    points     INTEGER,                    -- NULL where the expert did not score ("-")
    PRIMARY KEY (run_id, doc_id, factor)
);
