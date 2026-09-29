-- Runs and lineage (ARC-02).
CREATE TABLE runs (
    run_id        TEXT PRIMARY KEY,
    stage         TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('running', 'complete', 'failed')),
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    code_version  TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    config_hash   TEXT NOT NULL,
    report_path   TEXT,               -- relative to the data root (DEC-22)
    error         TEXT
);

CREATE TABLE run_inputs (
    run_id        TEXT NOT NULL REFERENCES runs (run_id),
    input_run_id  TEXT NOT NULL REFERENCES runs (run_id),
    PRIMARY KEY (run_id, input_run_id)
);

-- C1 output (Spec_C1_Acquire.md §2.2).
CREATE TABLE documents (
    run_id         TEXT NOT NULL REFERENCES runs (run_id),
    doc_id         TEXT NOT NULL,
    call_code      TEXT NOT NULL,
    call_series    TEXT NOT NULL,
    programme      TEXT NOT NULL,
    period         TEXT NOT NULL CHECK (period IN ('2021-2027', '2014-2020', 'RRF', 'VP')),
    doc_type       TEXT NOT NULL CHECK (doc_type IN ('main_call', 'product_description')),
    source         TEXT NOT NULL CHECK (source IN ('scraped', 'manual')),
    source_url     TEXT,
    original_name  TEXT NOT NULL,
    file_path      TEXT NOT NULL,     -- relative to the data root
    sha256         TEXT NOT NULL,
    page_count     INTEGER,
    downloaded_at  TEXT,
    tender_status  TEXT,
    PRIMARY KEY (run_id, doc_id)
);

-- L2 output (Spec_L2_Consolidate.md §2.2).
CREATE TABLE consolidated_factors (
    run_id         TEXT NOT NULL REFERENCES runs (run_id),
    doc_id         TEXT NOT NULL,
    factor         TEXT NOT NULL,
    value_json     TEXT,              -- the value as JSON; NULL if not determined
    points         INTEGER,           -- the expert's points (manual source only)
    chosen_source  TEXT,
    rule           TEXT NOT NULL CHECK (rule IN ('manual', 'preferred', 'fallback', 'none')),
    evidence       TEXT,
    evidence_page  INTEGER,
    PRIMARY KEY (run_id, doc_id, factor)
);

-- L3 output (Spec_L3_ScoreAndLabel.md §2.2). Exact values are stored as fractions, e.g. "7/4".
CREATE TABLE risk_labels (
    run_id            TEXT NOT NULL REFERENCES runs (run_id),
    doc_id            TEXT NOT NULL,
    n_determined      INTEGER NOT NULL,
    low_coverage      INTEGER NOT NULL CHECK (low_coverage IN (0, 1)),
    total_score       REAL NOT NULL,
    total_exact       TEXT NOT NULL,
    normalised_score  REAL NOT NULL,
    normalised_exact  TEXT NOT NULL,
    fixed_label       TEXT NOT NULL CHECK (fixed_label IN ('low', 'medium', 'high')),
    tercile_label     TEXT NOT NULL CHECK (tercile_label IN ('low', 'medium', 'high')),
    PRIMARY KEY (run_id, doc_id)
);

CREATE TABLE risk_label_factors (
    run_id        TEXT NOT NULL,
    doc_id        TEXT NOT NULL,
    factor        TEXT NOT NULL,
    points        REAL NOT NULL,
    points_exact  TEXT NOT NULL,
    origin        TEXT NOT NULL CHECK (origin IN ('band', 'manual', 'mean', 'top_rule')),
    PRIMARY KEY (run_id, doc_id, factor),
    FOREIGN KEY (run_id, doc_id) REFERENCES risk_labels (run_id, doc_id)
);
