-- E1 output (Spec_E1_ValidateExtraction.md §2.2). A source is regex, llm:<model id> or old_regex
-- (the old code line's regex points, the baseline of SPEC-E1-02). Rates are NULL where their
-- denominator is 0; each *_low / *_high pair is a 95% Wilson interval (SPEC-E1-05).
CREATE TABLE extraction_agreements (
    run_id                TEXT NOT NULL REFERENCES runs (run_id),
    source                TEXT NOT NULL,
    factor                TEXT NOT NULL,      -- a factor, or 'all' for every factor together
    subset                TEXT NOT NULL CHECK (subset IN ('all', '2021-2027', 'other')),  -- by period
    agree                 INTEGER NOT NULL,
    disagree              INTEGER NOT NULL,
    not_found             INTEGER NOT NULL,
    not_comparable        INTEGER NOT NULL,
    agreement             REAL,               -- agree / (agree + disagree + not_found)
    agreement_low         REAL,
    agreement_high        REAL,
    agreement_found       REAL,               -- agree / (agree + disagree)
    agreement_found_low   REAL,
    agreement_found_high  REAL,
    coverage              REAL,               -- (agree + disagree) / (agree + disagree + not_found)
    mean_abs_diff         REAL,               -- mean |extracted - gold| in points, where both have points
    confusion_json        TEXT NOT NULL,      -- 4 × 4: rows gold points 0-3, columns extracted points 0-3
    PRIMARY KEY (run_id, source, factor, subset)
);

-- The detail table: one row per gold document, factor and source (the material of E3's error analysis).
CREATE TABLE extraction_details (
    run_id         TEXT NOT NULL REFERENCES runs (run_id),
    source         TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    factor         TEXT NOT NULL,
    gold_points    INTEGER,                   -- NULL where the expert did not score ("-")
    value_json     TEXT,                      -- the extracted value as stored by L1
    points         INTEGER,
    points_origin  TEXT CHECK (points_origin IN ('band', 'top_rule', 'given')),  -- given: the baseline's points
    status         TEXT,                      -- the observation's status; NULL if the document is missing from the run
    category       TEXT NOT NULL CHECK (category IN ('agree', 'disagree', 'not_found', 'not_comparable')),
    note           TEXT,                      -- why a value gave no points, e.g. an empty activity list (DEC-33)
    evidence       TEXT,
    evidence_page  INTEGER,
    warnings_json  TEXT,
    PRIMARY KEY (run_id, source, doc_id, factor)
);

-- The risk label of each gold document computed from one source's points, the way L3 would
-- (SPEC-E1-04). Source 'gold' is the reference, computed from the expert's points.
CREATE TABLE extraction_labels (
    run_id            TEXT NOT NULL REFERENCES runs (run_id),
    source            TEXT NOT NULL,
    doc_id            TEXT NOT NULL,
    n_determined      INTEGER NOT NULL,
    total_score       REAL NOT NULL,
    total_exact       TEXT NOT NULL,
    normalised_score  REAL NOT NULL,
    normalised_exact  TEXT NOT NULL,
    fixed_label       TEXT NOT NULL CHECK (fixed_label IN ('low', 'medium', 'high')),
    tercile_label     TEXT NOT NULL CHECK (tercile_label IN ('low', 'medium', 'high')),
    PRIMARY KEY (run_id, source, doc_id)
);

-- Label-level agreement of each source with the reference (SPEC-E1-04).
CREATE TABLE extraction_label_agreements (
    run_id            TEXT NOT NULL REFERENCES runs (run_id),
    source            TEXT NOT NULL,
    label_kind        TEXT NOT NULL CHECK (label_kind IN ('fixed', 'tercile')),
    n_documents       INTEGER NOT NULL,
    agree             INTEGER NOT NULL,
    agreement         REAL,
    agreement_low     REAL,
    agreement_high    REAL,
    kappa_quadratic   REAL,                   -- NULL where undefined (no disagreement is possible by chance)
    error_sizes_json  TEXT NOT NULL,          -- counts of source minus reference (low 0, medium 1, high 2), -2 … +2
    mean_total_diff   REAL NOT NULL,          -- mean of the source's total minus the reference total
    PRIMARY KEY (run_id, source, label_kind)
);
