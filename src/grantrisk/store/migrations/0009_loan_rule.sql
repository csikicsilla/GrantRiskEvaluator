-- DEC-40: points set by the loan rule have their own origin, 'loan_rule'. SQLite cannot change a
-- CHECK constraint, so both tables are rebuilt with their rows.

CREATE TABLE risk_label_factors_new (
    run_id        TEXT NOT NULL,
    doc_id        TEXT NOT NULL,
    factor        TEXT NOT NULL,
    points        REAL NOT NULL,
    points_exact  TEXT NOT NULL,
    origin        TEXT NOT NULL CHECK (origin IN ('band', 'manual', 'mean', 'top_rule', 'loan_rule')),
    PRIMARY KEY (run_id, doc_id, factor),
    FOREIGN KEY (run_id, doc_id) REFERENCES risk_labels (run_id, doc_id)
);
INSERT INTO risk_label_factors_new SELECT run_id, doc_id, factor, points, points_exact, origin FROM risk_label_factors;
DROP TABLE risk_label_factors;
ALTER TABLE risk_label_factors_new RENAME TO risk_label_factors;

CREATE TABLE extraction_details_new (
    run_id         TEXT NOT NULL REFERENCES runs (run_id),
    source         TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    factor         TEXT NOT NULL,
    gold_points    INTEGER,                   -- NULL where the expert did not score ("-")
    value_json     TEXT,                      -- the extracted value as stored by L1
    points         INTEGER,
    points_origin  TEXT CHECK (points_origin IN ('band', 'top_rule', 'loan_rule', 'given')),  -- given: the baseline's points
    status         TEXT,                      -- the observation's status; NULL if the document is missing from the run
    category       TEXT NOT NULL CHECK (category IN ('agree', 'disagree', 'not_found', 'not_comparable')),
    note           TEXT,                      -- why a value gave no points, e.g. an empty activity list (DEC-33)
    evidence       TEXT,
    evidence_page  INTEGER,
    warnings_json  TEXT,
    PRIMARY KEY (run_id, source, doc_id, factor)
);
INSERT INTO extraction_details_new SELECT run_id, source, doc_id, factor, gold_points, value_json, points, points_origin,
    status, category, note, evidence, evidence_page, warnings_json FROM extraction_details;
DROP TABLE extraction_details;
ALTER TABLE extraction_details_new RENAME TO extraction_details;
