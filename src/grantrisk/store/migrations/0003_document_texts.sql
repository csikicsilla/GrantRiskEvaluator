-- C2 output (Spec_C2_Convert.md §2.2). Rows are stored one document at a time (SPEC-C2-10);
-- later stages read only complete runs.
CREATE TABLE document_texts (
    run_id             TEXT NOT NULL REFERENCES runs (run_id),
    doc_id             TEXT NOT NULL,
    status             TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
    error              TEXT,
    markdown           TEXT,
    raw_path           TEXT,               -- the converter's output, relative to the data root
    raw_reused         INTEGER NOT NULL DEFAULT 0 CHECK (raw_reused IN (0, 1)),
    char_count         INTEGER,
    page_count         INTEGER,
    table_count        INTEGER,
    ocr_pages_json     TEXT,
    converter          TEXT NOT NULL,
    converter_version  TEXT NOT NULL,
    settings_hash      TEXT NOT NULL,
    cleaning_version   TEXT NOT NULL,
    warnings_json      TEXT,
    duration_s         REAL,
    PRIMARY KEY (run_id, doc_id)
);
