-- Each resumption of an unfinished run, with the code version that continued it (ARC-02, DEC-57).
-- runs.code_version stays the version the run started with.
CREATE TABLE run_resumes (
    run_id        TEXT NOT NULL REFERENCES runs (run_id),
    resume        INTEGER NOT NULL,       -- 1 for the first resumption, then 2, …
    resumed_at    TEXT NOT NULL,
    code_version  TEXT NOT NULL,
    PRIMARY KEY (run_id, resume)
);
