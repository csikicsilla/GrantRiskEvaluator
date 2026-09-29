-- M1 represent and M2 train & predict (Spec_M1_Represent.md, Spec_M2_TrainPredict.md).

-- The embedding cache (SPEC-M1-06). One vector per text, model, provider and chunking;
-- the vectors are .npy files in the file store, and every M1 run may reuse them.
CREATE TABLE feature_cache (
    cache_key       TEXT PRIMARY KEY,     -- sha256 of the provenance below
    representation  TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    revision        TEXT,
    provider        TEXT NOT NULL,        -- "local", or the hosted provider (DEC-24)
    location        TEXT NOT NULL CHECK (location IN ('local', 'hosted')),
    params_json     TEXT NOT NULL,        -- the chunking and pooling parameters
    text_hash       TEXT NOT NULL,        -- sha256 of the plain text (SPEC-M1-01)
    vector_path     TEXT NOT NULL,        -- relative to the data root
    dim             INTEGER NOT NULL,
    n_chunks        INTEGER NOT NULL,
    n_tokens        INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    created_by_run  TEXT NOT NULL REFERENCES runs (run_id)
);

-- M1 output (Spec_M1_Represent.md §2.2). For tfidf the feature is the plain text itself,
-- identified by its hash; M2 derives it again from the C2 run and checks the hash.
CREATE TABLE features (
    run_id          TEXT NOT NULL REFERENCES runs (run_id),
    representation  TEXT NOT NULL,
    doc_id          TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('ok', 'missing')),
    text_hash       TEXT NOT NULL,
    cache_key       TEXT REFERENCES feature_cache (cache_key),  -- NULL for tfidf and for a missing vector
    n_chunks        INTEGER,
    n_tokens        INTEGER,
    error           TEXT,
    PRIMARY KEY (run_id, representation, doc_id)
);

-- M2 output (Spec_M2_TrainPredict.md §2.2). One M2 run holds the whole grid.
CREATE TABLE cv_schemes (
    run_id       TEXT NOT NULL REFERENCES runs (run_id),
    scheme       TEXT NOT NULL,           -- stratified, or grouped (SPEC-M2-08)
    splitter     TEXT NOT NULL,
    params_json  TEXT NOT NULL,
    PRIMARY KEY (run_id, scheme)
);

-- FoldAssignment: the test fold of each document in each repeat; it trains on the others.
CREATE TABLE cv_folds (
    run_id  TEXT NOT NULL,
    scheme  TEXT NOT NULL,
    doc_id  TEXT NOT NULL,
    repeat  INTEGER NOT NULL,
    fold    INTEGER NOT NULL,
    PRIMARY KEY (run_id, scheme, doc_id, repeat),
    FOREIGN KEY (run_id, scheme) REFERENCES cv_schemes (run_id, scheme)
);

-- ModelRun: one per scheme, representation and classifier.
CREATE TABLE model_runs (
    run_id          TEXT NOT NULL,
    scheme          TEXT NOT NULL,
    representation  TEXT NOT NULL,
    classifier      TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('complete', 'failed')),
    params_json     TEXT NOT NULL,
    warnings_json   TEXT,
    error           TEXT,
    duration_s      REAL,
    PRIMARY KEY (run_id, scheme, representation, classifier),
    FOREIGN KEY (run_id, scheme) REFERENCES cv_schemes (run_id, scheme)
);

-- Prediction: out of fold, one per model run, document and repeat (SPEC-M2-05).
CREATE TABLE predictions (
    run_id           TEXT NOT NULL,
    scheme           TEXT NOT NULL,
    representation   TEXT NOT NULL,
    classifier       TEXT NOT NULL,
    doc_id           TEXT NOT NULL,
    repeat           INTEGER NOT NULL,
    fold             INTEGER NOT NULL,
    true_label       TEXT NOT NULL CHECK (true_label IN ('low', 'medium', 'high')),
    predicted_label  TEXT NOT NULL CHECK (predicted_label IN ('low', 'medium', 'high')),
    p_low            REAL NOT NULL,
    p_medium         REAL NOT NULL,
    p_high           REAL NOT NULL,
    PRIMARY KEY (run_id, scheme, representation, classifier, doc_id, repeat),
    FOREIGN KEY (run_id, scheme, representation, classifier)
        REFERENCES model_runs (run_id, scheme, representation, classifier)
);

-- The terms with the largest average weight per class, for TF-IDF + logistic regression (SPEC-M2-06).
CREATE TABLE top_terms (
    run_id       TEXT NOT NULL,
    scheme       TEXT NOT NULL,
    class_label  TEXT NOT NULL CHECK (class_label IN ('low', 'medium', 'high')),
    rank         INTEGER NOT NULL,
    term         TEXT NOT NULL,
    mean_weight  REAL NOT NULL,           -- over all folds; a fold without the term counts as 0
    n_folds      INTEGER NOT NULL,        -- the folds whose vocabulary holds the term
    PRIMARY KEY (run_id, scheme, class_label, rank),
    FOREIGN KEY (run_id, scheme) REFERENCES cv_schemes (run_id, scheme)
);
