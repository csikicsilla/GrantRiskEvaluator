-- E2 output (Spec_E2_EvaluateModels.md §2.2). A model run is (M2 run, scheme, representation,
-- classifier); the M2 run is the E2 run's input. Nothing is averaged across model runs (ISS-28).

-- Metric: one row per model run, scope and name. Scopes: fold:<repeat>.<fold>, repeat:<repeat>,
-- pooled (all repeats together), summary, class:<label>, period:<period> (SPEC-E2-02, -03, -08).
CREATE TABLE metrics (
    run_id          TEXT NOT NULL REFERENCES runs (run_id),
    scheme          TEXT NOT NULL,          -- stratified, or grouped (SPEC-E2-10)
    representation  TEXT NOT NULL,
    classifier      TEXT NOT NULL,
    scope           TEXT NOT NULL,
    name            TEXT NOT NULL,
    value           REAL,                   -- NULL where the metric is undefined; see flag
    flag            TEXT,                   -- why a value is undefined, set by convention, or based on fewer values
    PRIMARY KEY (run_id, scheme, representation, classifier, scope, name)
);

-- The error-size table (SPEC-E2-04): predicted minus tercile label, low = 0, medium = 1, high = 2.
CREATE TABLE error_sizes (
    run_id          TEXT NOT NULL REFERENCES runs (run_id),
    scheme          TEXT NOT NULL,
    representation  TEXT NOT NULL,
    classifier      TEXT NOT NULL,
    error           INTEGER NOT NULL CHECK (error BETWEEN -2 AND 2),
    mean_count      REAL NOT NULL,          -- documents per repeat, averaged over the repeats
    sd_count        REAL,                   -- over the repeats (n - 1)
    share           REAL NOT NULL,          -- mean_count / documents
    PRIMARY KEY (run_id, scheme, representation, classifier, error)
);

-- The ROC curves from the out-of-fold predictions of all repeats together (SPEC-E2-03), at
-- FPR = point / (points - 1); curve is a class (one-vs-rest) or 'macro' (their pointwise mean).
CREATE TABLE roc_curves (
    run_id          TEXT NOT NULL REFERENCES runs (run_id),
    scheme          TEXT NOT NULL,
    representation  TEXT NOT NULL,
    classifier      TEXT NOT NULL,
    curve           TEXT NOT NULL CHECK (curve IN ('low', 'medium', 'high', 'macro')),
    point           INTEGER NOT NULL,
    fpr             REAL NOT NULL,
    tpr             REAL NOT NULL,
    PRIMARY KEY (run_id, scheme, representation, classifier, curve, point)
);

-- The model comparison table (SPEC-E2-05, -06), one row per model run, per scheme.
CREATE TABLE model_comparison (
    run_id              TEXT NOT NULL REFERENCES runs (run_id),
    scheme              TEXT NOT NULL,
    representation      TEXT NOT NULL,
    classifier          TEXT NOT NULL,
    rank                INTEGER NOT NULL,   -- by mean macro-F1 over the folds, from 1; ties share a rank
    f1_macro_mean       REAL,               -- over the folds: the headline figure (INT-EVAL-07)
    f1_macro_sd         REAL,
    accuracy_mean       REAL,               -- over the folds
    roc_auc_ovr_macro_mean REAL,            -- over the folds
    kappa_quadratic_mean   REAL,            -- over the folds
    error2_share        REAL,               -- |error| = 2, averaged over the repeats
    is_baseline         INTEGER NOT NULL CHECK (is_baseline IN (0, 1)),
    is_best             INTEGER NOT NULL CHECK (is_best IN (0, 1)),
    not_above_baseline  INTEGER CHECK (not_above_baseline IN (0, 1)),  -- NULL for the baseline itself
    PRIMARY KEY (run_id, scheme, representation, classifier)
);

-- The grid view of SPEC-E2-06: the mean macro-F1 of each representation (over its classifiers,
-- the baseline excluded) and of each classifier (over the representations).
CREATE TABLE model_grid (
    run_id         TEXT NOT NULL REFERENCES runs (run_id),
    scheme         TEXT NOT NULL,
    axis           TEXT NOT NULL CHECK (axis IN ('representation', 'classifier')),
    key            TEXT NOT NULL,
    f1_macro_mean  REAL,
    n_models       INTEGER NOT NULL,
    PRIMARY KEY (run_id, scheme, axis, key)
);

-- SPEC-E2-09: the best model run against each other model run, on the paired fold differences
-- of macro-F1 (reference minus other), with the corrected resampled t-test.
CREATE TABLE significance (
    run_id                    TEXT NOT NULL REFERENCES runs (run_id),
    scheme                    TEXT NOT NULL,
    representation            TEXT NOT NULL,
    classifier                TEXT NOT NULL,
    reference_representation  TEXT NOT NULL,
    reference_classifier      TEXT NOT NULL,
    n_folds                   INTEGER NOT NULL,
    mean_diff                 REAL NOT NULL,
    sd_diff                   REAL NOT NULL,
    t_stat                    REAL NOT NULL,
    df                        INTEGER NOT NULL,
    p_value                   REAL NOT NULL,
    PRIMARY KEY (run_id, scheme, representation, classifier)
);

-- SPEC-E2-07: the cross-table of the tercile and the fixed label of the L3 run, overall ('all')
-- and per period; the counts per label are its margins.
CREATE TABLE label_distributions (
    run_id         TEXT NOT NULL REFERENCES runs (run_id),
    subset         TEXT NOT NULL CHECK (subset IN ('all', '2021-2027', '2014-2020', 'RRF', 'VP')),
    tercile_label  TEXT NOT NULL CHECK (tercile_label IN ('low', 'medium', 'high')),
    fixed_label    TEXT NOT NULL CHECK (fixed_label IN ('low', 'medium', 'high')),
    n              INTEGER NOT NULL,
    PRIMARY KEY (run_id, subset, tercile_label, fixed_label)
);
