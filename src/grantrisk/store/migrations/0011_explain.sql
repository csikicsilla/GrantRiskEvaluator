-- E4 (DEC-64, DEC-69): the explanatory analyses of the representations. Exploratory; reported in full.
-- explain_results: per analysis, representation, classifier, target and setting, the mean and the
-- standard deviation (n - 1) of a metric over the folds. The per-fold values are in the run's e4_folds.csv.
CREATE TABLE explain_results (
    run_id          TEXT NOT NULL REFERENCES runs (run_id),
    analysis        TEXT NOT NULL CHECK (analysis IN ('factor_probe', 'combination', 'learning_curve',
                                                      'error_overlap', 'context_length')),
    representation  TEXT NOT NULL,          -- 'tfidf+e5' for a combination; 'majority' for the probe baseline
    classifier      TEXT NOT NULL,          -- logreg, svm, rf, ridge (the score probe) or majority
    target          TEXT NOT NULL,          -- a factor, normalised_score or tercile_label
    setting         TEXT NOT NULL,          -- the training share of the learning curve; '' otherwise
    metric          TEXT NOT NULL,          -- f1_macro, balanced_accuracy or spearman
    mean            REAL,
    sd              REAL,
    n               INTEGER NOT NULL,       -- the folds with a defined value
    PRIMARY KEY (run_id, analysis, representation, classifier, target, setting, metric)
);

-- explain_tests: representation minus reference on the paired folds (corrected resampled t-test with its
-- 95% interval), or McNemar's exact test on the modal predictions (error_overlap); p_holm within the analysis.
CREATE TABLE explain_tests (
    run_id          TEXT NOT NULL REFERENCES runs (run_id),
    analysis        TEXT NOT NULL,
    representation  TEXT NOT NULL,
    reference       TEXT NOT NULL,
    classifier      TEXT NOT NULL,
    target          TEXT NOT NULL,
    setting         TEXT NOT NULL,
    metric          TEXT NOT NULL,
    test            TEXT NOT NULL CHECK (test IN ('corrected_t', 'mcnemar_exact')),
    n               INTEGER NOT NULL,       -- folds, or documents for McNemar
    mean_diff       REAL,                   -- NULL for McNemar
    ci_low          REAL,
    ci_high         REAL,
    statistic       REAL,                   -- t, or the smaller discordant count for McNemar
    p_value         REAL NOT NULL,
    p_holm          REAL NOT NULL,
    details_json    TEXT,
    PRIMARY KEY (run_id, analysis, representation, reference, classifier, target, setting, metric)
);
