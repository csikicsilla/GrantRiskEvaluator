"""Synthetic C1, L3 and M2 runs for the E2 and E3 tests, written straight into the tables."""

import json
import random

from grantrisk.store import runs
from grantrisk.store.db import transaction

LABELS = ("low", "medium", "high")
PERIOD_OF = {0: "2021-2027", 1: "2014-2020", 2: "RRF", 3: "VP"}


def corpus(n_per_class=10):
    """doc_id → (tercile label, fixed label, period); the periods cycle over all four."""
    docs = {}
    i = 0
    for label in LABELS:
        for _ in range(n_per_class):
            fixed = label if i % 4 else LABELS[(LABELS.index(label) + 1) % 3]
            docs[f"doc{i:03d}"] = (label, fixed, PERIOD_OF[i % 4])
            i += 1
    return docs


def seed_l3(conn, docs):
    """Complete C1, L2 and L3 runs over ``docs`` (doc_id → (tercile, fixed, period))."""
    c1 = runs.start(conn, "C1", {"test": True})
    with transaction(conn):
        for doc_id, (_, _, period) in docs.items():
            conn.execute(
                "INSERT INTO documents (run_id, doc_id, call_code, call_series, programme, period,"
                " doc_type, source, original_name, file_path, sha256)"
                " VALUES (?, ?, ?, ?, 'GINOP', ?, 'main_call', 'scraped', 'x.pdf', 'pdf/x.pdf', ?)",
                (c1, doc_id, f"CODE-{doc_id}", f"SERIES-{doc_id}", period, doc_id),
            )
        runs.complete(conn, c1)
    l2 = runs.start(conn, "L2", {"test": True}, inputs=[c1])
    with transaction(conn):
        runs.complete(conn, l2)
    l3 = runs.start(conn, "L3", {"test": True}, inputs=[l2, c1])
    with transaction(conn):
        for doc_id, (tercile, fixed, _) in docs.items():
            conn.execute("INSERT INTO risk_labels VALUES (?, ?, 10, 0, 1.0, '1', 5.0, '5', ?, ?)",
                         (l3, doc_id, fixed, tercile))
        runs.complete(conn, l3)
    return c1, l3


def folds_of(doc_ids, labels, n_splits=5, n_repeats=5, seed=0):
    """(doc_id, repeat) → fold: a stratified assignment, every document tested once per repeat."""
    rng = random.Random(seed)
    out = {}
    for r in range(n_repeats):
        for label in LABELS:
            members = [d for d in doc_ids if labels[d] == label]
            rng.shuffle(members)
            for i, d in enumerate(members):
                out[(d, r)] = i % n_splits
    return out


def noisy(accuracy, seed):
    """A model that is right with probability ``accuracy``; its probabilities favour its label."""
    rng = random.Random(seed)

    def predict(doc_id, true_label, repeat):
        label = true_label if rng.random() < accuracy else rng.choice([c for c in LABELS if c != true_label])
        probs = [rng.uniform(0.05, 0.3) for _ in LABELS]
        probs[LABELS.index(label)] += 1.0
        total = sum(probs)
        return label, [p / total for p in probs]

    return predict


def majority():
    def predict(doc_id, true_label, repeat):
        return "high", [0.3, 0.3, 0.4]

    return predict


def seed_m2(conn, l3, docs, models, failed=(), scheme="stratified", m2=None, n_repeats=5):
    """A complete M2 run with the given model runs: (representation, classifier) → predict function."""
    labels = {d: t for d, (t, _, _) in docs.items()}
    doc_ids = sorted(docs)
    folds = folds_of(doc_ids, labels, n_repeats=n_repeats)
    new = m2 is None
    if new:
        m2 = runs.start(conn, "M2", {"test": True}, inputs=[l3])
    with transaction(conn):
        conn.execute("INSERT INTO cv_schemes VALUES (?, ?, 'test', '{}')", (m2, scheme))
        conn.executemany("INSERT INTO cv_folds VALUES (?, ?, ?, ?, ?)",
                         [(m2, scheme, d, r, f) for (d, r), f in folds.items()])
        for (rep, clf), predict in models.items():
            status = "failed" if (rep, clf) in failed else "complete"
            conn.execute("INSERT INTO model_runs VALUES (?, ?, ?, ?, ?, '{}', NULL, NULL, 1.0)",
                         (m2, scheme, rep, clf, status))
            if status == "failed":
                continue
            rows = []
            for d in doc_ids:
                for r in range(n_repeats):
                    label, probs = predict(d, labels[d], r)
                    rows.append((m2, scheme, rep, clf, d, r, folds[(d, r)], labels[d], label, *probs))
            conn.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        if new:
            runs.complete(conn, m2)
    return m2


def default_models():
    return {
        ("tfidf", "logreg"): noisy(0.8, 1),
        ("tfidf", "svm"): noisy(0.7, 2),
        ("tfidf", "majority"): majority(),
        ("hubert", "logreg"): noisy(0.6, 3),
        ("hubert", "svm"): noisy(0.5, 4),
        ("hubert", "majority"): majority(),
    }


def rows(conn, table, run_id):
    """Every row of ``table`` for ``run_id``, without the run id, sorted."""
    cur = conn.execute(f"SELECT * FROM {table} WHERE run_id = ?", (run_id,))
    return sorted(json.dumps(list(r)[1:]) for r in cur)
