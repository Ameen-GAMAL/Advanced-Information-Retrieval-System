"""Unit tests for the weight optimiser."""

import numpy as np
import pandas as pd

from ir_system.optimization import WeightOptimizer, softmax


def test_softmax_lies_on_simplex():
    w = softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()


def _make_runs():
    """Two rankers: a strong one and pure noise, over 8 queries."""
    rng = np.random.default_rng(0)
    strong_rows, noise_rows, qrels_rows = [], [], []
    for q in range(8):
        qid = f"q{q}"
        relevance = rng.normal(size=10)
        relevant = set(np.argsort(relevance)[-3:])
        for i in range(10):
            docno = f"d{q}_{i}"
            qrels_rows.append((qid, docno, 1 if i in relevant else 0))
            strong_rows.append((qid, docno, float(relevance[i] + rng.normal(scale=0.2))))
            noise_rows.append((qid, docno, float(rng.normal())))
    runs = {
        "bm25": pd.DataFrame(noise_rows, columns=["qid", "docno", "score"]),
        "strong": pd.DataFrame(strong_rows, columns=["qid", "docno", "score"]),
    }
    qrels = pd.DataFrame(qrels_rows, columns=["qid", "docno", "label"])
    return runs, qrels


def test_optimizer_beats_noisy_baseline():
    runs, qrels = _make_runs()
    optimizer = WeightOptimizer(runs, qrels, baseline="bm25")
    result = optimizer.optimize(grid_resolution=6, max_iter=100)
    # The tuned fusion should not be worse than the noisy baseline, and
    # should place most of its weight on the strong ranker.
    assert result.validation_score >= result.baseline_score
    assert result.weights["strong"] > result.weights["bm25"]


def test_optimizer_weights_sum_to_one():
    runs, qrels = _make_runs()
    result = WeightOptimizer(runs, qrels, baseline="bm25").optimize(grid_resolution=4, max_iter=50)
    assert abs(sum(result.weights.values()) - 1.0) < 1e-6
