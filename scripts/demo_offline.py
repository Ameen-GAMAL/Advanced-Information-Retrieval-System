#!/usr/bin/env python
"""Self-contained demo of the fusion + weight-optimisation core.

Runs the full ranking-as-optimisation loop on *synthetic* component
runs, so it needs only numpy / pandas / scipy — no PyTerrier, torch or
tensorflow, and no downloads. It exists to let a reviewer see the
optimiser recover good fusion weights and improve MAP over the BM25
baseline in a few seconds.

The synthetic generator models each ranker as a noisy observer of a
hidden relevance signal, with BERT the least noisy and BM25 (the
baseline) the most noisy — mirroring the real system's ordering.

Usage:
    python scripts/demo_offline.py [--queries 60] [--seed 0]
"""

import argparse

import numpy as np
import pandas as pd

from ir_system.evaluation import evaluate_run
from ir_system.fusion import fuse
from ir_system.optimization import WeightOptimizer

# Per-ranker observation noise (higher = weaker ranker).
RANKER_NOISE = {"bm25": 1.30, "gensim": 1.00, "elmo": 0.80, "bert": 0.55}


def synthesize(n_queries: int, pool: int, seed: int):
    """Generate synthetic component runs and binary qrels.

    Each query has a latent relevance vector over its candidate pool.
    Every ranker scores the pool as ``relevance + Gaussian(noise)``; the
    top few documents by true relevance are labelled relevant.
    """
    rng = np.random.default_rng(seed)
    runs = {name: [] for name in RANKER_NOISE}
    qrels_rows = []

    for q in range(n_queries):
        qid = f"q{q}"
        docnos = [f"d{q}_{i}" for i in range(pool)]
        relevance = rng.normal(size=pool)
        n_relevant = rng.integers(2, 6)
        relevant = set(np.argsort(relevance)[-n_relevant:])
        for i, docno in enumerate(docnos):
            qrels_rows.append((qid, docno, 1 if i in relevant else 0))
        for name, noise in RANKER_NOISE.items():
            scores = relevance + rng.normal(scale=noise, size=pool)
            for docno, score in zip(docnos, scores):
                runs[name].append((qid, docno, float(score)))

    run_frames = {
        name: pd.DataFrame(rows, columns=["qid", "docno", "score"])
        for name, rows in runs.items()
    }
    qrels = pd.DataFrame(qrels_rows, columns=["qid", "docno", "label"])
    return run_frames, qrels


def split_queries(qrels: pd.DataFrame, runs, seed: int):
    """Deterministic 50/50 validation/test split over query ids."""
    rng = np.random.default_rng(seed)
    qids = np.array(sorted(qrels["qid"].unique()))
    rng.shuffle(qids)
    half = len(qids) // 2
    val, test = set(qids[:half]), set(qids[half:])

    def subset(frame, keep):
        return frame[frame["qid"].isin(keep)].reset_index(drop=True)

    return (
        {n: subset(r, val) for n, r in runs.items()},
        subset(qrels, val),
        {n: subset(r, test) for n, r in runs.items()},
        subset(qrels, test),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=60)
    parser.add_argument("--pool", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs, qrels = synthesize(args.queries, args.pool, args.seed)
    val_runs, val_qrels, test_runs, test_qrels = split_queries(qrels, runs, args.seed)

    print("=" * 60)
    print("Ranking as optimisation — synthetic demo")
    print("=" * 60)

    print("\nComponent MAP on the validation split:")
    for name, run in val_runs.items():
        print(f"  {name:<8} {evaluate_run(run, val_qrels)['MAP']:.4f}")

    optimizer = WeightOptimizer(val_runs, val_qrels, baseline="bm25")
    result = optimizer.optimize(grid_resolution=6, max_iter=150)

    print("\nTuned fusion weights (maximising validation MAP):")
    for name, weight in result.weights.items():
        print(f"  {name:<8} {weight:.3f}")

    baseline_test = evaluate_run(test_runs["bm25"], test_qrels)["MAP"]
    fused_test = evaluate_run(fuse(test_runs, result.weights), test_qrels)["MAP"]
    improvement = (fused_test - baseline_test) / baseline_test

    print("\nHeld-out test split:")
    print(f"  BM25 baseline MAP : {baseline_test:.4f}")
    print(f"  Tuned fusion MAP  : {fused_test:.4f}")
    print(f"  Improvement       : {improvement:+.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
