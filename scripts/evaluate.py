#!/usr/bin/env python
"""Evaluate the tuned fusion against the BM25 baseline on the test split.

Loads the weights produced by ``tune_weights.py`` and reports MAP,
nDCG@10 and P@10 for every component ranker and for the tuned fusion on
the held-out test queries — the queries the weights were *not* fitted
to. Results are written to ``artifacts/results.json`` and printed as a
table.

Usage:
    python scripts/evaluate.py [--config config/config.yaml]
"""

import argparse
import json
import logging
from pathlib import Path

from ir_system.data import restrict_qrels, split_topics
from ir_system.evaluation import evaluate_run
from ir_system.fusion import fuse
from ir_system.pipeline import RetrievalPipeline, load_config


def _format_table(rows: dict) -> str:
    metrics = ["MAP", "nDCG@10", "P@10"]
    header = f"{'system':<16}" + "".join(f"{m:>10}" for m in metrics)
    lines = [header, "-" * len(header)]
    for name, scores in rows.items():
        lines.append(f"{name:<16}" + "".join(f"{scores[m]:>10.4f}" for m in metrics))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    config = load_config(args.config)

    weights_path = Path(config["artifacts_dir"]) / "fusion_weights.json"
    if not weights_path.exists():
        raise SystemExit(
            f"{weights_path} not found — run scripts/tune_weights.py first."
        )
    weights = json.loads(weights_path.read_text())["weights"]

    pipeline = RetrievalPipeline(config)
    _, test_topics = split_topics(
        pipeline.topics,
        validation_fraction=config["evaluation"]["validation_fraction"],
        seed=config["evaluation"]["seed"],
    )
    test_qrels = restrict_qrels(pipeline.qrels, test_topics)

    runs = pipeline.component_runs(test_topics)
    fused = fuse(runs, weights)

    results = {name: evaluate_run(run, test_qrels) for name, run in runs.items()}
    results["fusion (tuned)"] = evaluate_run(fused, test_qrels)

    baseline_map = results["bm25"]["MAP"]
    fusion_map = results["fusion (tuned)"]["MAP"]
    improvement = (fusion_map - baseline_map) / baseline_map if baseline_map else 0.0

    table = _format_table(results)
    print("\n" + table)
    print(f"\nMAP improvement of tuned fusion over BM25 baseline: {improvement:+.1%}")

    out_path = Path(config["artifacts_dir"]) / "results.json"
    out_path.write_text(
        json.dumps(
            {"weights": weights, "test_metrics": results, "map_improvement": improvement},
            indent=2,
        )
    )
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
