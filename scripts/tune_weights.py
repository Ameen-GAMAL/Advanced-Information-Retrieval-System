#!/usr/bin/env python
"""Tune late-fusion weights to maximise MAP on the validation query set.

Frames ranking as an optimisation problem: the free parameters are the
per-ranker fusion weights, and the objective is validation MAP. The
resulting weights are written to ``artifacts/fusion_weights.json`` for
the evaluation script to consume.

Usage:
    python scripts/tune_weights.py [--config config/config.yaml]
"""

import argparse
import json
import logging
from pathlib import Path

from ir_system.data import restrict_qrels, split_topics
from ir_system.optimization import WeightOptimizer
from ir_system.pipeline import RetrievalPipeline, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    config = load_config(args.config)

    pipeline = RetrievalPipeline(config)
    validation_topics, _ = split_topics(
        pipeline.topics,
        validation_fraction=config["evaluation"]["validation_fraction"],
        seed=config["evaluation"]["seed"],
    )
    validation_qrels = restrict_qrels(pipeline.qrels, validation_topics)

    runs = pipeline.component_runs(validation_topics)
    optimizer = WeightOptimizer(runs, validation_qrels, baseline="bm25")
    result = optimizer.optimize(
        grid_resolution=config["optimization"]["grid_resolution"],
        max_iter=config["optimization"]["max_iter"],
    )

    out_path = Path(config["artifacts_dir"]) / "fusion_weights.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "weights": result.weights,
                "validation_MAP": result.validation_score,
                "baseline_MAP": result.baseline_score,
                "relative_improvement": result.relative_improvement,
            },
            indent=2,
        )
    )
    print(f"\nTuned fusion weights written to {out_path}")
    print(json.dumps(result.weights, indent=2))
    print(
        f"Validation MAP {result.validation_score:.4f} "
        f"vs BM25 {result.baseline_score:.4f} "
        f"({result.relative_improvement:+.1%})"
    )


if __name__ == "__main__":
    main()
