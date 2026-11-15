#!/usr/bin/env python
"""Evaluate the original weighted-fusion ensemble on Vaswani or SciFact."""

import argparse
import importlib.metadata
import json
import logging
import platform
from pathlib import Path
from time import perf_counter

import pandas as pd

from ir_system.benchmark import subset_queries
from ir_system.data import restrict_qrels, split_topics
from ir_system.evaluation import evaluate_run
from ir_system.fusion import fuse
from ir_system.optimization import WeightOptimizer
from ir_system.pipeline import RetrievalPipeline, load_config


def scifact_experiment(config: dict, cache_dir: Path, device: str):
    """Use the common SciFact BM25 candidates with the original reranker ensemble."""
    from ir_system.collections import SCIFACT_ID, load_scifact
    from ir_system.rankers.bert_reranker import BertReranker
    from ir_system.rankers.elmo_ranker import ElmoRanker
    from ir_system.rankers.gensim_ranker import GensimRanker
    from ir_system.search import SearchEngine

    collection = load_scifact(cache_dir)
    engine = SearchEngine(SCIFACT_ID, collection.documents, cache_dir)
    models = config["models"]
    elmo = ElmoRanker(hub_url=models["elmo"], cache_dir=cache_dir / "elmo")
    bert = BertReranker(model_name=models["bert"], revision=models.get("bert_revision"), device=device)
    gensim = GensimRanker(collection.documents, model_path=None, **models["word2vec"])

    def component_runs(topics: pd.DataFrame) -> dict[str, pd.DataFrame]:
        query_map = dict(zip(topics.qid, topics["query"]))
        pool = pd.concat(
            [engine.search(query, profile="bm25").results.assign(qid=qid) for qid, query in query_map.items()],
            ignore_index=True,
        )
        return {
            "bm25": pool,
            "bert": bert.score(pool, query_map, collection.documents),
            "elmo": elmo.score(pool, query_map, collection.documents),
            "gensim": gensim.score(pool, query_map, collection.documents),
        }

    def topics(queries: dict[str, str]) -> pd.DataFrame:
        return pd.DataFrame(queries.items(), columns=["qid", "query"])

    validation = topics(subset_queries(collection.queries["train"], 50, 42))
    return (
        component_runs, len(collection.documents), collection.fingerprint,
        validation, topics(collection.queries["test"]),
        restrict_qrels(collection.qrels["train"], validation), collection.qrels["test"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--dataset", choices=("vaswani", "scifact"), default="vaswani")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--device", default="cpu", help="SciFact MiniLM inference device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    config = load_config(args.config)
    started = perf_counter()
    if args.dataset == "scifact":
        (component_runs, document_count, corpus_fingerprint,
         validation, test, validation_qrels, test_qrels) = scifact_experiment(config, args.cache_dir, args.device)
        dataset = "beir/scifact"
    else:
        pipeline = RetrievalPipeline(config)
        validation, test = split_topics(pipeline.topics, **config["evaluation"])
        validation_qrels = restrict_qrels(pipeline.qrels, validation)
        test_qrels = restrict_qrels(pipeline.qrels, test)
        # Load the archival model before the cross-encoder so download failures fail early.
        _ = pipeline.elmo
        component_runs, document_count = pipeline.component_runs, len(pipeline.corpus)
        dataset, corpus_fingerprint = config["dataset"], None
    validation_runs = component_runs(validation)
    solution = WeightOptimizer(validation_runs, validation_qrels, baseline="bm25").optimize(
        **config["optimization"]
    )
    test_runs = component_runs(test)
    test_runs["fusion (tuned)"] = fuse(test_runs, solution.weights)
    metrics = {profile: evaluate_run(run, test_qrels) for profile, run in test_runs.items()}
    baseline = metrics["bm25"]["MAP"]
    delta = metrics["fusion (tuned)"]["MAP"] - baseline
    packages = (
        "numpy", "pandas", "scipy", "python-terrier", "ir-datasets", "torch",
        "sentence-transformers", "transformers", "tensorflow", "tensorflow-hub", "gensim",
    )
    report = {
        "dataset": dataset,
        "document_count": document_count,
        "corpus_fingerprint": corpus_fingerprint,
        "validation_qids": validation.qid.tolist(),
        "test_qids": test.qid.tolist(),
        "config": config,
        "runtime": {"python": platform.python_version(), "os": platform.platform(),
                    "packages": {package: importlib.metadata.version(package) for package in packages}},
        "seconds": perf_counter() - started,
        "weights": solution.weights,
        "validation_map": solution.validation_score,
        "test_metrics": metrics,
        "test_map_delta": delta,
        "test_map_relative_change": delta / baseline if baseline else None,
    }
    default_output = "benchmarks/vaswani/summary.json" if args.dataset == "vaswani" else "benchmarks/scifact/legacy.json"
    output = args.output or Path(default_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Legacy benchmark written to {output}")


if __name__ == "__main__":
    main()
