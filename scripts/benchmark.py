#!/usr/bin/env python
"""Benchmark SciFact search profiles and export query-level evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from ir_system.benchmark import (
    compare_with_bm25,
    run_profiles,
    select_profiles,
    subset_queries,
    sweep_profiles,
    write_benchmark,
)


SOURCE_FILES = (
    "scripts/benchmark.py",
    "src/ir_system/__init__.py",
    "src/ir_system/benchmark.py",
    "src/ir_system/collections.py",
    "src/ir_system/data.py",
    "src/ir_system/evaluation.py",
    "src/ir_system/indexing.py",
    "src/ir_system/rankers/__init__.py",
    "src/ir_system/rankers/bert_reranker.py",
    "src/ir_system/rankers/bm25.py",
    "src/ir_system/search.py",
)


def _source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        source = (root / name).read_bytes().replace(b"\r\n", b"\n")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(source + b"\0")
    return digest.hexdigest()


def _runtime_environment(torch, device: str) -> dict:
    import pyterrier as pt

    packages = {}
    for name in (
        "numpy", "pandas", "torch", "transformers", "sentence-transformers", "huggingface-hub",
        "python-terrier", "pyjnius", "tokenizers", "safetensors", "ir-measures",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
        "torch_cuda": torch.version.cuda if device.startswith("cuda") else None,
        "java_version": str(pt.java.J.System.getProperty("java.version")),
        "java_vendor": str(pt.java.J.System.getProperty("java.vendor")),
        "terrier_version": str(pt.terrier.version()),
        "terrier_helper_version": str(pt.terrier.configure["helper_version"]),
        "packages": packages,
    }


def _progress(profile: str, count: int, total: int) -> None:
    if count == 1 or count % 25 == 0 or count == total:
        print(f"  {profile}: {count}/{total} queries", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--model-dir", type=Path, help="Optional local model snapshots")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rerank-depths", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dev-queries", type=int, default=50, help="Seeded sample of the official train split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p95-budget-ms", type=float, help="Optional development-stage latency constraint")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Use two queries per split for a wiring check")
    args = parser.parse_args()
    if args.torch_threads < 1 or args.dev_queries < 1 or args.rrf_k < 1:
        parser.error("--torch-threads, --dev-queries and --rrf-k must be positive")
    try:
        profiles = sweep_profiles(args.rerank_depths, args.candidate_k)
    except ValueError as exc:
        parser.error(str(exc))

    import torch

    from ir_system.collections import SCIFACT_ID, load_scifact
    from ir_system.search import SearchEngine

    torch.set_num_threads(args.torch_threads)
    source_digest = _source_digest()
    output = args.output_dir or (Path("data/cache/benchmark-smoke") if args.smoke else Path("benchmarks/scifact"))
    if args.smoke and output.resolve() == Path("benchmarks/scifact").resolve():
        parser.error("a smoke run requires a separate output directory")

    started = perf_counter()
    batch = load_scifact(args.cache_dir)
    if set(batch.queries["train"]) & set(batch.queries["test"]):
        raise ValueError("the SciFact training and test splits share query IDs")
    dev_queries = subset_queries(batch.queries["train"], 2 if args.smoke else args.dev_queries, args.seed)
    test_queries = subset_queries(batch.queries["test"], 2 if args.smoke else None, args.seed)
    warmup = list(dev_queries.values())[:2]
    engine = SearchEngine(
        SCIFACT_ID, batch.documents, args.cache_dir,
        model_dir=args.model_dir, candidate_k=args.candidate_k, rrf_k=args.rrf_k, device=args.device,
    )
    print(f"Preparing SciFact index and encoders on {args.device}", flush=True)
    preparation = perf_counter()
    engine.prepare()
    preparation_seconds = perf_counter() - preparation

    print("Sweeping ranking profiles on development queries", flush=True)
    dev = run_profiles(
        engine, dev_queries, batch.qrels["train"], profiles,
        candidate_k=args.candidate_k, warmup_queries=warmup, progress=_progress,
    )
    selection, test_profiles = select_profiles(dev, args.p95_budget_ms)
    print(f"Frozen before held-out evaluation: {', '.join(profile.id for profile in test_profiles)}", flush=True)
    test = run_profiles(
        engine, test_queries, batch.qrels["test"], test_profiles,
        candidate_k=args.candidate_k, warmup_queries=warmup, keep_rankings=True, progress=_progress,
    )
    summary = {
        "schema_version": 1,
        "dataset": SCIFACT_ID,
        "corpus_fingerprint": batch.fingerprint,
        "source_sha256": source_digest,
        "source_files": SOURCE_FILES,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke": args.smoke,
        "engine": engine.metadata,
        "runtime": _runtime_environment(torch, args.device),
        "config": {
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
            "rerank_depths": sorted(set(args.rerank_depths)),
            "seed": args.seed,
            "development_split": "train",
            "held_out_split": "test",
            "warmup_queries_per_profile": len(warmup),
            "latency_protocol": "Serial, warm engine, batch size one query, wall clock around engine.search",
            "bootstrap_replicates": 2_000,
        },
        "preparation_seconds": preparation_seconds,
        "total_seconds": perf_counter() - started,
        "dev_query_ids": list(dev_queries),
        "test_query_ids": list(test_queries),
        "selection": selection,
        "dev": dev.summaries,
        "test": test.summaries,
        "test_vs_bm25": compare_with_bm25(test.query_rows, seed=args.seed),
    }
    write_benchmark(output, summary, dev, test)
    for profile_id, row in test.summaries.items():
        print(f"{profile_id:20s} nDCG@10={row['metrics']['ndcg_at_10']:.4f}  p95={row['latency_ms']['p95']:.1f} ms")
    print(f"Benchmark evidence written to {output}", flush=True)


if __name__ == "__main__":
    main()
