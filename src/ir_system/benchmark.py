"""Search quality, latency and frozen profile selection for labeled queries."""

from __future__ import annotations

import csv
import gzip
import io
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .evaluation import average_precision, ndcg_at_k, precision_at_k

BASE_PROFILES = ("bm25", "dense", "hybrid")
RERANK_PROFILES = ("bm25+minilm", "hybrid+minilm")


@dataclass(frozen=True)
class Profile:
    name: str
    rerank_depth: int = 0

    @property
    def id(self) -> str:
        return f"{self.name}@{self.rerank_depth}" if self.rerank_depth else self.name


@dataclass
class BenchmarkRuns:
    summaries: dict
    query_rows: list[dict]
    ranking_rows: list[dict]


def sweep_profiles(depths: Sequence[int], candidate_k: int) -> list[Profile]:
    if not depths or any(depth < 1 or depth > candidate_k for depth in depths):
        raise ValueError("rerank depths must lie in [1, candidate_k]")
    return [Profile(name) for name in BASE_PROFILES] + [
        Profile(name, depth) for name in RERANK_PROFILES for depth in sorted(set(depths))
    ]


def subset_queries(queries: Mapping[str, str], count: int | None, seed: int) -> dict[str, str]:
    ids = sorted(queries)
    if count is not None and count < 1:
        raise ValueError("query subset size must be positive")
    if count is not None and count < len(ids):
        rng = np.random.default_rng(seed)
        ids = sorted(rng.choice(ids, size=count, replace=False).tolist())
    return {qid: queries[qid] for qid in ids}


def score_ranking(ranking: Sequence[str], labels: Mapping[str, int], cutoff: int) -> dict:
    """Score a returned list at the candidate cutoff and at rank ten."""
    if cutoff < 1:
        raise ValueError("candidate cutoff must be positive")
    ranked = list(ranking[:cutoff])
    if len(set(ranked)) != len(ranked):
        raise ValueError("the run returned duplicate document IDs for a query")
    relevant = {docno for docno, label in labels.items() if label > 0}
    gains = {docno: max(0, int(label)) for docno, label in labels.items()}
    return {
        "ndcg_at_10": ndcg_at_k(ranked, gains, 10),
        "precision_at_10": precision_at_k(ranked, gains, 10),
        f"map_at_{cutoff}": average_precision(ranked, gains),
        f"recall_at_{cutoff}": len(relevant.intersection(ranked)) / len(relevant) if relevant else 0.0,
    }


def _percentiles(samples: Sequence[float]) -> dict[str, float]:
    p50, p95 = np.percentile(samples, [50, 95])
    return {"p50": float(p50), "p95": float(p95)}


def run_profiles(
    engine,
    queries: Mapping[str, str],
    qrels: pd.DataFrame,
    profiles: Sequence[Profile],
    *,
    candidate_k: int,
    warmup_queries: Sequence[str] = (),
    keep_rankings: bool = False,
    progress: Callable[[str, int, int], None] | None = None,
) -> BenchmarkRuns:
    """Run equal cuts for each profile, measuring single-query warmed latency."""
    labels_by_qid: dict[str, dict[str, int]] = {}
    for qid, docno, label in qrels[["qid", "docno", "label"]].itertuples(index=False):
        labels_by_qid.setdefault(str(qid), {})[str(docno)] = int(label)
    if not queries:
        raise ValueError("benchmark query split is empty")
    unjudged = set(queries) - set(labels_by_qid)
    if unjudged:
        raise ValueError(f"{len(unjudged)} benchmark queries have no judgments")

    report = BenchmarkRuns({}, [], [])
    for profile in profiles:
        kwargs = {"profile": profile.name, "top_k": candidate_k}
        if profile.rerank_depth:
            kwargs["rerank_depth"] = profile.rerank_depth
        for query in warmup_queries:
            engine.search(query, **kwargs)
        query_rows = []
        stages: dict[str, list[float]] = {}
        for qid, query in queries.items():
            start = perf_counter()
            result = engine.search(query, **kwargs)
            elapsed_ms = (perf_counter() - start) * 1_000
            if len(result.results) > candidate_k:
                raise ValueError(f"{profile.id} returned more than {candidate_k} results")
            docnos = result.results["docno"].astype(str).tolist()
            for stage, value in result.stages_ms.items():
                stages.setdefault(stage, []).append(float(value))
            judged = labels_by_qid[qid]
            query_rows.append({
                "profile_id": profile.id,
                "qid": qid,
                **score_ranking(docnos, judged, candidate_k),
                "latency_ms": elapsed_ms,
            })
            if keep_rankings:
                for rank, (docno, score) in enumerate(
                    result.results[["docno", "score"]].itertuples(index=False), 1
                ):
                    docno = str(docno)
                    bm25_rank, dense_rank = result.sources.get(docno, (None, None))
                    report.ranking_rows.append({
                        "profile_id": profile.id, "qid": qid, "rank": rank, "docno": docno,
                        "score": float(score), "judged_relevance": judged.get(docno),
                        "bm25_rank": bm25_rank, "dense_rank": dense_rank,
                    })
            if progress:
                progress(profile.id, len(query_rows), len(queries))
        report.summaries[profile.id] = {
            "profile": profile.name,
            "rerank_depth": profile.rerank_depth,
            "n_queries": len(query_rows),
            "metrics": {
                metric: float(np.mean([row[metric] for row in query_rows]))
                for metric in ("ndcg_at_10", "precision_at_10", f"map_at_{candidate_k}", f"recall_at_{candidate_k}")
            },
            "latency_ms": _percentiles([row["latency_ms"] for row in query_rows]),
            "stage_latency_ms": {stage: _percentiles(values) for stage, values in sorted(stages.items())},
        }
        report.query_rows.extend(query_rows)
    return report


def select_profiles(dev: BenchmarkRuns, p95_budget_ms: float | None = None) -> tuple[dict, list[Profile]]:
    """Choose profile and both comparison depths using development scores only."""
    if p95_budget_ms is not None and p95_budget_ms <= 0:
        raise ValueError("p95 budget must be positive")

    def quality_sort(profile_id: str):
        row = dev.summaries[profile_id]
        return -row["metrics"]["ndcg_at_10"], row["latency_ms"]["p95"], row["rerank_depth"], profile_id

    def select(ids):
        eligible = [
            profile_id for profile_id in ids
            if p95_budget_ms is None or dev.summaries[profile_id]["latency_ms"]["p95"] <= p95_budget_ms
        ]
        if eligible:
            return min(eligible, key=quality_sort), True
        return min(ids, key=lambda key: (dev.summaries[key]["latency_ms"]["p95"], quality_sort(key))), False

    best_id, within_budget = select(list(dev.summaries))
    frozen = [Profile(name) for name in BASE_PROFILES]
    rerank_depths = {}
    for name in RERANK_PROFILES:
        winner, _ = select([key for key, row in dev.summaries.items() if row["profile"] == name])
        depth = dev.summaries[winner]["rerank_depth"]
        rerank_depths[name] = depth
        frozen.append(Profile(name, depth))
    return {
        "metric": "ndcg_at_10",
        "p95_budget_ms": p95_budget_ms,
        "best_profile_id": best_id,
        "within_p95_budget_on_dev": within_budget,
        "selected_rerank_depths": rerank_depths,
    }, frozen


def compare_with_bm25(query_rows: Sequence[dict], *, replicates: int = 2_000, seed: int = 42) -> dict:
    """Paired query bootstrap intervals for absolute nDCG@10 deltas."""
    scores: dict[str, dict[str, float]] = {}
    for row in query_rows:
        scores.setdefault(row["profile_id"], {})[row["qid"]] = row["ndcg_at_10"]
    baseline = scores.pop("bm25")
    ids = sorted(baseline)
    if replicates < 1 or not ids:
        raise ValueError("paired bootstrap requires queries and at least one replicate")
    sampled = np.random.default_rng(seed).integers(0, len(ids), size=(replicates, len(ids)))
    comparisons = {}
    for profile_id, per_query in scores.items():
        if per_query.keys() != baseline.keys():
            raise ValueError(f"{profile_id} and BM25 do not contain the same query IDs")
        deltas = np.array([per_query[qid] - baseline[qid] for qid in ids])
        low, high = np.percentile(deltas[sampled].mean(axis=1), [2.5, 97.5])
        wins = int(np.sum(deltas > 1e-12))
        losses = int(np.sum(deltas < -1e-12))
        comparisons[profile_id] = {
            "ndcg_at_10_delta": float(deltas.mean()),
            "paired_bootstrap_95_ci": [float(low), float(high)],
            "wins": wins,
            "ties": len(ids) - wins - losses,
            "losses": losses,
        }
    return comparisons


def _write_csv(out, rows: list[dict], fields: Sequence[str]) -> None:
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def write_benchmark(output: Path, summary: dict, dev: BenchmarkRuns, test: BenchmarkRuns) -> None:
    """Write machine-readable evidence, with the summary published last."""
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("dev_query_scores.csv", dev.query_rows), ("test_query_scores.csv", test.query_rows)):
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            _write_csv(stream, rows, list(rows[0]))
    fields = ("profile_id", "qid", "rank", "docno", "score", "judged_relevance", "bm25_rank", "dense_rank")
    with (output / "test_rankings.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                _write_csv(stream, test.ranking_rows, fields)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
