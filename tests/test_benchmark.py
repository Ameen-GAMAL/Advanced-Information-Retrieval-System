import csv
import gzip
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from ir_system.benchmark import (
    BenchmarkRuns,
    Profile,
    compare_with_bm25,
    run_profiles,
    score_ranking,
    select_profiles,
    subset_queries,
    sweep_profiles,
    write_benchmark,
)


class FixtureSearchEngine:
    def __init__(self):
        self.calls = []

    def search(self, query, *, profile, top_k, rerank_depth=None):
        self.calls.append((query, profile, top_k, rerank_depth))
        order = ["d1", "d2"] if profile == "bm25" else ["d2", "d1"]
        return SimpleNamespace(
            results=pd.DataFrame({"docno": order, "score": [1.0, 0.5]}).head(top_k),
            stages_ms={"total": 1.0},
            sources={"d1": (1, None), "d2": (2, 1)},
        )


def test_quality_metrics_respect_cutoff_and_reject_duplicate_docs():
    labels = {"d1": 1, "d2": 1, "d3": -1}
    assert score_ranking(["d1", "d3", "d2"], labels, 2) == pytest.approx({
        "ndcg_at_10": 1 / (1 + 1 / 1.584962500721156),
        "precision_at_10": 0.1,
        "map_at_2": 0.5,
        "recall_at_2": 0.5,
    })
    with pytest.raises(ValueError, match="duplicate"):
        score_ranking(["d1", "d1"], labels, 2)


def test_profile_selection_uses_dev_quality_with_optional_latency_budget():
    profiles = sweep_profiles([1, 2], 2)
    scores = {
        "bm25": (0.3, 1), "dense": (0.4, 2), "hybrid": (0.5, 3),
        "bm25+minilm@1": (0.6, 4), "bm25+minilm@2": (0.8, 8),
        "hybrid+minilm@1": (0.7, 5), "hybrid+minilm@2": (0.9, 9),
    }
    dev = BenchmarkRuns({
        profile.id: {
            "profile": profile.name, "rerank_depth": profile.rerank_depth,
            "metrics": {"ndcg_at_10": scores[profile.id][0]},
            "latency_ms": {"p95": scores[profile.id][1]},
        }
        for profile in profiles
    }, [], [])
    selection, test = select_profiles(dev)
    assert selection["best_profile_id"] == "hybrid+minilm@2"
    assert [profile.id for profile in test] == ["bm25", "dense", "hybrid", "bm25+minilm@2", "hybrid+minilm@2"]
    constrained, _ = select_profiles(dev, p95_budget_ms=5)
    assert constrained["best_profile_id"] == "hybrid+minilm@1"
    assert constrained["selected_rerank_depths"] == {"bm25+minilm": 1, "hybrid+minilm": 1}
    impossible, _ = select_profiles(dev, p95_budget_ms=0.1)
    assert impossible["best_profile_id"] == "bm25"
    assert not impossible["within_p95_budget_on_dev"]


def test_subset_is_stable_and_rejects_invalid_sizes():
    queries = {str(i): str(i) for i in range(10)}
    result = subset_queries(queries, 4, seed=7)
    assert len(result) == 4
    assert result == subset_queries(dict(reversed(list(queries.items()))), 4, seed=7)
    assert subset_queries(queries, None, seed=7) == queries
    with pytest.raises(ValueError, match="positive"):
        subset_queries(queries, 0, seed=7)


def test_fixture_runs_only_score_measured_queries_and_export_provenance(tmp_path):
    engine = FixtureSearchEngine()
    labels = pd.DataFrame({"qid": ["q1", "q2"], "docno": ["d2", "d2"], "label": [1, 1]})
    runs = run_profiles(
        engine, {"q1": "first", "q2": "second"}, labels,
        [Profile("bm25"), Profile("hybrid+minilm", 1)], candidate_k=2,
        warmup_queries=["warmup"], keep_rankings=True,
    )
    assert engine.calls == [
        ("warmup", "bm25", 2, None), ("first", "bm25", 2, None), ("second", "bm25", 2, None),
        ("warmup", "hybrid+minilm", 2, 1), ("first", "hybrid+minilm", 2, 1), ("second", "hybrid+minilm", 2, 1),
    ]
    assert runs.summaries["bm25"]["n_queries"] == 2
    assert runs.summaries["bm25"]["metrics"]["map_at_2"] == 0.5
    assert runs.summaries["hybrid+minilm@1"]["metrics"]["map_at_2"] == 1.0
    assert runs.summaries["hybrid+minilm@1"]["stage_latency_ms"]["total"] == {"p50": 1, "p95": 1}
    comparison = compare_with_bm25(runs.query_rows, replicates=25)
    rerank = comparison["hybrid+minilm@1"]
    assert rerank["wins"] == 2
    assert rerank["paired_bootstrap_95_ci"] == [rerank["ndcg_at_10_delta"]] * 2

    write_benchmark(tmp_path, {"test": runs.summaries}, runs, runs)
    assert json.loads((tmp_path / "summary.json").read_text())["test"] == runs.summaries
    with gzip.open(tmp_path / "test_rankings.csv.gz", "rt", newline="") as stream:
        ranked = list(csv.DictReader(stream))
    assert len(ranked) == 8
    assert ranked[1] == {
        "profile_id": "bm25", "qid": "q1", "rank": "2", "docno": "d2", "score": "0.5",
        "judged_relevance": "1", "bm25_rank": "2", "dense_rank": "1",
    }
