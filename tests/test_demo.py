import json

from ir_system.collections import SCIFACT_ID
from ir_system.demo import Demo, best_measured_profile, load_benchmark
from ir_system.search import SearchEngine


def test_benchmark_matches_ranking_and_complete_test_split_across_devices(tmp_path):
    documents = {"1": "Example abstract."}
    engine = SearchEngine(SCIFACT_ID, documents, tmp_path)
    demo = Demo(SCIFACT_ID, "SciFact", "archive-id", documents, {}, {}, {}, {"held-out"}, engine)

    summary_path = tmp_path / "benchmarks" / "scifact" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary = {
        "schema_version": 1,
        "smoke": False,
        "dataset": SCIFACT_ID,
        "corpus_fingerprint": "archive-id",
        "engine": {**engine.metadata, "device": "cuda:0"},
        "test_query_ids": ["held-out"],
        "selection": {"best_profile_id": "hybrid+minilm@20"},
        "test": {
            "bm25": {"profile": "bm25"},
            "hybrid+minilm@20": {"profile": "hybrid+minilm", "rerank_depth": 20},
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    report = load_benchmark(tmp_path, demo)
    assert best_measured_profile(report) == ("hybrid+minilm", 20)
    assert report.for_profile("bm25", 50) == {"profile": "bm25"}
    assert report.for_profile("hybrid+minilm", 50) is None

    changes = (
        {"smoke": True},
        {"test_query_ids": ["different"]},
        {"engine": {**summary["engine"], "rrf_k": 42}},
        {"corpus_fingerprint": "different"},
    )
    for change in changes:
        summary_path.write_text(json.dumps({**summary, **change}), encoding="utf-8")
        assert load_benchmark(tmp_path, demo).summary is None
