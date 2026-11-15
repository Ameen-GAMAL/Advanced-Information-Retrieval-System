"""Exercise search profiles with deterministic models and a tiny corpus."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep

import numpy as np
import pandas as pd
import pytest

from ir_system.search import PROFILES, QUERY_PREFIX, SearchEngine


class FakeBM25:
    def search(self, topics):
        qid, query = topics.iloc[0][["qid", "query"]]
        assert query == "neural search"
        return pd.DataFrame(
            [(qid, "c", 1.0), (qid, "b", 2.0), (qid, "b", 1.0), (qid, "a", 2.0)],
            columns=["qid", "docno", "score"],
        )


class FakeEncoder:
    def __init__(self):
        self.encoded = []

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, normalize_embeddings, **kwargs):
        self.encoded.append(list(texts))
        assert normalize_embeddings
        vectors = {
            "alpha": [0.0, 1.0],
            "bravo": [0.6, 0.8],
            "charlie": [0.8, 0.6],
            "delta": [1.0, 0.0],
            QUERY_PREFIX + "neural search?": [1.0, 0.0],
        }
        return np.array([vectors[text] for text in texts], dtype=np.float32)


class FakeReranker:
    def score(self, pool, queries, corpus):
        assert queries == {"0": "neural search?"}
        out = pool.copy()
        out["score"] = out["docno"].map({"a": -3.0, "b": -2.0, "c": -1.0, "d": 4.0})
        return out


@pytest.fixture
def corpus():
    return {"d": "delta", "c": "charlie", "a": "alpha", "b": "bravo"}


@pytest.fixture
def engine(tmp_path, corpus, monkeypatch):
    engine = SearchEngine("test", corpus, tmp_path, candidate_k=3)
    monkeypatch.setattr(engine, "_create_bm25", FakeBM25)
    monkeypatch.setattr(engine, "_create_encoder", FakeEncoder)
    monkeypatch.setattr(engine, "_create_reranker", FakeReranker)
    return engine


@pytest.mark.parametrize(
    "profile, expected, stages",
    [
        ("bm25", ["a", "b", "c"], {"bm25", "total"}),
        ("dense", ["d", "c", "b"], {"dense", "total"}),
        ("bm25+minilm", ["b", "a", "c"], {"bm25", "rerank", "total"}),
        ("hybrid", ["b", "c", "a"], {"bm25", "dense", "rrf", "total"}),
        ("hybrid+minilm", ["c", "b", "a"], {"bm25", "dense", "rrf", "rerank", "total"}),
    ],
)
def test_profiles_deduplicate_and_rank(engine, profile, expected, stages):
    result = engine.search("neural search?", profile=profile, rerank_depth=2)
    assert result.results["docno"].tolist() == expected
    assert result.results["score"].is_monotonic_decreasing
    assert result.stages_ms.keys() == stages


def test_hybrid_can_retrieve_a_document_outside_bm25(engine):
    engine.candidate_k = 4
    result = engine.search("neural search?", profile="hybrid")
    assert result.sources["d"] == (None, 1)
    assert result.sources["a"] == (1, 4)
    assert result.sources["b"] == (2, 3)


def test_rerank_depth_independent_of_display_cutoff(engine):
    result = engine.search("neural search?", profile="hybrid+minilm", rerank_depth=2, top_k=1)
    assert result.results["docno"].tolist() == ["c"]
    assert result.sources == {"c": (3, 2)}


def test_empty_and_invalid_requests_do_not_load_models(engine):
    result = engine.search("    ")
    assert result.results.empty and result.sources == {}
    assert result.stages_ms.keys() == {"total"}
    with pytest.raises(ValueError, match="unknown search profile"):
        engine.search("neural search?", profile="unknown")
    with pytest.raises(ValueError, match="positive"):
        engine.search("neural search?", top_k=0)
    assert engine._bm25 is engine._encoder is engine._reranker is None


def test_prepare_loads_selected_profiles_only(engine):
    engine.prepare(["bm25"])
    assert isinstance(engine._bm25, FakeBM25)
    assert engine._encoder is engine._reranker is None
    engine.prepare(PROFILES)
    assert isinstance(engine._encoder, FakeEncoder)
    assert isinstance(engine._reranker, FakeReranker)


def test_dense_cache_matches_content_and_model(engine, corpus, monkeypatch):
    engine.prepare(["dense"])
    assert engine._encoder.encoded == [["alpha", "bravo", "charlie", "delta"]]

    reused = SearchEngine("test", dict(reversed(corpus.items())), engine.cache_dir)
    monkeypatch.setattr(reused, "_create_encoder", FakeEncoder)
    reused.prepare(["dense"])
    assert reused._encoder.encoded == []
    assert reused.corpus_fingerprint == engine.corpus_fingerprint
    assert reused.search("neural search?", profile="dense").results.iloc[0]["docno"] == "d"

    changed = SearchEngine("test", {**corpus, "a": "bravo"}, engine.cache_dir)
    monkeypatch.setattr(changed, "_create_encoder", FakeEncoder)
    changed.prepare(["dense"])
    assert changed.corpus_fingerprint != engine.corpus_fingerprint
    assert changed._encoder.encoded == [["bravo", "bravo", "charlie", "delta"]]


def test_shared_engine_serializes_model_access(engine, monkeypatch):
    requests = Barrier(4, timeout=5)
    in_use = Lock()

    class Reranker(FakeReranker):
        def score(self, *args):
            assert in_use.acquire(blocking=False), "shared reranker was used concurrently"
            try:
                sleep(0.025)
                return super().score(*args)
            finally:
                in_use.release()

    def search(_):
        requests.wait()
        return engine.search("neural search?", profile="hybrid+minilm", rerank_depth=2)

    monkeypatch.setattr(engine, "_create_reranker", Reranker)
    engine.prepare()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(search, range(4)))
    assert all(result.results["docno"].tolist() == ["c", "b", "a"] for result in results)
