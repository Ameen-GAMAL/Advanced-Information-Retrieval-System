from types import SimpleNamespace

import numpy as np
import pandas as pd

from ir_system.fusion import fuse
from ir_system.rankers.elmo_ranker import ElmoRanker


def test_elmo_batches_by_length_and_reuses_only_matching_document_text(tmp_path):
    calls = []

    def signature(texts):
        calls.append(texts)
        return {"default": SimpleNamespace(numpy=lambda: np.array([[len(text)] for text in texts]))}

    ranker = ElmoRanker.__new__(ElmoRanker)
    ranker._tf = SimpleNamespace(constant=lambda texts: texts)
    ranker._signature = signature
    ranker._model_id = "fixture"
    ranker._document_vectors = {}
    ranker.batch_size = 2
    ranker.cache_dir = tmp_path

    corpus = {"a": "a longer input", "b": "one", "c": "two words", "d": "another input"}
    expected = np.array([[len(corpus[key])] for key in ("a", "b", "c")])
    np.testing.assert_array_equal(ranker._encode_documents(["a", "b", "c"], corpus), expected)
    assert calls == [["one", "two words"], ["a longer input"]]

    corpus["b"] = "edited"
    np.testing.assert_array_equal(
        ranker._encode_documents(["d", "a", "b"], corpus),
        np.array([[len(corpus[key])] for key in ("d", "a", "b")]),
    )
    assert calls[-1] == ["edited", "another input"]
    np.testing.assert_array_equal(ranker._encode_documents(["c", "a"], corpus), expected[[2, 0]])
    assert len(calls) == 3


def test_omitting_zero_weight_ranker_on_same_pool_preserves_fused_ranking():
    bm25 = pd.DataFrame({"qid": ["q"] * 3, "docno": ["a", "b", "c"], "score": [3.0, 2.0, 1.0]})
    bert = bm25.assign(score=[1.0, 3.0, 2.0])
    elmo = bm25.assign(score=[2.0, 1.0, 3.0])
    weights = {"bm25": 0.4, "bert": 0.6, "elmo": 0.0}
    full = fuse({"bm25": bm25, "bert": bert, "elmo": elmo}, weights)
    reduced = fuse({"bm25": bm25, "bert": bert}, weights)
    pd.testing.assert_frame_equal(full, reduced)
