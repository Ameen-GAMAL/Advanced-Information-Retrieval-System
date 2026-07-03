"""Unit tests for score normalisation and weighted fusion."""

import pandas as pd
import pytest

from ir_system.fusion import fuse, min_max_normalize


def _run(rows):
    return pd.DataFrame(rows, columns=["qid", "docno", "score"])


def test_min_max_normalize_scales_per_query():
    run = _run([("q1", "a", 10.0), ("q1", "b", 20.0), ("q2", "c", 5.0), ("q2", "d", 7.0)])
    out = min_max_normalize(run).set_index("docno")["score"]
    assert out["a"] == 0.0 and out["b"] == 1.0
    assert out["c"] == 0.0 and out["d"] == 1.0


def test_min_max_normalize_constant_scores_map_to_one():
    run = _run([("q1", "a", 4.0), ("q1", "b", 4.0)])
    out = min_max_normalize(run)["score"]
    assert (out == 1.0).all()


def test_fuse_weights_select_ranker():
    r1 = _run([("q1", "a", 1.0), ("q1", "b", 0.0)])
    r2 = _run([("q1", "a", 0.0), ("q1", "b", 1.0)])
    # All weight on r2 -> b should outrank a.
    fused = fuse({"r1": r1, "r2": r2}, {"r1": 0.0, "r2": 1.0})
    top = fused[fused["qid"] == "q1"].iloc[0]
    assert top["docno"] == "b"


def test_fuse_missing_document_gets_zero():
    r1 = _run([("q1", "a", 1.0), ("q1", "b", 0.5)])
    r2 = _run([("q1", "a", 1.0)])  # b absent from r2
    fused = fuse({"r1": r1, "r2": r2}, {"r1": 0.5, "r2": 0.5})
    assert set(fused["docno"]) == {"a", "b"}


def test_fuse_requires_weight_for_every_run():
    r1 = _run([("q1", "a", 1.0)])
    with pytest.raises(KeyError):
        fuse({"r1": r1}, {})


def test_fuse_output_ranks_are_contiguous():
    r1 = _run([("q1", "a", 3.0), ("q1", "b", 1.0), ("q1", "c", 2.0)])
    fused = fuse({"r1": r1}, {"r1": 1.0})
    ranks = fused[fused["qid"] == "q1"]["rank"].tolist()
    assert ranks == [0, 1, 2]
