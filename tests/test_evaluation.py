"""Unit tests for the rank-metric implementations."""

import pandas as pd

from ir_system.evaluation import (
    average_precision,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
)


def test_average_precision_textbook_example():
    # Ranking with relevant docs at positions 1, 3, 5 out of 3 relevant.
    ranking = ["a", "b", "c", "d", "e"]
    relevance = {"a": 1, "c": 1, "e": 1}
    # AP = mean(1/1, 2/3, 3/5) = 0.7556
    assert abs(average_precision(ranking, relevance) - 0.75555) < 1e-4


def test_average_precision_perfect_ranking_is_one():
    ranking = ["a", "b", "c", "d"]
    relevance = {"a": 1, "b": 1}
    assert average_precision(ranking, relevance) == 1.0


def test_average_precision_no_relevant_is_zero():
    assert average_precision(["a", "b"], {"c": 1}) == 0.0


def test_precision_at_k():
    ranking = ["a", "b", "c", "d"]
    relevance = {"a": 1, "c": 1}
    assert precision_at_k(ranking, relevance, 2) == 0.5
    assert precision_at_k(ranking, relevance, 4) == 0.5


def test_ndcg_perfect_ranking_is_one():
    ranking = ["a", "b", "c"]
    relevance = {"a": 1, "b": 1}
    assert abs(ndcg_at_k(ranking, relevance, 3) - 1.0) < 1e-9


def test_ndcg_penalises_late_relevance():
    good = ndcg_at_k(["a", "b", "c"], {"a": 1}, 3)
    bad = ndcg_at_k(["c", "b", "a"], {"a": 1}, 3)
    assert good > bad


def test_mean_average_precision_over_run():
    run = pd.DataFrame(
        [
            ("q1", "a", 3.0),
            ("q1", "b", 2.0),
            ("q1", "c", 1.0),
            ("q2", "x", 3.0),
            ("q2", "y", 2.0),
        ],
        columns=["qid", "docno", "score"],
    )
    qrels = pd.DataFrame(
        [("q1", "a", 1), ("q1", "c", 1), ("q2", "y", 1)],
        columns=["qid", "docno", "label"],
    )
    # q1 AP = mean(1/1, 2/3) = 0.8333 ; q2 AP = 1/2 = 0.5 ; MAP = 0.6667
    assert abs(mean_average_precision(run, qrels) - 0.66666) < 1e-4
