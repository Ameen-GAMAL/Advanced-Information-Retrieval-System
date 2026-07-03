"""Rank-based evaluation metrics implemented from first principles.

Self-contained implementations of MAP, P@k and nDCG@k so that the
weight-optimisation loop can evaluate thousands of candidate fusions
without a round trip through an external evaluation toolkit. The values
agree with ``pytrec_eval`` / ``ir_measures`` on binary-relevance
collections.

Qrels are a DataFrame with columns ``["qid", "docno", "label"]`` where a
label > 0 marks a relevant document.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import pandas as pd


def _qrels_by_query(qrels: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Group qrels into ``{qid: {docno: label}}`` for fast lookup."""
    out: Dict[str, Dict[str, int]] = {}
    for qid, docno, label in qrels[["qid", "docno", "label"]].itertuples(index=False):
        out.setdefault(str(qid), {})[str(docno)] = int(label)
    return out


def _ranked_docnos(run: pd.DataFrame) -> Dict[str, list]:
    """Group a run into ``{qid: [docno, ...]}`` ordered by descending score."""
    ordered = run.sort_values(["qid", "score"], ascending=[True, False], kind="mergesort")
    return {
        str(qid): group["docno"].astype(str).tolist()
        for qid, group in ordered.groupby("qid", sort=False)
    }


def average_precision(ranking: Sequence[str], relevance: Mapping[str, int]) -> float:
    """Average precision of a single ranked list against graded labels."""
    n_relevant = sum(1 for label in relevance.values() if label > 0)
    if n_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, docno in enumerate(ranking, start=1):
        if relevance.get(docno, 0) > 0:
            hits += 1
            precision_sum += hits / i
    return precision_sum / n_relevant


def precision_at_k(ranking: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """Fraction of the top-k results that are relevant."""
    top = ranking[:k]
    return sum(1 for docno in top if relevance.get(docno, 0) > 0) / k


def ndcg_at_k(ranking: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """Normalised discounted cumulative gain at cutoff ``k``."""
    dcg = sum(
        relevance.get(docno, 0) / math.log2(i + 1)
        for i, docno in enumerate(ranking[:k], start=1)
    )
    ideal_gains = sorted((label for label in relevance.values() if label > 0), reverse=True)
    idcg = sum(gain / math.log2(i + 1) for i, gain in enumerate(ideal_gains[:k], start=1))
    return dcg / idcg if idcg > 0 else 0.0


def _mean_over_queries(run: pd.DataFrame, qrels: pd.DataFrame, per_query_fn) -> float:
    """Average a per-query metric over every query that has qrels."""
    rankings = _ranked_docnos(run)
    relevance = _qrels_by_query(qrels)
    if not relevance:
        raise ValueError("qrels are empty")
    scores = [per_query_fn(rankings.get(qid, []), rels) for qid, rels in relevance.items()]
    return sum(scores) / len(scores)


def mean_average_precision(run: pd.DataFrame, qrels: pd.DataFrame) -> float:
    """MAP of a run — the objective maximised during weight tuning."""
    return _mean_over_queries(run, qrels, average_precision)


def mean_precision_at_k(run: pd.DataFrame, qrels: pd.DataFrame, k: int = 10) -> float:
    """Mean P@k over all assessed queries."""
    return _mean_over_queries(run, qrels, lambda r, rels: precision_at_k(r, rels, k))


def mean_ndcg_at_k(run: pd.DataFrame, qrels: pd.DataFrame, k: int = 10) -> float:
    """Mean nDCG@k over all assessed queries."""
    return _mean_over_queries(run, qrels, lambda r, rels: ndcg_at_k(r, rels, k))


def evaluate_run(run: pd.DataFrame, qrels: pd.DataFrame) -> Dict[str, float]:
    """Standard metric bundle used in the experiment reports."""
    return {
        "MAP": mean_average_precision(run, qrels),
        "nDCG@10": mean_ndcg_at_k(run, qrels, 10),
        "P@10": mean_precision_at_k(run, qrels, 10),
    }
