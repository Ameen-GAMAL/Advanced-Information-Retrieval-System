"""Score normalisation and weighted late fusion of retrieval runs.

A *run* is a pandas DataFrame with (at least) the columns
``["qid", "docno", "score"]`` — the standard PyTerrier result format.
Fusion combines several runs into one by min-max normalising the scores
of each run per query and taking a weighted sum:

    score(q, d) = sum_r  w_r * norm(score_r(q, d))

Documents missing from a run receive a normalised score of 0 for that
run. The weights ``w_r`` are the free parameters optimised in
:mod:`ir_system.optimization`.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

RUN_COLUMNS = ["qid", "docno", "score"]


def min_max_normalize(run: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise scores to [0, 1] independently for each query.

    Queries whose documents all share the same score are mapped to 1.0
    (every retrieved document is treated as equally good).
    """
    out = run.copy()
    grouped = out.groupby("qid")["score"]
    lo = grouped.transform("min")
    hi = grouped.transform("max")
    span = hi - lo
    constant = span <= 0
    span = span.mask(constant, 1.0)
    out["score"] = (out["score"] - lo) / span
    out.loc[constant, "score"] = 1.0
    return out


def fuse(
    runs: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
    normalize: bool = True,
) -> pd.DataFrame:
    """Fuse several runs into a single ranking via weighted score sum.

    Parameters
    ----------
    runs:
        Mapping from ranker name to its run DataFrame.
    weights:
        Mapping from ranker name to its fusion weight. Every key in
        ``runs`` must be present.
    normalize:
        Min-max normalise each run per query before fusing (recommended,
        since BM25, cross-encoder logits and cosine similarities live on
        very different scales).

    Returns
    -------
    A run DataFrame with columns ``qid, docno, score, rank`` sorted by
    descending score within each query.
    """
    if not runs:
        raise ValueError("at least one run is required")
    missing = set(runs) - set(weights)
    if missing:
        raise KeyError(f"missing fusion weights for rankers: {sorted(missing)}")

    merged: pd.DataFrame | None = None
    for name, run in runs.items():
        if not set(RUN_COLUMNS).issubset(run.columns):
            raise ValueError(f"run '{name}' must contain columns {RUN_COLUMNS}")
        prepared = min_max_normalize(run) if normalize else run
        prepared = prepared[RUN_COLUMNS].rename(columns={"score": f"score_{name}"})
        if merged is None:
            merged = prepared
        else:
            merged = merged.merge(prepared, on=["qid", "docno"], how="outer")

    score_cols = [f"score_{name}" for name in runs]
    merged[score_cols] = merged[score_cols].fillna(0.0)
    merged["score"] = sum(
        float(weights[name]) * merged[f"score_{name}"] for name in runs
    )

    fused = merged[["qid", "docno", "score"]].sort_values(
        ["qid", "score"], ascending=[True, False], kind="mergesort"
    )
    fused["rank"] = fused.groupby("qid").cumcount()
    return fused.reset_index(drop=True)
