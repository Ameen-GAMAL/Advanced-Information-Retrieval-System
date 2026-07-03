"""BM25 first-stage retrieval (the baseline the fusion must beat)."""

from __future__ import annotations

import pandas as pd

from ..data import init_pyterrier


class BM25Ranker:
    """Terrier BM25 retriever with tunable ``k1`` / ``b`` parameters.

    Produces the candidate pool that every neural reranker scores, and
    doubles as the baseline system in all reported comparisons.
    """

    def __init__(self, index, k1: float = 1.2, b: float = 0.75, num_results: int = 100) -> None:
        pt = init_pyterrier()
        self.num_results = num_results
        self._retriever = pt.BatchRetrieve(
            index,
            wmodel="BM25",
            controls={"bm25.k_1": str(k1), "bm25.b": str(b)},
            num_results=num_results,
            metadata=["docno"],
        )

    def search(self, topics: pd.DataFrame) -> pd.DataFrame:
        """Retrieve the top-``num_results`` documents for each topic."""
        run = self._retriever.transform(topics)
        run["qid"] = run["qid"].astype(str)
        run["docno"] = run["docno"].astype(str)
        return run[["qid", "docno", "score"]].reset_index(drop=True)
