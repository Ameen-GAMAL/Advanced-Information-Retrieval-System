"""BERT cross-encoder reranking.

Scores each (query, document) pair jointly with a BERT-architecture
cross-encoder fine-tuned for passage relevance (MS MARCO). Cross
attention between query and document tokens makes this the strongest
single reranker in the system, at the cost of one forward pass per
candidate pair — which is why it only rescores the BM25 candidate pool
rather than the whole collection.
"""

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class BertReranker:
    """Rescores a candidate run with a BERT cross-encoder.

    Parameters
    ----------
    model_name:
        Any Hugging Face cross-encoder checkpoint. The default is a
        6-layer BERT-family model distilled for MS MARCO passage
        ranking — a good speed/quality trade-off on CPU.
    batch_size:
        Pairs scored per forward pass.
    max_length:
        Token budget per (query, document) pair.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        max_length: int = 256,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        logger.info("loading cross-encoder %s", model_name)
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.batch_size = batch_size

    def score(
        self,
        run: pd.DataFrame,
        queries: Dict[str, str],
        corpus: Dict[str, str],
    ) -> pd.DataFrame:
        """Return ``run`` with scores replaced by cross-encoder logits.

        Parameters
        ----------
        run:
            Candidate pool (``qid, docno, score``) from the first stage.
        queries:
            Mapping ``qid -> query text``.
        corpus:
            Mapping ``docno -> document text``.
        """
        pairs = [
            (queries[qid], corpus[docno])
            for qid, docno in run[["qid", "docno"]].itertuples(index=False)
        ]
        logger.info("scoring %d query-document pairs", len(pairs))
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=True
        )
        out = run[["qid", "docno"]].copy()
        out["score"] = scores
        return out
