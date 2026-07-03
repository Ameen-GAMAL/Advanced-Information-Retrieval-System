"""Gensim Word2Vec semantic matching and query expansion.

Trains a Word2Vec skip-gram model on the target collection itself (no
external embeddings needed), then:

- scores candidates by cosine similarity between IDF-weighted mean word
  vectors of query and document, and
- optionally expands queries with the nearest neighbours of each query
  term, which feeds a stronger lexical match back into BM25.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenisation shared by training and inference."""
    from gensim.utils import simple_preprocess

    return simple_preprocess(text, deacc=True)


class GensimRanker:
    """Corpus-trained Word2Vec ranker with IDF-weighted mean pooling."""

    def __init__(
        self,
        corpus: Dict[str, str],
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 20,
        seed: int = 42,
        model_path: str | Path | None = "data/cache/word2vec.model",
    ) -> None:
        from gensim.models import Word2Vec

        self._tokenized = {docno: tokenize(text) for docno, text in corpus.items()}
        self._idf = self._compute_idf(self._tokenized)

        model_path = Path(model_path) if model_path else None
        if model_path and model_path.exists():
            logger.info("loading Word2Vec model from %s", model_path)
            self.model = Word2Vec.load(str(model_path))
        else:
            logger.info(
                "training Word2Vec (dim=%d, window=%d, epochs=%d) on %d documents",
                vector_size, window, epochs, len(corpus),
            )
            self.model = Word2Vec(
                sentences=list(self._tokenized.values()),
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                sg=1,  # skip-gram: better for small, technical corpora
                epochs=epochs,
                seed=seed,
                workers=4,
            )
            if model_path:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(model_path))

    @staticmethod
    def _compute_idf(tokenized: Dict[str, List[str]]) -> Dict[str, float]:
        """Smoothed inverse document frequency over the collection."""
        n_docs = len(tokenized)
        document_frequency: Counter = Counter()
        for tokens in tokenized.values():
            document_frequency.update(set(tokens))
        return {
            term: math.log((1 + n_docs) / (1 + df)) + 1.0
            for term, df in document_frequency.items()
        }

    def _embed(self, tokens: List[str]) -> np.ndarray:
        """IDF-weighted mean of in-vocabulary word vectors (zero if none)."""
        vectors, weights = [], []
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])
                weights.append(self._idf.get(token, 1.0))
        if not vectors:
            return np.zeros(self.model.vector_size, dtype=np.float32)
        stacked = np.average(np.asarray(vectors), axis=0, weights=np.asarray(weights))
        norm = np.linalg.norm(stacked)
        return stacked / norm if norm > 0 else stacked

    def score(
        self,
        run: pd.DataFrame,
        queries: Dict[str, str],
        corpus: Dict[str, str],
    ) -> pd.DataFrame:
        """Return ``run`` rescored by Word2Vec embedding cosine similarity."""
        doc_vectors = {
            docno: self._embed(self._tokenized.get(docno, tokenize(corpus[docno])))
            for docno in run["docno"].drop_duplicates()
        }
        query_vectors = {
            qid: self._embed(tokenize(queries[qid]))
            for qid in run["qid"].drop_duplicates()
        }
        scores = [
            float(np.dot(query_vectors[qid], doc_vectors[docno]))
            for qid, docno in run[["qid", "docno"]].itertuples(index=False)
        ]
        out = run[["qid", "docno"]].copy()
        out["score"] = scores
        return out

    def expand_query(self, query: str, terms_per_word: int = 2, min_similarity: float = 0.6) -> str:
        """Append the nearest Word2Vec neighbours of each query term.

        Used in the query-expansion ablation: the expanded query string
        is fed back into BM25 for a semantically broadened lexical match.
        """
        tokens = tokenize(query)
        expansion: List[str] = []
        for token in tokens:
            if token not in self.model.wv:
                continue
            for neighbour, similarity in self.model.wv.most_similar(token, topn=terms_per_word):
                if similarity >= min_similarity and neighbour not in tokens:
                    expansion.append(neighbour)
        return " ".join(tokens + expansion)
