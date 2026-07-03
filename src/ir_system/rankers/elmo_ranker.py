"""ELMo contextual-embedding reranking.

Represents queries and documents with deep contextualised word
representations (Peters et al., 2018) from TensorFlow Hub's ELMo v3
module, mean-pooled into a single 1024-d vector, and scores candidates
by cosine similarity. Unlike static word vectors, ELMo's BiLM assigns
different vectors to the same term in different contexts, which helps
disambiguate short technical queries.

Document embeddings are cached on disk because encoding the collection
is the dominant cost.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ELMO_HUB_URL = "https://tfhub.dev/google/elmo/3"


class ElmoRanker:
    """Cosine-similarity ranker over mean-pooled ELMo embeddings."""

    def __init__(
        self,
        hub_url: str = ELMO_HUB_URL,
        batch_size: int = 32,
        cache_dir: str | Path | None = "data/cache/elmo",
    ) -> None:
        import tensorflow as tf
        import tensorflow_hub as hub

        self._tf = tf
        logger.info("loading ELMo module from %s", hub_url)
        self._elmo = hub.load(hub_url)
        self._signature = self._elmo.signatures["default"]
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into (n, 1024) mean-pooled ELMo vectors."""
        tf = self._tf
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else "empty" for t in texts[start : start + self.batch_size]]
            outputs = self._signature(tf.constant(batch))
            # 'default' is the mean over the sequence of the weighted sum
            # of the BiLM layers — a fixed sentence representation.
            chunks.append(outputs["default"].numpy())
        return np.vstack(chunks)

    def _encode_documents(self, docnos: Sequence[str], corpus: Dict[str, str]) -> np.ndarray:
        """Encode documents, memoised on disk keyed by the docno set."""
        if self.cache_dir is None:
            return self.encode([corpus[d] for d in docnos])
        key = hashlib.sha1("\n".join(sorted(docnos)).encode()).hexdigest()[:16]
        cache_file = self.cache_dir / f"docs_{key}.npz"
        if cache_file.exists():
            cached = np.load(cache_file, allow_pickle=True)
            stored = {d: i for i, d in enumerate(cached["docnos"].tolist())}
            return cached["embeddings"][[stored[d] for d in docnos]]
        embeddings = self.encode([corpus[d] for d in docnos])
        np.savez_compressed(cache_file, docnos=np.array(list(docnos)), embeddings=embeddings)
        return embeddings

    def score(
        self,
        run: pd.DataFrame,
        queries: Dict[str, str],
        corpus: Dict[str, str],
    ) -> pd.DataFrame:
        """Return ``run`` rescored by query-document ELMo cosine similarity."""
        unique_docnos = run["docno"].drop_duplicates().tolist()
        logger.info("encoding %d documents with ELMo", len(unique_docnos))
        doc_matrix = self._encode_documents(unique_docnos, corpus)
        doc_matrix = doc_matrix / np.linalg.norm(doc_matrix, axis=1, keepdims=True).clip(min=1e-9)
        doc_index = {docno: i for i, docno in enumerate(unique_docnos)}

        qids = run["qid"].drop_duplicates().tolist()
        query_matrix = self.encode([queries[qid] for qid in qids])
        query_matrix = query_matrix / np.linalg.norm(query_matrix, axis=1, keepdims=True).clip(min=1e-9)
        query_index = {qid: i for i, qid in enumerate(qids)}

        rows_q = run["qid"].map(query_index).to_numpy()
        rows_d = run["docno"].map(doc_index).to_numpy()
        similarities = np.einsum("ij,ij->i", query_matrix[rows_q], doc_matrix[rows_d])

        out = run[["qid", "docno"]].copy()
        out["score"] = similarities
        return out
