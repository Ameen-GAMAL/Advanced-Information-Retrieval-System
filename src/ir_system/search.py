"""Shared, versioned keyword, dense, and hybrid search profiles."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .indexing import build_index
from .rankers.bert_reranker import BertReranker
from .rankers.bm25 import BM25Ranker

PROFILES = ("bm25", "dense", "bm25+minilm", "hybrid", "hybrid+minilm")
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RUN_COLUMNS = ["qid", "docno", "score"]
QUERY_ID = "0"
logger = logging.getLogger(__name__)

Ranked = list[tuple[str, float]]


@dataclass(frozen=True)
class SearchResult:
    """One ranked run and its provenance.

    Timings are the wall-clock milliseconds for each executed stage. Source
    ranks are one-based in the original BM25 and dense runs, respectively.
    Reranked runs use reciprocal final rank as their run score so a consumer
    can sort by score without mixing cross-encoder logits and tail scores.
    """

    results: pd.DataFrame
    stages_ms: dict[str, float]
    sources: dict[str, tuple[int | None, int | None]]


@dataclass(frozen=True)
class _IndexCorpus:
    documents: dict[str, str]

    def get_corpus_iter(self) -> Iterable[dict[str, str]]:
        for docno, text in self.documents.items():
            yield {"docno": docno, "text": text}


def _corpus_fingerprint(corpus: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for docno, text in corpus.items():
        for value in (docno, text):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _ordered(scores: Iterable[tuple[str, float]], limit: int) -> Ranked:
    return sorted(scores, key=lambda row: (-row[1], row[0]))[:limit]


def _run(rows: Ranked) -> pd.DataFrame:
    return pd.DataFrame(
        [(QUERY_ID, docno, score) for docno, score in rows],
        columns=RUN_COLUMNS,
    )


class SearchEngine:
    """Retrieve independent top-k runs, fuse them, and optionally rerank.

    The default device is CPU. An explicit device may be used for model
    inference while exact vector matching remains on CPU. Models are loaded
    at immutable revisions. With `model_dir` or `IR_MODEL_DIR`, prepared
    snapshots can be loaded from `<checkpoint-name>-<revision>` directories.
    """

    def __init__(
        self,
        dataset: str,
        corpus: dict[str, str],
        cache_dir: Path,
        *,
        model_dir: Path | None = None,
        candidate_k: int = 100,
        rrf_k: int = 60,
        device: str = "cpu",
    ) -> None:
        if not corpus:
            raise ValueError("search requires a nonempty corpus")
        if candidate_k < 1 or rrf_k < 1:
            raise ValueError("candidate_k and rrf_k must be positive")

        self.dataset = dataset
        self.corpus = dict(sorted(corpus.items()))
        self.docnos = tuple(self.corpus)
        self.corpus_fingerprint = _corpus_fingerprint(self.corpus)
        self.cache_dir = Path(cache_dir)
        configured_model_dir = model_dir or os.getenv("IR_MODEL_DIR")
        self.model_dir = Path(configured_model_dir) if configured_model_dir else None
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.device = device
        self._bm25 = None
        self._encoder = None
        self._vectors = None
        self._reranker = None
        self._resource_lock = RLock()

    @property
    def metadata(self) -> dict:
        """Return JSON-serializable corpus, model, and ranking settings."""
        return {
            "dataset": self.dataset,
            "corpus_fingerprint": self.corpus_fingerprint,
            "candidate_k": self.candidate_k,
            "rrf_k": self.rrf_k,
            "device": self.device,
            "models": {
                "dense": {"id": DENSE_MODEL, "revision": DENSE_REVISION},
                "rerank": {"id": RERANK_MODEL, "revision": RERANK_REVISION},
            },
        }

    def _model_location(self, model_id: str, revision: str) -> tuple[str, str | None]:
        if self.model_dir is not None:
            local = self.model_dir / f"{model_id.split('/')[-1]}-{revision}"
            if local.is_dir():
                return str(local.resolve()), None
        return model_id, revision

    def _create_bm25(self):
        index_dir = self.cache_dir / f"terrier-v1-{self.corpus_fingerprint}"
        index = build_index(_IndexCorpus(self.corpus), index_dir)
        return BM25Ranker(index, num_results=self.candidate_k)

    def _create_encoder(self):
        from sentence_transformers import SentenceTransformer

        model, revision = self._model_location(DENSE_MODEL, DENSE_REVISION)
        return SentenceTransformer(model, revision=revision, device=self.device)

    def _create_reranker(self):
        model, revision = self._model_location(RERANK_MODEL, RERANK_REVISION)
        return BertReranker(model_name=model, revision=revision, device=self.device)

    def _get_bm25(self):
        with self._resource_lock:
            if self._bm25 is None:
                self._bm25 = self._create_bm25()
            return self._bm25

    def _get_reranker(self):
        with self._resource_lock:
            if self._reranker is None:
                self._reranker = self._create_reranker()
            return self._reranker

    def _document_vectors(self, encoder) -> np.ndarray:
        cache = self.cache_dir / f"dense-v1-{self.corpus_fingerprint}-{DENSE_REVISION}.npy"
        if cache.is_file():
            try:
                vectors = np.load(cache, allow_pickle=False)
                if (vectors.ndim == 2 and vectors.shape[0] == len(self.docnos)
                        and vectors.shape[1] == encoder.get_sentence_embedding_dimension()
                        and vectors.dtype == np.float32 and np.isfinite(vectors).all()):
                    return vectors
                logger.warning("ignoring mismatched dense cache %s", cache)
            except (OSError, ValueError, EOFError):
                logger.warning("ignoring unreadable dense cache %s", cache)

        logger.info("encoding %d corpus documents with %s", len(self.docnos), DENSE_MODEL)
        vectors = np.asarray(
            encoder.encode(list(self.corpus.values()), batch_size=32,
                           normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, delete=False) as handle:
                temporary_path = Path(handle.name)
                np.save(handle, vectors, allow_pickle=False)
            temporary_path.replace(cache)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return vectors

    def _get_dense(self):
        with self._resource_lock:
            if self._vectors is None:
                encoder = self._create_encoder()
                self._vectors = self._document_vectors(encoder)
                self._encoder = encoder
            return self._encoder, self._vectors

    def prepare(self, profiles: Sequence[str] = PROFILES) -> None:
        """Load or build resources needed by the requested search profiles."""
        unknown = set(profiles) - set(PROFILES)
        if unknown:
            raise ValueError(f"unknown search profiles: {sorted(unknown)}")
        if any(profile != "dense" for profile in profiles):
            self._get_bm25()
        if any(profile == "dense" or profile.startswith("hybrid") for profile in profiles):
            self._get_dense()
        if any(profile.endswith("+minilm") for profile in profiles):
            self._get_reranker()

    def _keyword_search(self, query: str) -> Ranked:
        # Pass plain terms to Terrier so punctuation is not treated as query syntax.
        lexical_query = " ".join(re.findall(r"[^\W_]+", query))
        if not lexical_query:
            return []
        topics = pd.DataFrame({"qid": [QUERY_ID], "query": [lexical_query]})
        ranked = self._get_bm25().search(topics)
        unique = ranked.groupby("docno")["score"].max()
        return _ordered(unique.items(), self.candidate_k)

    def _dense_search(self, query: str) -> Ranked:
        encoder, vectors = self._get_dense()
        query_vector = encoder.encode(
            [QUERY_PREFIX + query], normalize_embeddings=True, show_progress_bar=False,
        )[0]
        scores = np.einsum("ij,j->i", vectors, np.asarray(query_vector, dtype=np.float32))
        # Sorted docnos and a stable sort provide a deterministic tie rule.
        best = np.argsort(-scores, kind="stable")[:self.candidate_k]
        return [(self.docnos[i], float(scores[i])) for i in best]

    def _rerank(self, query: str, ranked: Ranked, depth: int) -> Ranked:
        head, tail = ranked[:depth], ranked[depth:]
        if not head:
            return ranked
        predictions = self._get_reranker().score(_run(head), {QUERY_ID: query}, self.corpus)
        ordered = _ordered(predictions[["docno", "score"]].itertuples(index=False, name=None), depth)
        return [(docno, 1 / rank) for rank, (docno, _) in enumerate(ordered + tail, start=1)]

    def search(
        self,
        query: str,
        *,
        profile: str = "hybrid",
        rerank_depth: int = 50,
        top_k: int = 100,
    ) -> SearchResult:
        """Return up to `min(top_k, candidate_k)` documents for one query."""
        if profile not in PROFILES:
            raise ValueError(f"unknown search profile: {profile!r}")
        if top_k < 1 or rerank_depth < 0:
            raise ValueError("top_k must be positive and rerank_depth must be nonnegative")

        started = perf_counter()
        # The demo caches this engine across sessions. Keep the Terrier reader
        # and model inference single-threaded while still allowing lazy loads.
        with self._resource_lock:
            return self._search(query, profile, rerank_depth, top_k, started)

    def _search(self, query: str, profile: str, rerank_depth: int, top_k: int, started: float) -> SearchResult:
        query = query.strip()
        stages: dict[str, float] = {}
        sources: dict[str, tuple[int | None, int | None]] = {}
        if not query:
            return SearchResult(_run([]), {"total": (perf_counter() - started) * 1000}, sources)

        sparse: Ranked = []
        dense: Ranked = []
        if profile != "dense":
            stage = perf_counter()
            sparse = self._keyword_search(query)
            stages["bm25"] = (perf_counter() - stage) * 1000
        if profile == "dense" or profile.startswith("hybrid"):
            stage = perf_counter()
            dense = self._dense_search(query)
            stages["dense"] = (perf_counter() - stage) * 1000

        sparse_ranks = {docno: i for i, (docno, _) in enumerate(sparse, start=1)}
        dense_ranks = {docno: i for i, (docno, _) in enumerate(dense, start=1)}
        ranked = dense if profile == "dense" else sparse
        if profile.startswith("hybrid"):
            stage = perf_counter()
            scores: dict[str, float] = {}
            for ranks in (sparse_ranks, dense_ranks):
                for docno, rank in ranks.items():
                    scores[docno] = scores.get(docno, 0.0) + 1 / (self.rrf_k + rank)
            ranked = _ordered(scores.items(), self.candidate_k)
            stages["rrf"] = (perf_counter() - stage) * 1000
        if profile.endswith("+minilm") and rerank_depth > 0 and ranked:
            stage = perf_counter()
            ranked = self._rerank(query, ranked, min(rerank_depth, len(ranked)))
            stages["rerank"] = (perf_counter() - stage) * 1000

        ranked = ranked[:top_k]
        sources = {docno: (sparse_ranks.get(docno), dense_ranks.get(docno)) for docno, _ in ranked}
        results = _run(ranked)
        stages["total"] = (perf_counter() - started) * 1000
        return SearchResult(results, stages, sources)
