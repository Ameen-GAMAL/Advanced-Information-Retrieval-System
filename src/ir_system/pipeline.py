"""End-to-end pipeline: index -> BM25 pool -> component rerankers.

Shared by the tuning and evaluation scripts so both operate on runs
produced in exactly the same way.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from .data import load_corpus_map, load_dataset, load_topics_and_qrels
from .indexing import build_index

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("config/config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load the experiment configuration YAML."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class RetrievalPipeline:
    """Builds every component run for a given set of topics.

    Heavy resources (index, corpus map, neural models) are constructed
    once and reused across topic sets, so tuning on the validation split
    and evaluating on the test split share the same warmed-up state.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.dataset = load_dataset(config["dataset"])
        self.topics, self.qrels = load_topics_and_qrels(self.dataset)
        self.corpus = load_corpus_map(self.dataset)
        self.index = build_index(self.dataset, config["index_dir"])

        from .rankers.bm25 import BM25Ranker

        bm25_cfg = config["retrieval"]["bm25"]
        self.bm25 = BM25Ranker(
            self.index,
            k1=bm25_cfg["k1"],
            b=bm25_cfg["b"],
            num_results=bm25_cfg["num_results"],
        )
        self._bert = None
        self._elmo = None
        self._gensim = None

    # Lazy properties: each neural model loads only when first used.

    @property
    def bert(self):
        if self._bert is None:
            from .rankers.bert_reranker import BertReranker

            self._bert = BertReranker(model_name=self.config["models"]["bert"])
        return self._bert

    @property
    def elmo(self):
        if self._elmo is None:
            from .rankers.elmo_ranker import ElmoRanker

            self._elmo = ElmoRanker(hub_url=self.config["models"]["elmo"])
        return self._elmo

    @property
    def gensim(self):
        if self._gensim is None:
            from .rankers.gensim_ranker import GensimRanker

            self._gensim = GensimRanker(self.corpus, **self.config["models"]["word2vec"])
        return self._gensim

    def component_runs(self, topics: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute the BM25 pool plus every reranker's scores over it."""
        queries = dict(zip(topics["qid"], topics["query"]))

        logger.info("BM25 first-stage retrieval for %d topics", len(topics))
        pool = self.bm25.search(topics)

        runs = {"bm25": pool}
        logger.info("BERT cross-encoder reranking")
        runs["bert"] = self.bert.score(pool, queries, self.corpus)
        logger.info("ELMo embedding reranking")
        runs["elmo"] = self.elmo.score(pool, queries, self.corpus)
        logger.info("Gensim Word2Vec reranking")
        runs["gensim"] = self.gensim.score(pool, queries, self.corpus)
        return runs
