"""Dataset access and query-set splitting.

The experiments use the Vaswani collection (11,429 scientific abstracts,
93 queries with binary relevance judgements), fetched automatically
through PyTerrier's ``ir_datasets`` integration. Queries are split into
a *validation* set — used to tune the fusion weights — and a disjoint
*test* set used only for the final report, so the tuned weights are
never evaluated on the queries they were fitted to.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def init_pyterrier():
    """Import and initialise PyTerrier (starts the Terrier JVM once)."""
    import pyterrier as pt

    if not pt.started():
        pt.init()
    return pt


def load_dataset(name: str = "irds:vaswani"):
    """Return a PyTerrier dataset handle for ``name``."""
    pt = init_pyterrier()
    return pt.get_dataset(name)


def load_topics_and_qrels(dataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch topics (qid, query) and qrels (qid, docno, label) as DataFrames."""
    topics = dataset.get_topics()
    qrels = dataset.get_qrels()
    topics["qid"] = topics["qid"].astype(str)
    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)
    return topics, qrels


def load_corpus_map(dataset) -> Dict[str, str]:
    """Materialise the corpus as ``{docno: text}`` for the rerankers.

    Vaswani is small enough (~11k short abstracts) to keep fully in
    memory; larger collections would swap this for the index meta store.
    """
    corpus = {str(doc["docno"]): doc["text"] for doc in dataset.get_corpus_iter()}
    logger.info("loaded %d documents into memory", len(corpus))
    return corpus


def split_topics(
    topics: pd.DataFrame,
    validation_fraction: float = 0.5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly split topics into (validation, test) sets.

    The split is deterministic for a given seed so that tuning and
    evaluation scripts, run separately, agree on which queries are
    held out.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    qids = np.sort(topics["qid"].unique())
    rng.shuffle(qids)
    n_val = int(round(len(qids) * validation_fraction))
    val_qids = set(qids[:n_val])
    validation = topics[topics["qid"].isin(val_qids)].reset_index(drop=True)
    test = topics[~topics["qid"].isin(val_qids)].reset_index(drop=True)
    logger.info("split %d topics into %d validation / %d test", len(topics), len(validation), len(test))
    return validation, test


def restrict_qrels(qrels: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
    """Keep only the qrels whose queries appear in ``topics``."""
    return qrels[qrels["qid"].isin(set(topics["qid"]))].reset_index(drop=True)
