"""Terrier index construction."""

from __future__ import annotations

import logging
from pathlib import Path

from .data import init_pyterrier

logger = logging.getLogger(__name__)


def build_index(dataset, index_dir: str | Path, overwrite: bool = False):
    """Build (or load) a Terrier inverted index for ``dataset``.

    Document text is stored in the index meta store alongside ``docno``
    so downstream rerankers can recover passage text without a second
    pass over the collection.

    Returns a Terrier ``IndexRef``-compatible object usable by
    retrieval transformers.
    """
    pt = init_pyterrier()
    # Terrier resolves *relative* index paths against its internal ".var"
    # directory, which won't match the folder we create here and fails with
    # "path ... does not exist, or cannot be written to". Always hand Terrier
    # an absolute path so both sides agree on the location.
    index_dir = Path(index_dir).resolve()
    properties_file = index_dir / "data.properties"

    if properties_file.exists() and not overwrite:
        logger.info("loading existing index from %s", index_dir)
        return pt.IndexFactory.of(str(properties_file))

    index_dir.mkdir(parents=True, exist_ok=True)
    logger.info("indexing corpus into %s", index_dir)
    indexer = pt.IterDictIndexer(
        str(index_dir),
        meta={"docno": 32, "text": 4096},
        overwrite=True,
    )
    index_ref = indexer.index(dataset.get_corpus_iter())
    return pt.IndexFactory.of(index_ref)
