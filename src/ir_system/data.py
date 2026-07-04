"""Dataset access and query-set splitting.

The experiments use the Vaswani collection (11,429 scientific abstracts,
93 queries with binary relevance judgements), fetched automatically
through PyTerrier's ``ir_datasets`` integration. Queries are split into
a *validation* set — used to tune the fusion weights — and a disjoint
*test* set used only for the final report, so the tuned weights are
never evaluated on the queries they were fitted to.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Terrier is compiled for Java 11+ (class file version 55). An older JVM
# (e.g. Java 8) loads but then fails with UnsupportedClassVersionError, so
# discovery must reject anything below this.
MIN_JAVA_MAJOR = 11


def _has_jvm(java_home: str) -> bool:
    """True if ``java_home`` contains a usable ``jvm.dll`` / ``libjvm``."""
    root = Path(java_home)
    if not root.is_dir():
        return False
    names = ("jvm.dll", "libjvm.so", "libjvm.dylib")
    for sub in ("bin/server", "jre/bin/server", "lib/server", "jre/lib/server"):
        if any((root / sub / name).exists() for name in names):
            return True
    return False


def _java_major(path: str) -> Optional[int]:
    """Best-effort major Java version parsed from an install directory name.

    Handles both the legacy ``1.8`` scheme (-> 8) and the modern ``17``
    scheme (-> 17), e.g. ``jre1.8.0_471`` -> 8, ``jdk-17.0.15`` -> 17.
    """
    numbers = re.findall(r"\d+", os.path.basename(path.rstrip("/\\")))
    if not numbers:
        return None
    first = int(numbers[0])
    if first == 1 and len(numbers) > 1:  # "1.8" style
        return int(numbers[1])
    return first


def _discover_java_home() -> Optional[str]:
    """Find the newest Java 11+ install on this machine, or ``None``.

    Windows Java auto-updates frequently rename the install directory
    (e.g. ``jdk-17.0.15`` -> ``jdk-17``), which leaves a stale
    ``JAVA_HOME`` pointing at a directory that no longer exists — the
    usual cause of PyTerrier's "no jvm dll found" error. Scan the common
    install roots and return the highest-versioned JVM that satisfies
    Terrier's minimum, so an old Java 8 alongside a Java 17 is skipped.
    """
    roots = [
        r"C:\Program Files\Java",
        r"C:\Program Files\Eclipse Adoptium",
        r"C:\Program Files\Microsoft\jdk",
        r"C:\Program Files (x86)\Java",
        "/usr/lib/jvm",
        "/Library/Java/JavaVirtualMachines",
    ]
    candidates = []
    for root in roots:
        if os.path.isdir(root):
            candidates.extend(glob.glob(os.path.join(root, "*")))
    # Some macOS JDKs nest the home under Contents/Home.
    candidates += [
        os.path.join(c, "Contents", "Home")
        for c in list(candidates)
        if os.path.isdir(os.path.join(c, "Contents", "Home"))
    ]

    usable = [
        (_java_major(c) or 0, c)
        for c in candidates
        if _has_jvm(c) and (_java_major(c) or 0) >= MIN_JAVA_MAJOR
    ]
    if not usable:
        return None
    # Highest major version wins; break ties on path for determinism.
    usable.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return usable[0][1]


def ensure_java_home() -> None:
    """Point ``JAVA_HOME`` at a working JVM before PyTerrier starts.

    Leaves a valid ``JAVA_HOME`` untouched; otherwise auto-discovers one
    and sets it for this process. Raises a clear error if no JVM exists,
    instead of the cryptic ``no jvm dll found`` from the JNI layer.
    """
    current = os.environ.get("JAVA_HOME")
    if current and _has_jvm(current):
        return

    discovered = _discover_java_home()
    if discovered:
        if current:
            logger.warning(
                "JAVA_HOME=%r has no JVM; using discovered Java at %r instead.",
                current, discovered,
            )
        os.environ["JAVA_HOME"] = discovered
        return

    raise RuntimeError(
        f"No Java {MIN_JAVA_MAJOR}+ runtime found. PyTerrier/Terrier needs it.\n"
        "Install a JDK (e.g. https://adoptium.net) and set JAVA_HOME to it, "
        f"or fix the current JAVA_HOME (={current!r}), which points to no valid JVM."
    )


def init_pyterrier():
    """Import and initialise PyTerrier (starts the Terrier JVM once)."""
    ensure_java_home()
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
