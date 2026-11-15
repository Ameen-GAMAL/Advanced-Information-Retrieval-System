"""Versioned access to the public BEIR SciFact collection."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


SCIFACT_ID = "beir/scifact"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"


@dataclass(frozen=True)
class CorpusBatch:
    documents: dict[str, str]
    titles: dict[str, str]
    queries: dict[str, dict[str, str]]
    qrels: dict[str, pd.DataFrame]
    fingerprint: str


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _archive(cache_dir: Path) -> Path:
    path = cache_dir / "scifact.zip"
    if path.is_file() and _sha256(path) == SCIFACT_SHA256:
        return path

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with urlopen(SCIFACT_URL, timeout=60) as response, NamedTemporaryFile(
            dir=cache_dir, prefix="scifact-", suffix=".download", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
        if _sha256(temporary_path) != SCIFACT_SHA256:
            raise ValueError("The downloaded SciFact archive does not match the expected SHA-256.")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def _read_jsonl(archive: ZipFile, name: str) -> list[dict]:
    with archive.open(f"scifact/{name}.jsonl") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_scifact(cache_dir: Path) -> CorpusBatch:
    """Load the official BEIR train and test query sets from a verified archive.

    The title and abstract are indexed together. Split membership comes from
    BEIR's judgment files, which omit original claims without a retrieval target.
    """
    with ZipFile(_archive(Path(cache_dir))) as archive:
        corpus = _read_jsonl(archive, "corpus")
        documents = {row["_id"]: f"{row['title']}\n{row['text']}".strip() for row in corpus}
        titles = {row["_id"]: row["title"] for row in corpus}
        queries = {row["_id"]: row["text"] for row in _read_jsonl(archive, "queries")}
        split_queries = {}
        split_qrels = {}
        for split in ("train", "test"):
            with io.TextIOWrapper(archive.open(f"scifact/qrels/{split}.tsv"), encoding="utf-8") as handle:
                rows = [
                    (row["query-id"], row["corpus-id"], int(row["score"]))
                    for row in csv.DictReader(handle, delimiter="\t")
                ]
            judgments = pd.DataFrame(rows, columns=["qid", "docno", "label"])
            split_queries[split] = {qid: queries[qid] for qid in sorted(judgments.qid.unique())}
            split_qrels[split] = judgments
    return CorpusBatch(documents, titles, split_queries, split_qrels, SCIFACT_SHA256)
