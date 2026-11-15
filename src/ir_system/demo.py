"""Build a common search engine for the Streamlit demo and its startup checks."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

from .collections import SCIFACT_ID, load_scifact
from .search import PROFILES, SearchEngine


QUERY_LIMIT = 512
PROFILE_LABELS = {
    "bm25": "BM25 only",
    "dense": "Dense retrieval",
    "bm25+minilm": "BM25 + MiniLM",
    "hybrid": "Hybrid retrieval",
    "hybrid+minilm": "Hybrid + MiniLM",
}

# Examples are illustrative. The benchmark aggregates the complete test split.
SCIFACT_EXAMPLES = {
    "Exact term · SIRT1": "SIRT1 and ageing",
    "Paraphrase · bone strength": "How does regular exercise help bones stay strong?",
    "Held-out example · dense retrieval win": "Silencing of Bcl2 is important for the maintenance and progression of tumors.",
    "Held-out example · BM25 win": "Venules have a thinner or absent smooth layer compared to arterioles.",
}


@dataclass(frozen=True)
class Demo:
    dataset: str
    label: str
    fingerprint: str
    documents: dict[str, str]
    titles: dict[str, str]
    examples: dict[str, str]
    judgments: dict[str, dict[str, int]]
    test_query_ids: set[str]
    engine: SearchEngine


@dataclass(frozen=True)
class BenchmarkReport:
    summary: dict | None
    note: str = ""

    def for_profile(self, profile: str, depth: int) -> dict | None:
        if self.summary is None:
            return None
        profile_id = f"{profile}@{depth}" if profile.endswith("+minilm") else profile
        return self.summary["test"].get(profile_id)


def build_demo(root: Path) -> Demo:
    """Load the configured corpus, build indexes if needed, and warm model resources."""
    cache_dir = Path(os.getenv("IR_CACHE_DIR", str(root / "data" / "cache")))
    corpus_mode = os.getenv("IR_DEMO_CORPUS", "scifact")
    if corpus_mode == "scifact":
        corpus = load_scifact(cache_dir)
        dataset, label, fingerprint = SCIFACT_ID, "BEIR SciFact", corpus.fingerprint
        documents, titles = corpus.documents, corpus.titles
        examples = SCIFACT_EXAMPLES
        test_query_ids = set(corpus.queries["test"])
        judgments = {}
        for split, queries in corpus.queries.items():
            for qid, group in corpus.qrels[split].groupby("qid"):
                judgments[queries[qid]] = dict(zip(group["docno"], group["label"]))
    elif corpus_mode == "fixture":
        corpus_bytes = (root / "tests" / "fixtures" / "demo_corpus.json").read_bytes()
        fixture = json.loads(corpus_bytes)
        dataset, label = "demo/fixture", "Synthetic smoke-test corpus"
        fingerprint = hashlib.sha256(corpus_bytes).hexdigest()
        titles = {doc["id"]: doc["title"] for doc in fixture["documents"]}
        documents = {doc["id"]: f"{doc['title']}\n{doc['text']}" for doc in fixture["documents"]}
        examples = fixture["examples"]
        judgments = fixture["judgments"]
        test_query_ids = set()
    else:
        raise ValueError(f"Unsupported IR_DEMO_CORPUS: {corpus_mode!r}")

    engine = SearchEngine(dataset, documents, cache_dir)
    engine.prepare(PROFILES)
    return Demo(dataset, label, fingerprint, documents, titles, examples, judgments, test_query_ids, engine)


def load_benchmark(root: Path, demo: Demo) -> BenchmarkReport:
    """Display results only when they describe the demo's corpus and engine settings."""
    if demo.dataset != SCIFACT_ID:
        return BenchmarkReport(None, "Offline benchmark metrics apply to BEIR SciFact.")

    path = root / "benchmarks" / "scifact" / "summary.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return BenchmarkReport(None, "This build has no published SciFact benchmark.")

    comparison_engine = report.get("engine") if isinstance(report, dict) else None
    test_query_ids = report.get("test_query_ids") if isinstance(report, dict) else None
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("smoke") is not False
        or report.get("dataset") != demo.dataset
        or report.get("corpus_fingerprint") != demo.fingerprint
        or not isinstance(comparison_engine, dict)
        or _ranking_settings(comparison_engine) != _ranking_settings(demo.engine.metadata)
        or not isinstance(test_query_ids, list)
        or any(not isinstance(qid, str) for qid in test_query_ids)
        or len(test_query_ids) != len(demo.test_query_ids)
        or set(test_query_ids) != demo.test_query_ids
        or not isinstance(report.get("test"), dict)
        or "bm25" not in report["test"]
    ):
        return BenchmarkReport(None, "The published benchmark uses a different search configuration.")

    return BenchmarkReport(report)


def _ranking_settings(metadata: dict) -> dict:
    return {key: value for key, value in metadata.items() if key != "device"}


def best_measured_profile(report: BenchmarkReport) -> tuple[str, int]:
    if report.summary:
        selection = report.summary.get("selection", {})
        profile_id = selection.get("best_profile_id")
        measured = report.summary["test"].get(profile_id)
        if measured and measured.get("profile") in PROFILES:
            return measured["profile"], measured.get("rerank_depth") or 50
    return "hybrid+minilm", 50


def make_snippet(document: str, title: str, query: str, max_chars: int = 320) -> str:
    """Return a short extract around a query term while avoiding title repetition."""
    title_line, newline, rest = document.partition("\n")
    text = " ".join((rest if newline and title_line == title else document).split())
    if len(text) <= max_chars:
        return text

    # A word boundary avoids matching e.g. "gene" within "oxygen".
    terms = set(re.findall(r"\w{4,}", query.casefold()))
    positions = [match.start() for match in re.finditer(r"\w+", text) if match.group().casefold() in terms]
    start = max(0, positions[0] - 70) if positions else 0
    if start:
        preceding_space = text.rfind(" ", 0, start)
        start = preceding_space + 1 if preceding_space >= 0 else 0
    end = min(len(text), start + max_chars)
    if end < len(text):
        last_space = text.rfind(" ", start, end)
        end = last_space if last_space > start else end
    return f"{'…' if start else ''}{text[start:end]}{'…' if end < len(text) else ''}"
