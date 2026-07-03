"""Streamlit demo: interactive search over BM25 + BERT/ELMo/Gensim fusion.

Lets a user type a free-text query, run it through the BM25 first stage,
rerank the pool with the selected neural components, and see the tuned
(or manually adjusted) late-fusion ranking — side by side with each
individual component's ranking.

Run locally:
    streamlit run app.py

Deploy: see the Dockerfile in the repo root (HF Spaces Docker SDK).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from ir_system.fusion import fuse
from ir_system.pipeline import RetrievalPipeline, load_config

st.set_page_config(page_title="Advanced Information Retrieval System", layout="wide")

CONFIG_PATH = "config/config.yaml"
WEIGHTS_PATH = Path("artifacts/fusion_weights.json")
RESULTS_PATH = Path("artifacts/results.json")

# Terrier's query parser treats +-()[]{}^"~*?:\/ as operators; a free-text
# UI query shouldn't accidentally trigger them, so strip to plain words.
_UNSAFE_QUERY_CHARS = re.compile(r"[^\w\s]")


def sanitize_query(text: str) -> str:
    return _UNSAFE_QUERY_CHARS.sub(" ", text).strip()


@st.cache_resource(show_spinner="Building BM25 index (first run only, ~1-2 min)...")
def get_pipeline() -> RetrievalPipeline:
    config = load_config(CONFIG_PATH)
    return RetrievalPipeline(config)


@st.cache_resource(show_spinner="Loading BERT cross-encoder...")
def get_bert(_pipeline: RetrievalPipeline):
    return _pipeline.bert


@st.cache_resource(show_spinner="Loading ELMo module (large download, first run only)...")
def get_elmo(_pipeline: RetrievalPipeline):
    return _pipeline.elmo


@st.cache_resource(show_spinner="Training/loading Gensim Word2Vec...")
def get_gensim(_pipeline: RetrievalPipeline):
    return _pipeline.gensim


def load_tuned_weights() -> dict:
    if WEIGHTS_PATH.exists():
        return json.loads(WEIGHTS_PATH.read_text())["weights"]
    return {"bm25": 0.25, "bert": 0.25, "elmo": 0.25, "gensim": 0.25}


def load_offline_metrics() -> dict | None:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return None


st.title("🔎 Advanced Information Retrieval System")
st.caption(
    "PyTerrier BM25 baseline reranked by BERT, ELMo and Gensim, "
    "combined by a MAP-tuned weighted fusion."
)

with st.sidebar:
    st.header("Rerankers")
    use_bert = st.checkbox("BERT cross-encoder", value=True)
    use_elmo = st.checkbox("ELMo embeddings", value=False, help="Heaviest component — large model download.")
    use_gensim = st.checkbox("Gensim Word2Vec", value=True)

    st.header("Fusion weights")
    tuned = load_tuned_weights()
    st.caption("Defaults come from `artifacts/fusion_weights.json` (MAP-tuned).")
    weights = {"bm25": st.slider("BM25", 0.0, 1.0, float(tuned.get("bm25", 0.25)), 0.05)}
    if use_bert:
        weights["bert"] = st.slider("BERT", 0.0, 1.0, float(tuned.get("bert", 0.25)), 0.05)
    if use_elmo:
        weights["elmo"] = st.slider("ELMo", 0.0, 1.0, float(tuned.get("elmo", 0.25)), 0.05)
    if use_gensim:
        weights["gensim"] = st.slider("Gensim", 0.0, 1.0, float(tuned.get("gensim", 0.25)), 0.05)

    top_k = st.number_input("Results to show", min_value=1, max_value=50, value=10)

metrics = load_offline_metrics()
if metrics:
    with st.expander("Offline evaluation results (from `scripts/evaluate.py`)"):
        rows = [{"system": name, **scores} for name, scores in metrics["test_metrics"].items()]
        st.dataframe(pd.DataFrame(rows).set_index("system"), use_container_width=True)
        st.metric("MAP improvement over BM25", f"{metrics['map_improvement']:+.1%}")
else:
    st.info(
        "No `artifacts/results.json` found yet — run `scripts/tune_weights.py` "
        "then `scripts/evaluate.py` to populate offline metrics.",
        icon="ℹ️",
    )

query = st.text_input("Search query", placeholder="e.g. compressible flow past a body")
search = st.button("Search", type="primary")

if search and query.strip():
    clean_query = sanitize_query(query)
    if not clean_query:
        st.warning("Query has no searchable terms after sanitisation.")
        st.stop()

    pipeline = get_pipeline()
    topics = pd.DataFrame([{"qid": "live", "query": clean_query}])

    with st.spinner("Retrieving with BM25..."):
        pool = pipeline.bm25.search(topics)

    if pool.empty:
        st.warning("No documents matched this query.")
        st.stop()

    queries = {"live": clean_query}
    runs = {"bm25": pool}

    if use_bert:
        with st.spinner("Reranking with BERT..."):
            runs["bert"] = get_bert(pipeline).score(pool, queries, pipeline.corpus)
    if use_elmo:
        with st.spinner("Reranking with ELMo..."):
            runs["elmo"] = get_elmo(pipeline).score(pool, queries, pipeline.corpus)
    if use_gensim:
        with st.spinner("Reranking with Gensim..."):
            runs["gensim"] = get_gensim(pipeline).score(pool, queries, pipeline.corpus)

    active_weights = {name: weights[name] for name in runs}
    total = sum(active_weights.values())
    if total <= 0:
        st.warning("At least one fusion weight must be greater than 0.")
        st.stop()
    active_weights = {name: w / total for name, w in active_weights.items()}

    fused = fuse(runs, active_weights).head(int(top_k))

    st.subheader(f"Fused results for: “{clean_query}”")
    st.caption("Weights used: " + ", ".join(f"{k}={v:.2f}" for k, v in active_weights.items()))

    for rank, row in enumerate(fused.itertuples(index=False), start=1):
        text = pipeline.corpus.get(row.docno, "")
        preview = text[:300] + ("..." if len(text) > 300 else "")
        st.markdown(f"**{rank}. `{row.docno}`** — score {row.score:.4f}")
        st.write(preview)
        st.divider()

    with st.expander("Per-component rankings"):
        cols = st.columns(len(runs))
        for col, (name, run) in zip(cols, runs.items()):
            with col:
                st.markdown(f"**{name}**")
                top = run.sort_values("score", ascending=False).head(int(top_k))
                st.dataframe(top[["docno", "score"]].reset_index(drop=True), use_container_width=True)
elif search:
    st.warning("Enter a query first.")
