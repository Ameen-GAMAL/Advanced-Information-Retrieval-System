"""Compare BM25, dense retrieval, and reranking over SciFact."""

from html import escape
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ir_system.demo import (
    PROFILE_LABELS,
    QUERY_LIMIT,
    BenchmarkReport,
    Demo,
    best_measured_profile,
    build_demo,
    load_benchmark,
    make_snippet,
)

SYSTEM_NAME = "Scientific Abstract Search"
COMPARE_COUNT = 10


@st.cache_resource(show_spinner="Preparing the search index and models…")
def get_demo() -> Demo:
    return build_demo(ROOT)


def show_benchmark(benchmark: BenchmarkReport, profile: str, depth: int) -> None:
    if benchmark.summary is None:
        st.caption(benchmark.note)
        return

    with st.expander("Held-out search benchmark"):
        runtime = benchmark.summary.get("runtime", {})
        benchmark_device = runtime.get("gpu") or runtime.get("cpu") or "recorded benchmark machine"
        st.caption(
            f"SciFact test results. Offline timing device: {benchmark_device}. "
            "This live demo runs on CPU."
        )
        row = benchmark.for_profile(profile, depth)
        baseline = benchmark.for_profile("bm25", depth)
        if row and baseline:
            ndcg = row["metrics"]["ndcg_at_10"]
            delta = ndcg - baseline["metrics"]["ndcg_at_10"]
            st.metric("Selected ranking nDCG@10", f"{ndcg:.3f}", delta=f"{delta:+.3f} vs BM25")
        else:
            st.caption(
                "This configuration has no published test result. "
                "The table shows the configurations that were measured."
            )

        result_rows = []
        for profile_id, run in benchmark.summary["test"].items():
            metrics, latency = run["metrics"], run["latency_ms"]
            result_rows.append(
                {
                    "Configuration": profile_id,
                    "Queries": run["n_queries"],
                    "nDCG@10": round(metrics["ndcg_at_10"], 3),
                    "Recall@100": round(metrics["recall_at_100"], 3),
                    "MAP@100": round(metrics["map_at_100"], 3),
                    "p50 (ms)": round(latency["p50"], 1),
                    "p95 (ms)": round(latency["p95"], 1),
                }
            )
        st.dataframe(pd.DataFrame(result_rows), hide_index=True, use_container_width=True)


def show_results(
    frame: pd.DataFrame,
    demo: Demo,
    sources: dict,
    query: str,
    baseline_positions: dict[str, int] | None = None,
    judgments: dict[str, int] | None = None,
) -> None:
    if frame.empty:
        st.info("No documents matched this query.")
        return

    for position, row in enumerate(frame.itertuples(index=False), start=1):
        doc_id = str(row.docno)
        title = demo.titles.get(doc_id) or f"Document {doc_id}"
        st.html(f"<strong>{position}. {escape(title)}</strong>")
        keywords_rank, dense_rank = sources.get(doc_id, (None, None))
        details = [f"Document {doc_id}"]
        if keywords_rank is not None:
            details.append(f"BM25 candidate #{keywords_rank}")
        if dense_rank is not None:
            details.append(f"Dense candidate #{dense_rank}")
        if baseline_positions is not None:
            baseline_rank = baseline_positions.get(doc_id)
            if baseline_rank is None:
                details.append(f"New to top {COMPARE_COUNT}")
            elif baseline_rank != position:
                change = baseline_rank - position
                details.append(f"{'Up' if change > 0 else 'Down'} {abs(change)} vs BM25")
        if judgments is not None and judgments.get(doc_id, 0) > 0:
            details.append("Judged relevant")
        st.caption(" · ".join(details))
        snippet = make_snippet(demo.documents.get(doc_id, ""), title, query)
        st.html(f"<p>{escape(snippet)}</p>")
        st.divider()


def main() -> None:
    st.set_page_config(page_title=SYSTEM_NAME, page_icon="🔎", layout="wide")
    st.title(SYSTEM_NAME)
    st.write("Search scientific abstracts and compare keyword matches with semantic retrieval and reranking.")

    demo = get_demo()
    st.caption(f"{demo.label} · {len(demo.documents):,} documents")
    benchmark = load_benchmark(ROOT, demo)
    default_profile, default_depth = best_measured_profile(benchmark)
    options = tuple(PROFILE_LABELS)

    example = st.selectbox(
        "Try an example",
        (None, *demo.examples),
        format_func=lambda name: name or "Write your own query",
    )
    if st.session_state.get("active_example") != example:
        st.session_state["active_example"] = example
        st.session_state["query_input"] = demo.examples[example] if example else ""

    with st.form("search"):
        query = st.text_input(
            "Search query",
            key="query_input",
            max_chars=QUERY_LIMIT,
            placeholder="e.g. Sirtuins and ageing",
        )
        mode_column, depth_column = st.columns(2)
        with mode_column:
            profile = st.selectbox(
                "Compare with BM25",
                options,
                index=options.index(default_profile),
                format_func=PROFILE_LABELS.__getitem__,
            )
        with depth_column:
            depths = (20, 50, 100)
            depth = st.selectbox(
                "MiniLM reranking depth",
                depths,
                index=depths.index(default_depth) if default_depth in depths else 1,
                help="Used by configurations that include MiniLM.",
            )
        search = st.form_submit_button("Search", type="primary")

    show_benchmark(benchmark, profile, depth)
    if not search:
        return
    query = query.strip()
    if not query:
        st.warning("Enter a query or choose an example.")
        return

    with st.spinner("Finding and ranking results…"):
        baseline = demo.engine.search(query, profile="bm25", top_k=COMPARE_COUNT)
        comparison = demo.engine.search(query, profile=profile, rerank_depth=depth, top_k=COMPARE_COUNT)

    baseline_positions = {
        str(docno): rank for rank, docno in enumerate(baseline.results["docno"], start=1)
    }
    judged = demo.judgments.get(query)
    if judged is not None:
        st.caption(
            "Results marked judged relevant have a positive relevance label for this query. "
            "The example queries illustrate individual outcomes from the published comparison."
        )

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("BM25 baseline")
        st.caption(f"Query time {baseline.stages_ms['total']:.1f} ms")
        show_results(baseline.results, demo, baseline.sources, query, judgments=judged)
    with right:
        st.subheader(PROFILE_LABELS[profile])
        st.caption(f"Query time {comparison.stages_ms['total']:.1f} ms")
        show_results(comparison.results, demo, comparison.sources, query, baseline_positions, judged)

    with st.expander("Live query timing by stage"):
        rows = [
            {"Configuration": name, "Stage": stage, "Milliseconds": round(elapsed, 1)}
            for name, result in (("BM25 baseline", baseline), (PROFILE_LABELS[profile], comparison))
            for stage, elapsed in result.stages_ms.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
