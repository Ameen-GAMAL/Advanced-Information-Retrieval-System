---
title: Scientific Abstract Search
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Scientific Abstract Search

Compare keyword and semantic retrieval over 5,183 scientific abstracts. The Streamlit demo shows results side by side with source ranks, relevance judgments for example queries, and per-stage timings. The demo and benchmark call the same search engine.

[Benchmark summary](benchmarks/scifact/summary.json) · [Per-query scores](benchmarks/scifact/test_query_scores.csv) · [Ranked runs](benchmarks/scifact/test_rankings.csv.gz) · [CI](https://github.com/Ameen-GAMAL/Advanced-Information-Retrieval-System/actions/workflows/ci.yml)

## Held-out results

The official BEIR SciFact test split contains 300 judged retrieval queries. We used a fixed 50-query sample of the official training split to select the default profile and MiniLM depths of 20, 50, or 100 before running the test. All configurations returned up to 100 documents per query.

| Search profile | nDCG@10 | Recall@100 | Warm p95 |
| --- | ---: | ---: | ---: |
| BM25 | 0.6839 | 0.9260 | 9.9 ms |
| Dense | 0.7127 | 0.9417 | 17.4 ms |
| Hybrid (BM25 + dense via RRF) | 0.7228 | 0.9717 | 24.2 ms |
| BM25 + MiniLM, depth 20 | 0.6822 | 0.9260 | 40.9 ms |
| Hybrid + MiniLM, depth 20 | 0.6859 | 0.9717 | 56.8 ms |
| Original weighted fusion, tuned on SciFact | 0.7140 | 0.9260 | — |

Hybrid returned the highest held-out average nDCG@10, **+0.0389 absolute** over BM25 (paired bootstrap 95% interval **+0.0178 to +0.0610**). Dense retrieval won the development sweep and is the demo default. MiniLM reranking at the selected depth reduced the hybrid profile’s nDCG@10 on this collection. The [full run](benchmarks/scifact/summary.json) includes MAP@100, p50/p95, development sweeps, per-stage timings, exact split IDs, model revisions, and the recorded runtime. Its warm timings were measured on an RTX 3070 Laptop GPU with one query at a time. The live demo displays its own CPU timings.

The [weighted-fusion comparator](benchmarks/scifact/legacy.json) used the same query splits and BM25 candidates. It reranks the full BM25 list, so its Recall@100 remains the same as BM25’s. It ran as a batch experiment and has no comparable per-query latency measurement.

These results cover one English scientific collection. Relevance labels are incomplete. A retrieved abstract can be useful even if it has no judgment, and a highly ranked paper may support or refute a claim.

## How search works

1. **Retrieve.** PyTerrier BM25 and exact cosine search over precomputed BGE-small embeddings each return up to 100 candidates. Either retriever can discover documents independently.
2. **Fuse.** Reciprocal Rank Fusion combines the two lists by rank. The demo records each result’s keyword and dense ranks.
3. **Rerank.** The optional MiniLM cross-encoder rescores a bounded shortlist. Remaining documents retain their first-stage order.

The dataset archive has a checked SHA-256. Indexes and document embeddings are cached by corpus content, and models load at pinned Hugging Face revisions. Search variants and their settings are in [SearchEngine](src/ir_system/search.py).

## Run the demo

With Python 3.12 and Java 17 installed:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/prepare_demo.py
python -m streamlit run app.py
```

The preparation step downloads the verified dataset and checkpoints and prebuilds both indexes. Subsequent runs reuse the cache. Try a preset query and switch profiles to inspect what changed.

The Docker image prepares the same assets at build time and serves the demo on port 7860:

```bash
docker build -t scientific-abstract-search .
docker run --rm -p 7860:7860 scientific-abstract-search
```

This repository can also be deployed as a [Hugging Face Docker Space](https://huggingface.co/docs/hub/spaces-sdks-docker). On a push or pull request, CI runs the unit tests and exercises the Docker image and UI against a small synthetic collection.

## Reproduce the SciFact benchmark

After installing the app dependencies and package, run:

```bash
python scripts/benchmark.py --output-dir benchmarks/scifact
```

The default is CPU. Pass `--device cuda:0` with a compatible CUDA build of PyTorch to reproduce the GPU inference setup. Exact vector matching remains on CPU. Run `python scripts/benchmark.py --help` for the depth sweep and optional development p95 budget. `--smoke` runs two queries per split and writes to a separate cache directory.

Latency covers warm serial calls to the search engine. It excludes model downloads, index construction, the browser, and network round trips. Compare timing values on the same machine and runtime.

## Original Vaswani weighted-fusion experiment

The original BM25, MiniLM, ELMo, and corpus-trained Word2Vec ensemble reached **0.2842 MAP** versus **0.2514** for BM25 on 47 held-out Vaswani queries, a **13.0% relative increase** in this rerun. See the [measured Vaswani report](benchmarks/vaswani/summary.json) for the exact split, weights, and component scores. The Word2Vec model uses multithreaded training, so repeat runs can vary slightly.

To rerun the original experiment, use a separate Python 3.12 environment with Java 17 and install the optional legacy dependencies:

```bash
python -m pip install -r requirements-legacy.txt
python -m pip install --no-deps -e .
python scripts/legacy_benchmark.py
```

The runner tunes its fusion weights on the Vaswani validation queries and writes held-out scores to `benchmarks/vaswani/summary.json`. To evaluate the ensemble on the same SciFact training sample and held-out split as the hybrid benchmark, run `python scripts/legacy_benchmark.py --dataset scifact`. On SciFact, weights below `0.00001` may be set to zero before testing if that preserves validation MAP. The measured run set ELMo’s weight to zero. Pass `--all-components` to also calculate its standalone held-out score. ELMo downloads Google's versioned checkpoint from its Kaggle archive. The lightweight `scripts/demo_offline.py` is an explicitly synthetic optimization example.

## Tests and data sources

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

- [BEIR](https://github.com/beir-cellar/beir) provides the SciFact benchmark packaging. [Allen Institute for AI SciFact](https://github.com/allenai/scifact) provides the claims and judgments (CC BY 4.0) and the S2ORC-derived abstracts (ODC-By 1.0). See [SciFact’s data license](https://github.com/allenai/scifact/blob/master/LICENSE.md).
- [Vaswani via ir_datasets](https://ir-datasets.com/vaswani.html) supplies the historical benchmark.
- [BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5), [MiniLM](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), and [Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf) are used by the current pipeline.
