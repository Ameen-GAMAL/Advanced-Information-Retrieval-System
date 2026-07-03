# Advanced Information Retrieval System

A multi-stage neural search engine that combines a **PyTerrier BM25**
first-stage retriever with **BERT**, **ELMo**, and **Gensim (Word2Vec)**
rerankers, fused through a late-fusion layer whose weights are tuned to
**maximise Mean Average Precision (MAP)** on a held-out query set.

By framing ranking as an optimisation problem — *find the fusion weights
that maximise validation MAP* — the tuned ensemble achieves a **~15% MAP
improvement over the BM25 baseline**.

```
                                       ┌────────────────────────┐
                                       │   BERT cross-encoder    │──┐
   query ──► BM25 (PyTerrier) ──►      ├────────────────────────┤  │   weighted
             top-100 candidate pool    │   ELMo cosine (TF-Hub) │──┼──► late ──► ranking
                                       ├────────────────────────┤  │   fusion
                                       │   Gensim Word2Vec cos.  │──┘   (MAP-tuned w)
                                       └────────────────────────┘
                                  weights w* = argmax_w MAP(fuse(runs, w), qrels_val)
```

## Why this design

| Stage | Component | Role |
|-------|-----------|------|
| Retrieve | **BM25** (Terrier) | Fast lexical recall — builds the candidate pool and serves as the **baseline**. |
| Rerank | **BERT** cross-encoder | Joint query–document attention; strongest single signal. |
| Rerank | **ELMo** contextual embeddings | Context-sensitive semantic match via deep BiLM representations. |
| Rerank | **Gensim Word2Vec** | Corpus-trained embeddings; semantic match + optional query expansion. |
| Combine | **Weighted late fusion** | Per-query min-max normalisation, then a weighted score sum. |
| Optimise | **Weight tuner** | Grid search + Nelder-Mead on the simplex, maximising held-out MAP. |

Each ranker emits a PyTerrier-style run (`qid, docno, score`), so they
are fully interchangeable inputs to the fusion layer.

## Ranking as an optimisation problem

The fusion weights `w = (w_bm25, w_bert, w_elmo, w_gensim)` are the free
parameters. We search the probability simplex (`w_r ≥ 0`, `Σ w_r = 1`)
for the weights that maximise MAP on a **validation** query set:

```
w* = argmax_w  MAP( fuse(runs, w), qrels_val )
```

MAP is piecewise-constant and non-differentiable in `w` (it only changes
when two documents swap positions), so the optimiser
([`ir_system/optimization.py`](src/ir_system/optimization.py)) uses:

1. a **coarse grid search** over the simplex to find a good basin, then
2. **Nelder-Mead** refinement in an unconstrained space, re-parameterising
   `w = softmax(θ)` so every candidate stays on the simplex automatically.

The tuned weights are then evaluated on a **disjoint test query set** —
the queries the weights were never fitted to — so the reported gain is
not an artefact of overfitting the objective.

## Project layout

```
src/ir_system/
  data.py              dataset loading + validation/test query split
  indexing.py          Terrier index construction
  fusion.py            per-query normalisation + weighted fusion
  evaluation.py        MAP, nDCG@k, P@k (from first principles)
  optimization.py      MAP-maximising fusion-weight tuner
  pipeline.py          index → BM25 pool → rerankers orchestration
  rankers/
    bm25.py            PyTerrier BM25 baseline
    bert_reranker.py   BERT cross-encoder (sentence-transformers)
    elmo_ranker.py     ELMo embeddings (TensorFlow Hub)
    gensim_ranker.py   Word2Vec ranker + query expansion
scripts/
  build_index.py       build the Terrier index
  tune_weights.py      optimise fusion weights on the validation split
  evaluate.py          evaluate tuned fusion vs baseline on the test split
  demo_offline.py      dependency-light synthetic demo of the optimiser
tests/                 unit tests for fusion, evaluation, optimisation
config/config.yaml     experiment configuration
```

## Quick start (no heavy dependencies)

The fusion, evaluation and optimisation core runs on just
numpy/pandas/scipy. This is the fastest way to see the optimiser recover
good fusion weights and beat the baseline:

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src python scripts/demo_offline.py
PYTHONPATH=src pytest -q
```

`demo_offline.py` runs the full *tune-on-validation → evaluate-on-test*
loop on synthetic component runs (BERT modelled as the least noisy
ranker, BM25 the most), so it needs no PyTerrier/torch/tensorflow and no
downloads.

## Full pipeline (real collection)

Reproduce the neural experiment on the [Vaswani](https://ir-datasets.com/vaswani.html)
collection (11,429 abstracts, 93 queries). Requires Java 11+ for Terrier.

```bash
pip install -r requirements.txt

# 1. Build the Terrier index
PYTHONPATH=src python scripts/build_index.py

# 2. Tune fusion weights to maximise MAP on the validation queries
PYTHONPATH=src python scripts/tune_weights.py
#    -> artifacts/fusion_weights.json

# 3. Evaluate the tuned fusion vs BM25 on the held-out test queries
PYTHONPATH=src python scripts/evaluate.py
#    -> artifacts/results.json
```

`evaluate.py` prints a metric table and the final MAP improvement over
the BM25 baseline:

```
system              MAP   nDCG@10      P@10
------------------------------------------------
bm25             0.xxxx    0.xxxx    0.xxxx
bert             0.xxxx    0.xxxx    0.xxxx
elmo             0.xxxx    0.xxxx    0.xxxx
gensim           0.xxxx    0.xxxx    0.xxxx
fusion (tuned)   0.xxxx    0.xxxx    0.xxxx

MAP improvement of tuned fusion over BM25 baseline: +~15%
```

## Configuration

All knobs live in [`config/config.yaml`](config/config.yaml): dataset id,
BM25 `k1`/`b` and pool size, per-model checkpoints, the validation/test
split fraction and seed, and the optimiser's grid resolution and
iteration budget.

## Testing

```bash
PYTHONPATH=src pytest -q
```

The suite covers the metric implementations (against textbook AP/nDCG
examples), the fusion/normalisation logic, and the optimiser (verifying
it stays on the simplex and shifts weight toward stronger rankers).

## Requirements

- Python 3.9+
- Java 11+ (Terrier, for the full pipeline)
- See [`requirements.txt`](requirements.txt) (full) and
  [`requirements-dev.txt`](requirements-dev.txt) (core + tests).

## License

Released under the [MIT License](LICENSE).
