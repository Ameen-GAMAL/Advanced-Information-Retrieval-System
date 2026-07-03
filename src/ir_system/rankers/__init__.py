"""Retrieval and reranking components.

Each ranker exposes a uniform interface producing PyTerrier-style run
DataFrames (``qid, docno, score``), which makes them interchangeable
inputs to the fusion layer:

- :class:`~ir_system.rankers.bm25.BM25Ranker` — lexical first stage.
- :class:`~ir_system.rankers.bert_reranker.BertReranker` — BERT
  cross-encoder relevance scoring.
- :class:`~ir_system.rankers.elmo_ranker.ElmoRanker` — contextual
  embedding cosine similarity.
- :class:`~ir_system.rankers.gensim_ranker.GensimRanker` — corpus-trained
  Word2Vec semantic matching and query expansion.

Imports are kept lazy so that lightweight components (fusion,
evaluation, optimisation) never pull in torch / tensorflow.
"""
