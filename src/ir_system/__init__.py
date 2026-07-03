"""Advanced Information Retrieval System.

A multi-stage search engine that combines a PyTerrier BM25 first-stage
retriever with BERT, ELMo and Gensim (Word2Vec) rerankers, fused through
a weighted late-fusion layer whose weights are tuned to maximise Mean
Average Precision (MAP) on a held-out query set.
"""

__version__ = "1.0.0"
__author__ = "Ameen Gamal"
