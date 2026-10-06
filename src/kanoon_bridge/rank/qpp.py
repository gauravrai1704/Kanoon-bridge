"""Query performance prediction (QPP) without relevance labels.  [owner: D]

Predicts how well retrieval is going for this query. Used twice:
  * fusion.py: set the lexical/dense weight alpha per query
  * rag/abstain.py: refuse to answer when retrieval looks weak

Pre-retrieval predictors (from the query and index statistics):
    max_idf, avg_idf, query_scope (share of docs containing any term)
Post-retrieval predictors (from the score list):
    score_gap = s1 - s2, normalised_top = s1 / mean(s1..s10), score std of top-10

Background: Tian et al., "What can predicted query performance tell us about agentic RAG",
IR-RAG workshop at SIGIR 2025.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QPPFeatures:
    max_idf: float = 0.0
    avg_idf: float = 0.0
    query_scope: float = 0.0
    score_gap: float = 0.0
    normalised_top: float = 0.0
    top_std: float = 0.0


def pre_retrieval(terms: list[str], idf: dict[str, float], df: dict[str, int], n_docs: int) -> QPPFeatures:
    """TODO(D): max/avg idf and query scope."""
    raise NotImplementedError("TODO(D): pre-retrieval QPP")


def post_retrieval(features: QPPFeatures, scores: list[float]) -> QPPFeatures:
    """Fill score-based predictors from a descending score list. TODO(D)."""
    raise NotImplementedError("TODO(D): post-retrieval QPP")


def alpha_from_qpp(f: QPPFeatures, default: float = 0.7) -> float:
    """Lexical weight in [0, 1]: high when lexical retrieval looks confident (high max_idf,
    big score gap), lower otherwise so the dense channel helps. TODO(D, with C): pick a
    simple monotone rule, tune on validation only, report it in the paper."""
    raise NotImplementedError("TODO(D): QPP -> alpha")


def should_abstain(f: QPPFeatures, gap_threshold: float = 0.05, top_threshold: float = 1.1) -> bool:
    """True when retrieval looks too weak to ground an answer (used by rag/abstain.py). TODO(D)."""
    raise NotImplementedError("TODO(D): abstention rule")
