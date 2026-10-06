"""Reflection: decide whether to try again, and how.  [agent owner]

    should_reformulate   uses rank/qpp.py on the fused result: weak retrieval (low max idf,
                         small top-1/top-2 gap) -> True. Never more than agent.max_rounds.
    reformulate          pseudo-relevance feedback (Rocchio-style, a lecture concept):
                         take the top-n fused documents, pick their highest tf-idf terms that
                         are not already in the query, add them with weight beta, and also
                         drop the query's lowest-idf term.

The trace records the reason for each extra round, so the video can show it.
"""

from __future__ import annotations

from kanoon_bridge.agent.plan import Plan, SubQuery
from kanoon_bridge.schema import AnalyzedQuery


def should_reformulate(fused: dict[str, float], aq: AnalyzedQuery, idf: dict[str, float],
                       round_no: int, max_rounds: int) -> tuple[bool, str]:
    """(retry?, reason). Working guard on round count; TODO(agent): the QPP test itself
    (qpp.pre_retrieval + post_retrieval + should_abstain-like thresholds)."""
    if round_no >= max_rounds:
        return False, f"reached max rounds ({max_rounds})"
    raise NotImplementedError("TODO(agent): QPP-based retry decision")


def reformulate(aq: AnalyzedQuery, fused: dict[str, float], docs, plan: Plan,
                top_n: int = 5, n_terms: int = 10, beta: float = 0.5) -> list[SubQuery]:
    """New sub-queries for the next round (kind='reformulated').

    TODO(agent): pseudo-relevance feedback from the top_n fused docs (`docs`: doc_id ->
    Document, from index/docstore.py); keep expansion terms out of stop words and section
    tokens of the wrong code.
    """
    raise NotImplementedError("TODO(agent): pseudo-relevance feedback reformulation")
