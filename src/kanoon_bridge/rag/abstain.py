"""Decide BEFORE generation whether retrieval is strong enough to ground an answer.

Uses rank/qpp.py signals (max idf, top-1/top-2 score gap). If weak, return
"Not enough grounding in the indexed law to answer this" instead of calling the LLM.
Metric: abstention precision on queries we know the corpus cannot answer.
"""

from __future__ import annotations

from kanoon_bridge.schema import SearchResult


def decide(result: SearchResult, idf: dict[str, float]) -> tuple[bool, str]:
    """(abstain?, reason). TODO: qpp.pre_retrieval + qpp.post_retrieval + qpp.should_abstain."""
    raise NotImplementedError("TODO: abstention decision")
