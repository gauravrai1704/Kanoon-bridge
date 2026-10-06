"""Proximity queries: `term1 /k term2` (both terms within k tokens).  [owner: A]

Westlaw-style search is standard in legal IR; it uses the positional index directly.
"""

from __future__ import annotations

from kanoon_bridge.index.positional import PositionalIndex


def within(index: PositionalIndex, term1: str, term2: str, k: int) -> set[str]:
    """Docs where some occurrence of term1 and term2 are at most k positions apart.

    TODO(A): for docs containing both terms, two-pointer walk over the two position lists
    (the lecture's positional intersect). Order does not matter (|p1 - p2| <= k).
    """
    raise NotImplementedError("TODO(A): positional proximity match")
