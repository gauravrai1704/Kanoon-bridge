"""Proximity queries: `term1 /k term2` (both terms within k tokens).  [owner: A — working]

Westlaw-style search is standard in legal IR; it uses the positional index directly.
"""

from __future__ import annotations

from kanoon_bridge.index.positional import PositionalIndex


def positions_within(p1: list[int], p2: list[int], k: int) -> bool:
    """Two-pointer walk over two sorted position lists: is some |a - b| <= k?"""
    i = j = 0
    while i < len(p1) and j < len(p2):
        if abs(p1[i] - p2[j]) <= k:
            return True
        if p1[i] < p2[j]:
            i += 1
        else:
            j += 1
    return False


def within(index: PositionalIndex, term1: str, term2: str, k: int) -> set[str]:
    """Docs where some occurrence of term1 and term2 are at most k positions apart
    (the lecture's positional intersect; order does not matter)."""
    a, b = index.postings.get(term1, {}), index.postings.get(term2, {})
    if len(a) > len(b):
        a, b = b, a
    return {d for d, pos in a.items() if d in b and positions_within(pos, b[d], k)}


def within_positions(pos1: dict[str, list[int]], pos2: dict[str, list[int]], k: int) -> set[str]:
    """Same test for arbitrary position maps (doc -> sorted positions), e.g. phrase starts."""
    if len(pos1) > len(pos2):
        pos1, pos2 = pos2, pos1
    return {d for d, p in pos1.items() if d in pos2 and positions_within(p, pos2[d], k)}
