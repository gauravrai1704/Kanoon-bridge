"""Proximity queries: `term1 /k term2` (both terms within k tokens).  [owner: A]

Westlaw-style search is standard in legal IR; it uses the positional index directly.
"""

from __future__ import annotations

from kanoon_bridge.index.positional import PositionalIndex


def within(index: PositionalIndex, term1: str, term2: str, k: int) -> set[str]:
    """Docs where some occurrence of term1 and term2 are at most k positions apart.

    Uses a two-pointer walk over the position lists.
    Order does not matter: |p1 - p2| <= k.
    """
    if k < 0:
        return set()

    docs1 = index.docs_with(term1)
    docs2 = index.docs_with(term2)

    # Only documents containing both terms can match.
    candidate_docs = docs1.intersection(docs2)

    results: set[str] = set()

    for doc_id in candidate_docs:
        positions1 = index.postings[term1][doc_id]
        positions2 = index.postings[term2][doc_id]

        i = 0
        j = 0

        while i < len(positions1) and j < len(positions2):
            difference = positions1[i] - positions2[j]

            if abs(difference) <= k:
                results.add(doc_id)
                break

            if difference < 0:
                i += 1
            else:
                j += 1

    return results