"""Inter-judge agreement for our hand-judged query sets.  [owner: D — working]

Report percent agreement and Cohen's kappa for each hand-built set (E2, E4, E6, E7),
computed over the (query, doc) pairs both judges graded.
"""

from __future__ import annotations

from collections import Counter

Qrels = dict[str, dict[str, int]]


def _pairs(a: Qrels, b: Qrels) -> list[tuple[int, int]]:
    return [(a[q][d], b[q][d]) for q in a if q in b for d in a[q] if d in b[q]]


def percent_agreement(a: Qrels, b: Qrels) -> float:
    pairs = _pairs(a, b)
    return sum(x == y for x, y in pairs) / len(pairs) if pairs else 0.0


def cohens_kappa(a: Qrels, b: Qrels) -> float:
    pairs = _pairs(a, b)
    if not pairs:
        return 0.0
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
