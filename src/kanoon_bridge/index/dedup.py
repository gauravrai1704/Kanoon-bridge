"""Near-duplicate judgments: shingling + MinHash + LSH.  [owner: Gaurav — working]

The same judgment often reaches a corpus more than once (reported twice, a corrected copy, a
connected appeal decided by one common order). Showing every copy wastes the top of the
ranking. Lecture: IIR §19.6 (near-duplicates and shingling).

    shingles   word 5-grams of the judgment text (lower-cased, numbers kept)
    MinHash    Broder, "On the resemblance and containment of documents", SEQUENCES 1997:
               P[min-hash equal] = Jaccard(shingles); 64 multiply-shift hash functions
               h(x) = ((a x + b) mod 2^64) >> 32 with random odd a (Dietzfelbinger et al. 1997)
    LSH        16 bands x 4 rows: pairs that agree on a whole band become candidates, then the
               MinHash estimate must reach `threshold` (default 0.85)
    groups     union-find over the accepted pairs; each doc maps to its group representative
               (the lexicographically smallest id)

scripts/02_build_index.py writes data/processed/index/near_duplicates.json (only docs that have a
duplicate); SearchOptions(collapse_duplicates=True) keeps the best-ranked doc of each group.
"""

from __future__ import annotations

import re
import zlib
from collections import defaultdict

import numpy as np

_MASK32 = np.uint64(32)


def shingles(text: str, k: int = 5) -> set[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < k:
        return {zlib.crc32(" ".join(words).encode())} if words else set()
    return {zlib.crc32(" ".join(words[i:i + k]).encode()) for i in range(len(words) - k + 1)}


def minhash(sh: set[int], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MinHash signature: per hash function, the minimum of h(x) over the shingles."""
    if not sh:
        return np.full(len(a), np.iinfo(np.uint64).max, dtype=np.uint64)
    x = np.fromiter(sh, dtype=np.uint64)[:, None]
    with np.errstate(over="ignore"):
        return ((a[None, :] * x + b[None, :]) >> _MASK32).min(axis=0)     # uint64 arithmetic wraps mod 2^64


def near_duplicate_groups(docs, num_perm: int = 64, bands: int = 16, threshold: float = 0.85,
                          k: int = 5, seed: int = 1) -> dict[str, str]:
    """doc_id -> group representative, for every doc that has at least one near-duplicate."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, np.iinfo(np.uint64).max, size=num_perm, dtype=np.uint64, endpoint=True) | np.uint64(1)
    b = rng.integers(0, np.iinfo(np.uint64).max, size=num_perm, dtype=np.uint64, endpoint=True)
    rows = num_perm // bands
    ids, sigs = [], []
    for d in docs:
        ids.append(d.doc_id)
        sigs.append(minhash(shingles(d.text, k), a, b))
    if not ids:
        return {}
    sig = np.vstack(sigs)
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i in range(len(ids)):
        for band in range(bands):
            buckets[(band, sig[i, band * rows:(band + 1) * rows].tobytes())].append(i)
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2 or len(members) > 200:          # a huge bucket = boilerplate, skip
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                p, q = members[i], members[j]
                if (p, q) in checked:
                    continue
                checked.add((p, q))
                if float(np.mean(sig[p] == sig[q])) >= threshold:
                    parent[find(p)] = find(q)
    groups: dict[int, list[str]] = defaultdict(list)
    for i, d in enumerate(ids):
        groups[find(i)].append(d)
    out: dict[str, str] = {}
    for members in groups.values():
        if len(members) > 1:
            rep = min(members)
            for d in members:
                out[d] = rep
    return out
