"""Word n-gram BM25 channel for case-as-query retrieval.  [owner: Gaurav — working]

When the query is a whole judgment (IL-PCSR's task), the strongest lexical evidence is shared
*phrasing*: the same facts, the same quoted passages, the same statutory language. Unigram BM25
loses word order. Following IL-PCSR (Paul et al., EMNLP 2025), whose best lexical baseline is BM25
over word 3-5-grams, this channel indexes word trigrams (an n-gram index generalises the biword
index of IIR §2.4.1) and scores them with BM25.

Build (scripts/02_build_index.py):
    text -> lower-case words (masks like [SECTION] removed) -> word ids -> trigram ids
    trigram id = (w1 * V + w2) * V + w3 (int64), one column per distinct trigram
    doc x trigram matrix, each cell already the BM25 weight
        idf(g) * tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avg|d|))       (scipy CSR)
Query: its distinct trigrams -> columns (binary search) -> score = M[:, cols].sum over columns.

Used only for long queries (search.py: SearchOptions.ngram and ngram_min_words), fused with the
unigram relevance as (1 - beta) * lexical + beta * ngram on a 0-1 scale, and given to LTR as a feature.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

_MASK = re.compile(r"\[[A-Z?_ /]+\]")
_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD.findall(_MASK.sub(" ", text).lower())


@dataclass
class NgramIndex:
    n: int
    vocab: dict[str, int]
    features: np.ndarray                 # sorted distinct n-gram ids (int64)
    matrix: object                       # scipy.sparse.csc_matrix [docs x features], BM25 weights
    doc_ids: list[str]
    k1: float = 1.2
    b: float = 0.75
    _row: dict[str, int] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, docs, n: int = 3, k1: float = 1.2, b: float = 0.75) -> "NgramIndex":
        from scipy import sparse

        vocab: dict[str, int] = {}
        per_doc: list[np.ndarray] = []
        doc_ids: list[str] = []
        for d in docs:
            ids = np.fromiter((vocab.setdefault(w, len(vocab)) for p in d.paragraphs for w in words(p.text)), dtype=np.int64)
            per_doc.append(ids)
            doc_ids.append(d.doc_id)
        V = max(len(vocab), 1)
        grams = [cls._grams(ids, n, V) for ids in per_doc]
        features = np.unique(np.concatenate([g for g in grams if len(g)] or [np.zeros(0, np.int64)]))
        rows, cols, tfs, lens = [], [], [], np.array([len(g) for g in grams], dtype=np.float64)
        for i, g in enumerate(grams):
            if not len(g):
                continue
            u, c = np.unique(g, return_counts=True)
            rows.append(np.full(len(u), i, dtype=np.int32))
            cols.append(np.searchsorted(features, u).astype(np.int32))
            tfs.append(c.astype(np.float32))
        rows, cols, tfs = np.concatenate(rows), np.concatenate(cols), np.concatenate(tfs)
        N = len(doc_ids)
        df = np.bincount(cols, minlength=len(features)).astype(np.float64)
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
        avg = lens.mean() if N else 1.0
        norm = 1 - b + b * lens / (avg or 1.0)
        w = idf[cols] * tfs * (k1 + 1) / (tfs + k1 * norm[rows])
        matrix = sparse.csc_matrix((w.astype(np.float32), (rows, cols)), shape=(N, len(features)))
        return cls(n=n, vocab=vocab, features=features, matrix=matrix, doc_ids=doc_ids, k1=k1, b=b)

    @staticmethod
    def _grams(ids: np.ndarray, n: int, V: int) -> np.ndarray:
        if len(ids) < n:
            return np.zeros(0, dtype=np.int64)
        g = ids[: len(ids) - n + 1].copy()
        for k in range(1, n):
            g = g * V + ids[k: len(ids) - n + 1 + k]
        return g

    # ------------------------------------------------------------------ query
    def query_grams(self, text: str) -> np.ndarray:
        """Column indexes of the query's distinct n-grams that occur in the collection."""
        ids = [self.vocab.get(w, -1) for w in words(text)]
        V = max(len(self.vocab), 1)
        out, run = [], []
        for i in ids:                                    # unknown words break the n-gram chain
            if i < 0:
                run = []
                continue
            run.append(i)
            if len(run) >= self.n:
                g = 0
                for x in run[-self.n:]:
                    g = g * V + x
                out.append(g)
        if not out:
            return np.zeros(0, dtype=np.int64)
        q = np.unique(np.array(out, dtype=np.int64))
        if not len(self.features):
            return np.zeros(0, dtype=np.int64)
        pos = np.searchsorted(self.features, q)
        safe = np.minimum(pos, len(self.features) - 1)
        return pos[(pos < len(self.features)) & (self.features[safe] == q)]

    def score(self, text: str, candidates: set[str] | None = None, top: int | None = None) -> dict[str, float]:
        cols = self.query_grams(text)
        if not len(cols):
            return {}
        s = np.asarray(self.matrix[:, cols].sum(axis=1)).ravel()
        nz = np.nonzero(s)[0]
        if top and len(nz) > top:
            nz = nz[np.argsort(-s[nz])[:top]]
        out = {self.doc_ids[i]: float(s[i]) for i in nz}
        if candidates is not None:
            out = {d: v for d, v in out.items() if d in candidates}
        return out

    def n_features(self) -> int:
        return len(self.features)


def describe(idx: NgramIndex) -> str:
    nnz = idx.matrix.nnz
    return (f"{idx.n}-gram index: {len(idx.doc_ids)} docs, {idx.n_features():,} distinct {idx.n}-grams, "
            f"{nnz:,} postings, {nnz * 8 / 2**20:.0f} MB")


def _check() -> None:                                    # tiny self-test
    from kanoon_bridge.schema import Document, DocType, Paragraph

    docs = [Document(doc_id=str(i), doc_type=DocType.PRECEDENT, title="", paragraphs=[Paragraph(t, "facts", 0)])
            for i, t in enumerate(["the accused stabbed the victim with a knife", "the victim was stabbed by the accused",
                                   "bail is the rule and jail the exception"])]
    idx = NgramIndex.build(docs)
    s = idx.score("he stabbed the victim with a knife near the house")
    assert max(s, key=s.get) == "0", s
    assert math.isfinite(sum(s.values()))
