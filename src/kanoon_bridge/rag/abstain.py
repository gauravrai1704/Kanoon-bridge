"""Decide BEFORE generation whether retrieval is strong enough to ground an answer.  [owner: Gaurav — working]

Uses rank/qpp.py signals: max idf of the query terms (is anything specific being asked?) and the
head of each ranked list (a clear winner, or a flat list of near-ties?). Abstain when the query
has no specific term, nothing was retrieved, or BOTH the statute and precedent heads are flat.
Metric: abstention precision on questions we know the corpus cannot answer (rag_questions.jsonl
rows with "answerable": false).
"""

from __future__ import annotations

from kanoon_bridge.rag._util import idf_lookup
from kanoon_bridge.rank import qpp


def _query_terms(result) -> list[str]:
    q = getattr(result, "query", None)
    if hasattr(q, "weighted_terms"):                 # AnalyzedQuery (layer 1)
        return list(q.weighted_terms())
    from kanoon_bridge.rag._util import terms        # Query (layer 2 AgentResult)

    return terms(getattr(q, "text", "") or "")


def decide(result, idf=None, query_terms: list[str] | None = None, min_idf: float = 0.3,
           gap_threshold: float = 0.05, top_threshold: float = 1.1) -> tuple[bool, str]:
    """(abstain?, reason). `idf`: dict or callable term -> idf (precedent index)."""
    look = idf_lookup(idf)
    qterms = query_terms if query_terms is not None else _query_terms(result)
    vals = [look(t) for t in qterms]
    vals = [v for v in vals if v > 0]
    max_idf = max(vals, default=0.0)
    if idf is not None and max_idf < min_idf:
        return True, f"no specific query term (max idf {max_idf:.2f})"
    if not result.statutes and not result.precedents:
        return True, "nothing retrieved"
    if hasattr(result, "subresults"):
        # agent (RRF) scores are near-ties by construction; only the checks above apply
        return False, f"ok (max idf {max_idf:.2f}; agent result)"
    flat = []
    for name, hits in (("statutes", result.statutes), ("precedents", result.precedents)):
        f = qpp.post_retrieval(qpp.QPPFeatures(max_idf=max_idf or 1.0), [h.score for h in hits])
        flat.append(not hits or qpp.should_abstain(f, gap_threshold, top_threshold, min_idf=0.0))
    if all(flat):
        return True, "flat score head in both statutes and precedents"
    return False, f"ok (max idf {max_idf:.2f})"
