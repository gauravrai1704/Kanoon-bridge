"""Reflection: decide whether to try again, and how.  [layer 2 — working]

    should_reformulate   QPP on the round's evidence: too few fused results, or low retrieval
                         confidence (rank/qpp.py: max idf of the query terms + the top-1/top-2
                         gap of the ORIGINAL sub-query's core scores; RRF scores are near-ties
                         by construction, so they are not used for the gap). Never more than
                         agent.max_rounds rounds.
    reformulate          pseudo-relevance feedback (Rocchio-style, a lecture concept): take the
                         top-n fused precedents, score their ratio/decision terms by
                         (1 + log10 tf) x idf summed over those docs (the centroid), add the
                         best n_terms not already in the query with weight beta, and drop the
                         query's lowest-idf content term. Section tokens of the superseded code
                         and terms in fewer than 2 documents are never added.

The trace records the reason for each extra round, so the video can show it.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Callable

from kanoon_bridge.agent.plan import Plan, SubQuery, _is_content
from kanoon_bridge.rank import qpp
from kanoon_bridge.schema import AnalyzedQuery, Code

FEEDBACK_ZONES = ("ratio", "decision")


def should_reformulate(fused: dict[str, float], aq: AnalyzedQuery, idf: dict[str, float],
                       round_no: int, max_rounds: int, core_scores: list[float] | None = None,
                       min_results: int = 10, min_confidence: float = 0.35) -> tuple[bool, str]:
    """(retry?, reason)."""
    if round_no >= max_rounds:
        return False, f"reached max rounds ({max_rounds})"
    if len(fused) < min_results:
        return True, f"only {len(fused)} precedents found (< {min_results})"
    vals = [v for v in (idf.get(t, 0.0) for t in aq.weighted_terms()) if v > 0]
    f = qpp.QPPFeatures(max_idf=max(vals, default=0.0))
    f = qpp.post_retrieval(f, list(core_scores or []))
    conf = qpp.confidence(f)
    detail = f"confidence {conf:.2f} (max idf {f.max_idf:.2f}, top gap {f.score_gap:.2f})"
    if conf < min_confidence:
        return True, f"weak retrieval: {detail} < {min_confidence}"
    return False, f"retrieval looks fine: {detail}"


def feedback_terms(doc_ids: list[str], docs, analyze: Callable[[str], list[str]], idf: Callable[[str], float],
                   max_chars: int = 6000) -> Counter:
    """Rocchio centroid (unnormalised) of the feedback docs: term -> sum of (1 + log10 tf) x idf."""
    centroid: Counter = Counter()
    for d in doc_ids:
        doc = docs.get(d) if docs is not None else None
        if doc is None:
            continue
        text = "\n".join(p.text for p in doc.paragraphs if p.zone in FEEDBACK_ZONES) or doc.text
        for t, c in Counter(analyze(text[:max_chars])).items():
            w = idf(t)
            if w > 0:
                centroid[t] += (1 + math.log10(c)) * w
    return centroid


def reformulate(aq: AnalyzedQuery, fused: dict[str, float], docs, plan: Plan,
                top_n: int = 5, n_terms: int = 10, beta: float = 0.5, *,
                idf: Callable[[str], float] | None = None, df: Callable[[str], int] | None = None,
                analyze: Callable[[str], list[str]] | None = None, round_no: int = 2) -> list[SubQuery]:
    """New sub-queries for the next round (kind='reformulated'); [] when there is nothing to use.

    `docs`: doc_id -> Document (index/docstore.py); `idf`/`df`: precedent-index statistics;
    `analyze`: the shared text pipeline (text -> analysed terms).
    """
    if not docs or not fused or analyze is None or idf is None:
        return []
    top = sorted(fused, key=fused.get, reverse=True)[:top_n]
    centroid = feedback_terms(top, docs, analyze, idf)
    query_terms = aq.weighted_terms()
    superseded = {Code.IPC: ("sec:bns:", "sec:bnss:", "sec:bsa:"),
                  Code.BNS: ("sec:ipc:", "sec:crpc:", "sec:iea:")}.get(aq.code_in_force)
    candidates = []
    for t, w in centroid.most_common():
        if t in query_terms or t.isdigit() or len(t) < 3:
            continue
        if superseded and t.startswith(superseded):
            continue
        if df is not None and df(t) < 2:
            continue
        candidates.append(t)
        if len(candidates) >= n_terms:
            break
    content = [t for t in dict.fromkeys(aq.tokens) if _is_content(t) and idf(t) > 0]
    drop = [min(content, key=idf)] if len(content) > 3 else []
    if not candidates and not drop:
        return []
    original = next((sq for sq in plan.subqueries if sq.kind == "original"), plan.subqueries[0])
    q = copy.deepcopy(original.query)
    q.query_id = None
    return [SubQuery(sq_id=f"r{round_no}q0", kind="reformulated", query=q,
                     options={"extra_terms": {t: beta for t in candidates}, "drop_terms": drop},
                     rationale=f"pseudo-relevance feedback from top {len(top)}: +" + " +".join(candidates)
                               + (f"  -{drop[0]}" if drop else ""))]
