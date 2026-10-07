"""Explicit relevance feedback (Rocchio).  [owner: Gaurav — working]

The user marks results as relevant (app checkbox, CLI --feedback P01,P06); the query moves
toward the centroid of those documents (Rocchio, "Relevance feedback in information retrieval",
SMART system 1971; IIR ch. 9):

    q' = alpha * q + beta * centroid(relevant) - gamma * centroid(non-relevant)

Implemented on top of the core retriever: the centroid's best terms (ratio/decision zones,
(1 + log10 tf) x idf, terms in >= 2 documents, never a section token of the superseded code)
become SearchOptions.extra_terms with weight beta; terms that only the non-relevant documents
carry get dropped (gamma > 0 drops at most 3). The pseudo-relevance feedback round of the agent
(agent/reflect.py) is the same mechanism with the top documents assumed relevant.
"""

from __future__ import annotations

from kanoon_bridge.schema import Code


def rocchio_options(engine, aq, docs, relevant: list[str], non_relevant: list[str] | None = None,
                    n_terms: int = 10, beta: float = 0.75, gamma: float = 0.25) -> dict:
    """SearchOptions overrides ({"extra_terms": ..., "drop_terms": ...}) for a feedback search."""
    from kanoon_bridge.agent.reflect import feedback_terms
    from kanoon_bridge.text.pipeline import analyze_text

    res = engine.analyzer.text_res
    whole = engine.precedent_index.whole

    def analyze(text: str) -> list[str]:
        return analyze_text(text, res)

    pos = feedback_terms(relevant, docs, analyze, whole.idf)
    neg = feedback_terms(non_relevant or [], docs, analyze, whole.idf) if gamma > 0 else {}
    query_terms = aq.weighted_terms()
    superseded = {Code.IPC: ("sec:bns:", "sec:bnss:", "sec:bsa:"),
                  Code.BNS: ("sec:ipc:", "sec:crpc:", "sec:iea:")}.get(aq.code_in_force)
    combined = {t: w / max(1, len(relevant)) - gamma * neg.get(t, 0.0) / max(1, len(non_relevant or []))
                for t, w in pos.items()}
    extra = []
    for t, w in sorted(combined.items(), key=lambda kv: -kv[1]):
        if w <= 0 or t in query_terms or t.isdigit() or len(t) < 3 or whole.df(t) < 2:
            continue
        if superseded and t.startswith(superseded):
            continue
        extra.append(t)
        if len(extra) >= n_terms:
            break
    drop = [t for t in query_terms if neg.get(t, 0) > 0 and pos.get(t, 0) == 0 and not t.startswith(("sec:", "off:"))][:3]
    return {"extra_terms": {t: beta for t in extra}, "drop_terms": drop}
