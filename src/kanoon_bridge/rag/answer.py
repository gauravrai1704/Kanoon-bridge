"""Layer 3 orchestration: retrieval result -> checked, cited answer.  [RAG owner — working wiring]

    from kanoon_bridge.rag.answer import answer
    ans = answer(agent_result_or_search_result, docs, cfg)
    ans.text, ans.flags, ans.abstained

Order (the cheap IR checks wrap the expensive LLM call):
    1. abstain.decide          QPP says weak?  -> stop, no LLM call
    2. chunker.make_chunks     top statutes + ratio/decision paragraphs, numbered
    3. generate.generate       LLM writes 3-5 cited sentences
    4. citation_check.check    tf-idf cosine of each sentence vs its cited chunk
    5. version_check.check     wrong code for the incident date, bare colliding numbers

Works on either a search.SearchResult (layer 1) or an agent.research.AgentResult (layer 2):
both expose .statutes and .precedents.
"""

from __future__ import annotations

from kanoon_bridge.config import Config, load_config
from kanoon_bridge.rag import abstain, chunker, citation_check, generate, version_check
from kanoon_bridge.rag.generate import Answer


def answer(result, docs, cfg: Config | None = None, idf: dict[str, float] | None = None,
           resolver=None, normalizer=None) -> Answer:
    cfg = cfg or load_config()
    rcfg = cfg.rag
    query = getattr(result, "query", None)
    query = getattr(query, "query", query)          # SearchResult.query is an AnalyzedQuery
    idf = idf or {}

    abstained, reason = abstain.decide(result, idf)
    if abstained:
        return Answer(text=f"Not enough grounding in the indexed law to answer this ({reason}).", abstained=True)

    chunks = chunker.make_chunks(result, docs, max_chunks=rcfg.max_chunks, max_chars=rcfg.max_chunk_chars)
    ans = generate.generate(query.text, chunks,
                            incident_date=str(query.incident_date or "unknown"), state=query.state or "unknown")
    ans = citation_check.check(ans, chunks, idf, threshold=rcfg.support_threshold)
    if resolver is not None and normalizer is not None:
        ans = version_check.check(ans, query.incident_date, resolver, normalizer)
    return ans
