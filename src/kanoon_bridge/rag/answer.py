"""Layer 3 orchestration: retrieval result -> checked, cited answer.  [owner: Gaurav — working]

    from kanoon_bridge.rag.answer import answer, RagPipeline
    rag = RagPipeline.load(engine)                    # docs, idf, resolver, normaliser
    ans = rag.answer(result)                          # SearchResult or AgentResult
    ans.text, ans.flags, ans.abstained, ans.supported_rate

Order (the cheap IR checks wrap the expensive LLM call):
    1. abstain.decide          QPP says weak?  -> stop, no LLM call
    2. chunker.make_chunks     top statutes + ratio/decision paragraphs, numbered
    3. generate.generate       Claude (or the offline extractive generator) writes cited sentences
    4. citation_check.check    tf-idf cosine of each sentence vs its cited chunk
    5. version_check.check     wrong code for the incident date, bare colliding numbers
"""

from __future__ import annotations

from dataclasses import dataclass

from kanoon_bridge.config import Config, load_config
from kanoon_bridge.rag import abstain, chunker, citation_check, generate, version_check
from kanoon_bridge.rag.generate import Answer


def _query_of(result):
    q = getattr(result, "query", None)
    return getattr(q, "query", q)          # SearchResult.query is an AnalyzedQuery


def answer(result, docs, cfg: Config | None = None, idf=None, resolver=None, normalizer=None,
           generator: str | None = None, closed_book: bool = False, check_abstain: bool = True) -> Answer:
    cfg = cfg or load_config()
    rcfg = cfg.rag
    query = _query_of(result)
    qterms = list(result.query.weighted_terms()) if hasattr(result.query, "weighted_terms") else None

    if check_abstain and not closed_book:
        abstained, reason = abstain.decide(result, idf, query_terms=qterms)
        if abstained:
            return Answer(text=f"Not enough grounding in the indexed law to answer this ({reason}).",
                          abstained=True, generator="abstain")

    chunks = chunker.make_chunks(result, docs, max_chunks=rcfg.max_chunks, max_chars=rcfg.max_chunk_chars,
                                 query_terms=qterms)
    ans = generate.generate(query.text, chunks, incident_date=str(query.incident_date or "unknown"),
                            state=query.state or "unknown", generator=generator or rcfg.get("generator", "auto"),
                            model=rcfg.model or None, closed_book=closed_book)
    ans.chunks = chunks
    ans = citation_check.check(ans, chunks, idf, threshold=rcfg.support_threshold, any_chunk=closed_book)
    if resolver is not None:
        ans = version_check.check(ans, query.incident_date, resolver, normalizer)
    return ans


@dataclass
class RagPipeline:
    """Everything answer() needs, loaded once from a SearchEngine."""

    cfg: Config
    docs: object
    idf: object
    resolver: object
    normalizer: object

    @classmethod
    def load(cls, engine, cfg: Config | None = None, docs=None) -> "RagPipeline":
        cfg = cfg or engine.cfg
        if docs is None:
            from kanoon_bridge.index.docstore import DocStore

            docs = DocStore.load(cfg)
        res = engine.analyzer.text_res
        return cls(cfg=cfg, docs=docs, idf=engine.precedent_index.whole.idf,
                   resolver=res.resolver, normalizer=res.normalizer)

    def answer(self, result, generator: str | None = None, closed_book: bool = False,
               use_normalizer: bool = True) -> Answer:
        return answer(result, self.docs, self.cfg, self.idf, self.resolver,
                      self.normalizer if use_normalizer else None, generator, closed_book)


def render(ans: Answer) -> str:
    """Plain-text answer with sources and flags (CLI)."""
    lines = [ans.text]
    if ans.chunks and not ans.abstained:
        lines.append("\nSources:")
        lines += [f"  {c.header()}" for c in ans.chunks]
    if ans.flags:
        lines.append("\nChecks:")
        lines += [f"  ! {f}" for f in ans.flags]
    elif not ans.abstained:
        lines.append("\nChecks: every sentence supported; no version problems")
    lines.append(f"\n(generator: {ans.generator}; not legal advice)")
    return "\n".join(lines)
