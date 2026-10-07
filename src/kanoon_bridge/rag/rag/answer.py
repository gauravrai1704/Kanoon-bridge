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
from kanoon_bridge.rag._util import idf_lookup
from kanoon_bridge.rag.generate import Answer


def _query_of(result):
    q = getattr(result, "query", None)
    return getattr(q, "query", q)          # SearchResult.query is an AnalyzedQuery


def answer(result, docs, cfg: Config | None = None, idf=None, resolver=None, normalizer=None,
           generator: str | None = None, closed_book: bool = False, check_abstain: bool = True,
           abstain_idf=None) -> Answer:
    """`idf` scores support (precedent index); `abstain_idf` (default: idf) judges whether the
    question has a specific term - RagPipeline passes max(precedent idf, statute idf)."""
    cfg = cfg or load_config()
    rcfg = cfg.rag
    query = _query_of(result)
    analyzed = getattr(result, "analyzed", None) or result.query     # AgentResult keeps it in .analyzed
    qterms = list(analyzed.weighted_terms()) if hasattr(analyzed, "weighted_terms") else None

    if check_abstain and not closed_book:
        abstained, reason = abstain.decide(result, abstain_idf or idf, query_terms=qterms,
                                           min_idf=rcfg.get("abstain_min_idf", 0.3))
        if abstained:
            return Answer(text=f"Not enough grounding in the indexed law to answer this ({reason}).",
                          abstained=True, generator="abstain")

    chunks = chunker.make_chunks(result, docs, max_chunks=rcfg.max_chunks, max_chars=rcfg.max_chunk_chars,
                                 query_terms=qterms)
    # (a question that names a section or offence is in scope by construction: skip this test)
    if check_abstain and not closed_book and hasattr(analyzed, "tokens") and not getattr(analyzed, "offence_ids", None):
        fixes = {c.source: c.term for c in getattr(analyzed, "corrections", []) if c.applied}
        content = [fixes.get(t, t) for t in analyzed.tokens if not t.startswith(("sec:", "off:")) and not t.isdigit()]
        if analyzed.detected_lang != "en":                 # Hindi/Hinglish: judge the English expansions
            content = [t for t in analyzed.expanded_terms if not t.startswith(("sec:", "off:"))] or content
        look = idf_lookup(abstain_idf or idf)
        # the best SINGLE source must cover the question: words scattered over several unrelated
        # sources ("rate" in a tax Act, "food" in an adulteration Act) do not ground an answer
        cov = max((abstain.coverage(content, [c.text], look, unseen_idf=rcfg.get("abstain_unseen_idf", 3.0))
                   for c in chunks), default=0.0)
        if content and cov < rcfg.get("abstain_min_coverage", 0.5):
            return Answer(text="Not enough grounding in the indexed law to answer this "
                               f"(the best source covers {cov:.0%} of the question's terms).",
                          abstained=True, generator="abstain")

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
    statute_idf: object = None

    @classmethod
    def load(cls, engine, cfg: Config | None = None, docs=None) -> "RagPipeline":
        cfg = cfg or engine.cfg
        if docs is None:
            from kanoon_bridge.index.docstore import DocStore

            docs = DocStore.load(cfg)
        res = engine.analyzer.text_res
        return cls(cfg=cfg, docs=docs, idf=engine.precedent_index.whole.idf,
                   resolver=res.resolver, normalizer=res.normalizer, statute_idf=engine.statute_index.whole.idf)

    def _abstain_idf(self, term: str) -> float:
        return max(self.idf(term), self.statute_idf(term) if self.statute_idf else 0.0)

    def answer(self, result, generator: str | None = None, closed_book: bool = False,
               use_normalizer: bool = True) -> Answer:
        return answer(result, self.docs, self.cfg, self.idf, self.resolver,
                      self.normalizer if use_normalizer else None, generator, closed_book,
                      abstain_idf=self._abstain_idf)


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
