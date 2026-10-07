"""ResearchAgent: plan -> execute -> fuse -> (reflect -> execute -> fuse) -> AgentResult.  [layer 2 — working]

    from kanoon_bridge.agent.research import ResearchAgent
    agent = ResearchAgent.load(engine)
    out = agent.run(Query("mere bhai ko chaku maara", state="delhi", incident_date="2025-03-01"))
    out.precedents[:5]; out.trace

AgentResult has the same .statutes / .precedents / .query / .timings_ms surface as a
SearchResult, plus .analyzed (the AnalyzedQuery) and the plans and sub-results, so the CLI,
the app, the evaluator (eval/run_eval.run_queries) and the RAG layer (rag/answer.py) accept
either one.

Sanity check (eval/agent_eval.py, tests/test_agent.py): with only the 'original' sub-query,
the agent returns the same ranking as the core.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from kanoon_bridge.agent import reflect
from kanoon_bridge.agent.executor import SubResult, run_plan
from kanoon_bridge.agent.fuse import fuse
from kanoon_bridge.agent.plan import Plan, RulePlanner, make_planner
from kanoon_bridge.config import Config, load_config
from kanoon_bridge.rank.topk import to_scored, top_k
from kanoon_bridge.schema import AnalyzedQuery, DocType, Query, ScoredDoc


@dataclass
class AgentResult:
    query: Query
    statutes: list[ScoredDoc] = field(default_factory=list)
    precedents: list[ScoredDoc] = field(default_factory=list)
    plans: list[Plan] = field(default_factory=list)              # one per round
    subresults: list[SubResult] = field(default_factory=list)    # every search the agent ran
    trace: list[tuple[str, str]] = field(default_factory=list)
    analyzed: AnalyzedQuery | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def n_searches(self) -> int:
        return len(self.subresults)

    def found_by(self, doc_id: str) -> list[str]:
        """Which sub-queries retrieved `doc_id` (for the demo: 'p17 came from q2 boolean')."""
        return [f"{sr.subquery.sq_id}:{sr.subquery.kind}" for sr in self.subresults
                if doc_id in sr.ranking("precedent") or doc_id in sr.ranking("statute")]


@dataclass
class ResearchAgent:
    engine: object                       # search.SearchEngine
    cfg: Config
    planner: object = field(default_factory=RulePlanner)
    docs: dict | None = None             # doc_id -> Document, for reformulation (index/docstore.py)
    fusion: str = "rrf"                  # rrf | combsum

    @classmethod
    def load(cls, engine, cfg: Config | None = None, docs=None, fusion: str | None = None,
             planner=None) -> "ResearchAgent":
        cfg = cfg or getattr(engine, "cfg", None) or load_config()
        if docs is None:
            try:
                from kanoon_bridge.index.docstore import DocStore

                docs = DocStore.load(cfg)
            except FileNotFoundError:
                pass
        res = getattr(getattr(engine, "analyzer", None), "text_res", None)
        whole = getattr(getattr(engine, "precedent_index", None), "whole", None)
        planner = planner or make_planner(cfg, normalizer=getattr(res, "normalizer", None),
                                          df=getattr(whole, "df", None))
        return cls(engine=engine, cfg=cfg, planner=planner, docs=docs,
                   fusion=fusion or cfg.agent.get("fusion", "rrf"))

    # ------------------------------------------------------------------ index statistics
    def _whole(self):
        return getattr(getattr(self.engine, "precedent_index", None), "whole", None)

    def _idf(self, terms) -> dict[str, float]:
        whole = self._whole()
        if whole is None:
            return {}
        try:
            return {t: whole.idf(t) for t in terms}
        except NotImplementedError:
            return {}

    def _analyze(self):
        res = getattr(getattr(self.engine, "analyzer", None), "text_res", None)
        if res is None:
            return None
        from kanoon_bridge.text.pipeline import analyze_text

        return lambda text: analyze_text(text, res)

    # ------------------------------------------------------------------ the loop
    def run(self, query: Query, options=None) -> AgentResult:
        t0 = time.perf_counter()
        acfg = self.cfg.agent
        out = AgentResult(query=query)
        aq = self.engine.analyzer.analyze(query)
        out.analyzed = aq
        idf = self._idf(aq.weighted_terms())

        plan = self.planner.plan(query, aq, idf)
        round_no = 1
        fused_prec: dict[str, float] = {}
        while True:
            out.plans.append(plan)
            for sq in plan.subqueries:
                out.trace.append((f"round {round_no} {sq.sq_id}", f"[{sq.kind}] {sq.query.text}  - {sq.rationale}"))
            for note in plan.notes:
                out.trace.append((f"round {round_no} note", note))
            new = run_plan(self.engine, plan, options, depth=acfg.depth)
            out.subresults += new
            for sr in new:
                n = len(sr.ranking(sr.subquery.target))
                out.trace.append((f"round {round_no} {sr.subquery.sq_id} hits", f"{n} {sr.subquery.target}s"))

            fused_prec = fuse(out.subresults, "precedent", self.fusion, acfg.rrf_k)
            core = next((sr for sr in new if sr.subquery.kind in ("original", "reformulated")), new[0] if new else None)
            retry, reason = reflect.should_reformulate(
                fused_prec, aq, idf, round_no, acfg.max_rounds, core.scores("precedent") if core else [],
                min_results=acfg.get("min_results", 10), min_confidence=acfg.get("retry_confidence", 0.35))
            out.trace.append((f"round {round_no} reflect", reason))
            if not retry:
                break
            whole = self._whole()
            new_sqs = reflect.reformulate(aq, fused_prec, self.docs, plan, acfg.prf_top_docs, acfg.prf_terms,
                                          acfg.get("prf_beta", 0.5), idf=getattr(whole, "idf", None),
                                          df=getattr(whole, "df", None), analyze=self._analyze(),
                                          round_no=round_no + 1)
            if not new_sqs:
                out.trace.append((f"round {round_no} reformulate", "nothing to add (no documents or feedback terms)"))
                break
            plan = Plan(subqueries=new_sqs, prefix=f"r{round_no + 1}q")
            round_no += 1

        k = options.top_k if options is not None else self.cfg.search.top_k
        out.precedents = to_scored(top_k(fused_prec, k), DocType.PRECEDENT,
                                   {d: {self.fusion: s} for d, s in fused_prec.items()})
        fused_stat = fuse(out.subresults, "statute", self.fusion, acfg.rrf_k)
        out.statutes = to_scored(top_k(fused_stat, k), DocType.STATUTE)
        out.trace.append(("searches", str(out.n_searches)))
        out.timings_ms = {"agent": (time.perf_counter() - t0) * 1000}
        return out
