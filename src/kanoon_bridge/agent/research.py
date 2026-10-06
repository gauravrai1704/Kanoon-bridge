"""ResearchAgent: plan -> execute -> fuse -> (reflect -> execute -> fuse) -> AgentResult.  [working orchestration]

    from kanoon_bridge.agent.research import ResearchAgent
    agent = ResearchAgent.load(engine)
    out = agent.run(Query("mere bhai ko chaku maara", state="delhi", incident_date="2025-03-01"))
    out.precedents[:5]; out.trace

Unfinished planner rules / reflection are skipped and recorded in the trace, so with only
the 'original' sub-query the agent returns the same ranking as the core (a useful sanity
check in eval/agent_eval.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kanoon_bridge.agent.executor import SubResult, run_plan
from kanoon_bridge.agent.fuse import fuse_subresults
from kanoon_bridge.agent.plan import Plan, RulePlanner
from kanoon_bridge.agent import reflect
from kanoon_bridge.config import Config, load_config
from kanoon_bridge.rank.topk import to_scored, top_k
from kanoon_bridge.schema import DocType, Query, ScoredDoc


@dataclass
class AgentResult:
    query: Query
    statutes: list[ScoredDoc] = field(default_factory=list)
    precedents: list[ScoredDoc] = field(default_factory=list)
    plans: list[Plan] = field(default_factory=list)              # one per round
    subresults: list[SubResult] = field(default_factory=list)    # every search the agent ran
    trace: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n_searches(self) -> int:
        return len(self.subresults)


@dataclass
class ResearchAgent:
    engine: object                       # search.SearchEngine
    cfg: Config
    planner: object = field(default_factory=RulePlanner)
    docs: dict | None = None             # doc_id -> Document, for reformulation (index/docstore.py)

    @classmethod
    def load(cls, engine, cfg: Config | None = None) -> "ResearchAgent":
        cfg = cfg or load_config()
        docs = None
        try:
            from kanoon_bridge.index.docstore import DocStore

            docs = DocStore.load(cfg)
        except FileNotFoundError:
            pass
        return cls(engine=engine, cfg=cfg, docs=docs)

    def _idf(self, terms) -> dict[str, float]:
        try:
            whole = self.engine.precedent_index.whole
            return {t: whole.idf(t) for t in terms}
        except (AttributeError, NotImplementedError):
            return {}

    def run(self, query: Query, options=None) -> AgentResult:
        acfg = self.cfg.agent
        out = AgentResult(query=query)
        aq = self.engine.analyzer.analyze(query)
        idf = self._idf(aq.weighted_terms())

        plan = self.planner.plan(query, aq, idf)
        round_no = 1
        while True:
            out.plans.append(plan)
            for sq in plan.subqueries:
                out.trace.append((f"round {round_no} {sq.sq_id}", f"[{sq.kind}] {sq.query.text}  - {sq.rationale}"))
            for note in plan.notes:
                out.trace.append((f"round {round_no} note", note))
            out.subresults += run_plan(self.engine, plan, options, depth=acfg.depth)

            fused_prec = fuse_subresults(out.subresults, "precedent", acfg.rrf_k)
            try:
                retry, reason = reflect.should_reformulate(fused_prec, aq, idf, round_no, acfg.max_rounds)
            except NotImplementedError as err:
                retry, reason = False, f"reflection skipped: {err}"
            out.trace.append((f"round {round_no} reflect", reason))
            if not retry:
                break
            try:
                new_sqs = reflect.reformulate(aq, fused_prec, self.docs, plan, acfg.prf_top_docs, acfg.prf_terms)
            except NotImplementedError as err:
                out.trace.append((f"round {round_no} reformulate", f"skipped: {err}"))
                break
            plan = Plan(subqueries=new_sqs)
            round_no += 1

        k = (options.top_k if options is not None else self.cfg.search.top_k)
        out.precedents = to_scored(top_k(fused_prec, k), DocType.PRECEDENT)
        out.statutes = to_scored(top_k(fuse_subresults(out.subresults, "statute", acfg.rrf_k), k), DocType.STATUTE)
        out.trace.append(("searches", str(out.n_searches)))
        return out
