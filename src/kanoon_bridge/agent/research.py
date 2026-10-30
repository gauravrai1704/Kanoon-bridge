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
    memo: object | None = None                                   # agent v2: memo.ResearchMemo

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


# --------------------------------------------------------------------------- agent v2


@dataclass
class ResearchAgentV2(ResearchAgent):
    """Routed, verified, issue-driven research (agent/router.py, verify.py, issues.py, memo.py).

        route     what kind of question, about which code family; which sub-queries are worth it
        plan      the RulePlanner's sub-queries the route allows (a pasted judgment gets the
                  cross-code variant only, at low fusion weight, so the core ranking dominates)
        execute + fuse + reflect   as in v1
        votes     statutes the top precedents cite, read in the code in force; used only when no
                  candidate heading matches the question (a fact narrative, not an offence name)
        verify    inspect the top statute candidates: right code family, in force on the date,
                  heading that matches the question (agent/verify.py)
        issues    split the verified provision into ingredients; look for evidence of each in
                  the precedents; one targeted search per open issue; stop at min_coverage
        memo      law in force (with drift note and computed punishment), issues and evidence

    `run` returns an AgentResult with `.memo` filled; every decision is in `.trace`.
    """

    def run(self, query: Query, options=None) -> AgentResult:
        from kanoon_bridge.agent.router import route as route_query

        t0 = time.perf_counter()
        acfg = self.cfg.agent
        out = AgentResult(query=query)
        aq = self.engine.analyzer.analyze(query)
        out.analyzed = aq
        idf = self._idf(aq.weighted_terms())
        rt = route_query(query, aq, self.cfg.ngram.get("min_query_words", 50))
        out.trace.append(("route", rt.describe()))

        plan = self.planner.plan(query, aq, idf) if hasattr(self.planner, "plan") else Plan()
        keep = {"original", "llm", *rt.rules}
        dropped = [sq.kind for sq in plan.subqueries if sq.kind not in keep]
        plan.subqueries = [sq for sq in plan.subqueries if sq.kind in keep]
        if rt.kind == "case_text":
            for sq in plan.subqueries:
                if sq.kind != "original":
                    sq.weight = min(sq.weight, acfg.get("case_text_weight", 0.3))
        if dropped:
            out.trace.append(("route", "skipped sub-queries: " + ", ".join(dropped)))

        # rounds 1..n: v1's loop (execute, fuse, reflect with pseudo-relevance feedback)
        round_no = 1
        while True:
            out.plans.append(plan)
            for sq in plan.subqueries:
                out.trace.append((f"round {round_no} {sq.sq_id}", f"[{sq.kind}] {sq.query.text}  - {sq.rationale}"))
            new = run_plan(self.engine, plan, options, depth=acfg.depth)
            out.subresults += new
            fused_prec = fuse(out.subresults, "precedent", self.fusion, acfg.rrf_k)
            core = next((sr for sr in new if sr.subquery.kind in ("original", "reformulated")), new[0] if new else None)
            retry, reason = reflect.should_reformulate(
                fused_prec, aq, idf, round_no, acfg.max_rounds, core.scores("precedent") if core else [],
                min_results=acfg.get("min_results", 10), min_confidence=acfg.get("retry_confidence", 0.35))
            out.trace.append((f"round {round_no} reflect", reason))
            if not retry or rt.kind == "case_text":
                break
            whole = self._whole()
            new_sqs = reflect.reformulate(aq, fused_prec, self.docs, plan, acfg.prf_top_docs, acfg.prf_terms,
                                          acfg.get("prf_beta", 0.5), idf=getattr(whole, "idf", None),
                                          df=getattr(whole, "df", None), analyze=self._analyze(),
                                          round_no=round_no + 1)
            if not new_sqs:
                break
            plan = Plan(subqueries=new_sqs, prefix=f"r{round_no + 1}q")
            round_no += 1

        k = options.top_k if options is not None else self.cfg.search.top_k
        self._refine_and_search_family(query, aq, rt, out, options)
        fused_stat = fuse(out.subresults, "statute", self.fusion, acfg.rrf_k)
        votes = self._citation_votes(fused_prec, aq, rt)
        if votes and acfg.get("vote_gate", 0.3) > 0:
            from kanoon_bridge.agent.verify import corrected, specificity

            head = sorted(fused_stat, key=fused_stat.get, reverse=True)[:5]
            best = max((specificity(corrected(query.text, aq), self._title(d)) for d in head), default=0.0)
            if best >= acfg.get("vote_gate", 0.3):
                out.trace.append(("citation votes", f"not used: the question matches a statute heading "
                                                    f"(word overlap {best:.2f})"))
                votes = {}
        if votes:
            w = acfg.get("vote_weight", 1.0)
            ranked = sorted(votes, key=votes.get, reverse=True)
            for r, d in enumerate(ranked, start=1):
                fused_stat[d] = fused_stat.get(d, 0.0) + w / (acfg.rrf_k + r)
            out.trace.append(("citation votes", "statutes the top precedents apply, in the code in force: "
                              + ", ".join(self._ref(d).upper() for d in ranked[:5])))
        if acfg.get("toc_weight", 0) > 0 and rt.kind != "case_text":
            fused_stat = self._toc_navigate(query, aq, rt, fused_stat, out)
        statutes = to_scored(top_k(fused_stat, max(k, 20)), DocType.STATUTE)
        statutes, notes = self._verifier().rerank(statutes, query, aq, self._family(rt), self.engine, self._title,
                                                  fact_question=rt.kind == "fact_question")
        out.trace += [("verify", n) for n in notes] or [("verify", "top statute kept")]
        out.statutes = statutes[:k]

        memo = self._memo(query, aq, rt, out.statutes)
        if rt.decompose and out.statutes:
            fused_prec = self._issues(query, aq, out, memo, fused_prec, options)
        out.precedents = to_scored(top_k(fused_prec, k), DocType.PRECEDENT,
                                   {d: {self.fusion: s} for d, s in fused_prec.items()})
        memo.searches = out.n_searches
        out.memo = memo
        out.trace.append(("searches", str(out.n_searches)))
        out.timings_ms = {"agent": (time.perf_counter() - t0) * 1000}
        return out

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _family(rt) -> str:
        return rt.family

    def _refine_and_search_family(self, query: Query, aq, rt, out: AgentResult, options) -> None:
        """Decide the code family from round 1 (router.refine_family), then search the statutes of
        that family's code in force (a statutes-only sub-query with a code filter)."""
        from kanoon_bridge.agent.plan import SubQuery, _variant
        from kanoon_bridge.agent.router import refine_family

        core = next((sr for sr in out.subresults if sr.subquery.kind == "original"), None)
        refs = [self._ref(d) for d in core.ranking("statute")[:5]] if core else []
        metas = getattr(self.engine.facets, "metas", {})
        prec_codes = [[c.split(":")[0] for c in metas[d].statutes_cited] for d in
                      (core.ranking("precedent")[:10] if core else []) if d in metas]
        rt.family, why = refine_family(rt, query, aq, refs, precedent_codes=prec_codes)
        out.trace.append(("family", f"{rt.family} law ({why})"))
        fam_code = self._family_code(rt, aq)
        if not fam_code or rt.kind == "case_text":
            return
        fq = _variant(query, query.text)
        fq.filters = {**fq.filters, "code": fam_code}
        plan = Plan(subqueries=[SubQuery("f0", "family_statutes", fq, target="statute",
                                         weight=self.cfg.agent.get("family_weight", 1.0),
                                         rationale=f"statutes of the {fam_code.upper()} only ({rt.family} law in force)")])
        out.plans.append(plan)
        out.trace.append(("f0", f"[family_statutes] {fq.text}  - {plan.subqueries[0].rationale}"))
        out.subresults += run_plan(self.engine, plan, options, depth=self.cfg.agent.depth)

    @staticmethod
    def _family_code(rt, aq) -> str | None:
        """The code of the routed family in force on the incident date (None without a date)."""
        if aq.code_in_force.value not in ("ipc", "bns"):
            return None
        new = aq.code_in_force.value == "bns"
        return {"penal": ("ipc", "bns"), "procedure": ("crpc", "bnss"), "evidence": ("iea", "bsa")}[rt.family][new]

    _toc = None                               # agent/toc.TableOfContents, built once per process

    def _toc_navigate(self, query: Query, aq, rt, fused_stat: dict[str, float], out: AgentResult) -> dict[str, float]:
        """STAIR-style navigation (agent/toc.py): choose chapters by their table-of-contents
        entries, then favour the chosen chapters' sections."""
        from kanoon_bridge.agent.toc import TableOfContents
        from kanoon_bridge.agent.verify import corrected, specificity

        acfg = self.cfg.agent
        if ResearchAgentV2._toc is None:
            try:
                ResearchAgentV2._toc = TableOfContents.build(self.cfg)
            except FileNotFoundError as err:
                out.trace.append(("toc", f"not used: {err}"))
                ResearchAgentV2._toc = False
        toc = ResearchAgentV2._toc
        if not toc:
            return fused_stat
        pair = {"penal": ("ipc", "bns"), "procedure": ("crpc", "bnss"), "evidence": ("iea", "bsa")}[self._family(rt)]
        code = self._family_code(rt, aq)
        text = corrected(query.text, aq)
        chosen = toc.navigate(text, {code} if code else set(pair), self.engine.statute_index.whole.idf,
                              acfg.get("toc_chapters", 2))
        if not chosen:
            return fused_stat
        doc_of = getattr(self, "_doc_of_ref", None)
        if doc_of is None:
            doc_of = {}
            for d, ts in (getattr(self.engine.bridge, "statute_terms", {}) or {}).items():
                for t in ts:
                    if t.startswith("sec:"):
                        doc_of.setdefault(t[4:], d)
            self._doc_of_ref = doc_of
        out_stat = dict(fused_stat)
        w, rk = acfg.get("toc_weight", 1.0), acfg.rrf_k
        for rank, (ci, _) in enumerate(chosen, start=1):
            ch = toc.chapters[ci]
            ids = [doc_of[r] for r in ch.refs if r in doc_of]
            for d in ids:
                if d in out_stat:
                    out_stat[d] += w / (rk + rank)
            if rank == 1 and ids and not any(d in fused_stat for d in ids):
                best = max(ids, key=lambda d: specificity(text, self._title(d)))
                out_stat[best] = max(out_stat.values(), default=0.0) * 0.5 + w / (rk + rank)
        out.trace.append(("toc", "chapters: " + "; ".join(
            f"{toc.chapters[ci].code.upper()} {toc.chapters[ci].chapter.title()} ({toc.chapters[ci].name})"
            for ci, _ in chosen)))
        return out_stat

    def _ref(self, doc_id: str) -> str:
        from kanoon_bridge import present

        return present.statute_ref(self.engine, doc_id) or doc_id

    def _citation_votes(self, fused_prec: dict[str, float], aq, rt, top_n: int = 10) -> dict[str, float]:
        """Statutes inferred from what the top precedents cite: sum of 1/rank per citation, each
        cited section read in the code in force (IPC 302 -> BNS 103 for a 2025 incident), kept
        only if it belongs to the routed code family and at least agent.min_voters precedents cite it."""
        from kanoon_bridge.agent.router import FAMILY_OF_CODE

        metas = getattr(self.engine.facets, "metas", {})
        bridge = getattr(self.engine, "bridge", None)
        terms = getattr(bridge, "statute_terms", {}) or {}
        if not terms:
            return {}
        by_ref = {}
        for doc_id, ts in terms.items():
            for t in ts:
                if t.startswith("sec:"):
                    by_ref.setdefault(t[4:], doc_id)
        norm = self.engine.analyzer.text_res.normalizer
        new_codes = {"bns", "bnss", "bsa"}
        in_force = aq.code_in_force.value if aq.code_in_force.value in ("ipc", "bns") else None
        votes: dict[str, float] = {}
        voters: dict[str, set[str]] = {}
        for r, d in enumerate(sorted(fused_prec, key=fused_prec.get, reverse=True)[:top_n], start=1):
            m = metas.get(d)
            for ref in (m.statutes_cited if m else []):
                code = ref.split(":")[0]
                if FAMILY_OF_CODE.get(code) != rt.family:
                    continue
                refs = [ref]
                if in_force and (code in new_codes) != (in_force == "bns"):
                    refs = norm.equivalents(ref) or []
                for x in refs:
                    doc = by_ref.get(x) or by_ref.get(x.split("(")[0])
                    if doc:
                        votes[doc] = votes.get(doc, 0.0) + 1.0 / r
                        voters.setdefault(doc, set()).add(d)
        # one judgment's citation is an anecdote; keep statutes at least `min_voters` precedents apply
        need = self.cfg.agent.get("min_voters", 2)
        return {doc: v for doc, v in votes.items() if len(voters[doc]) >= need}

    def _verifier(self):
        from kanoon_bridge.agent.verify import StatuteVerifier

        return StatuteVerifier.from_config(self.cfg)

    def _title(self, doc_id: str) -> str:
        doc = self.docs.get(doc_id) if self.docs is not None else None
        return (doc.title if doc is not None else "") or ""

    def _statute_text(self, doc_id: str) -> str:
        doc = self.docs.get(doc_id) if self.docs is not None else None
        return " ".join(p.text for p in doc.paragraphs if p.para_id >= 0) if doc is not None else ""

    def _memo(self, query: Query, aq, rt, statutes):
        from kanoon_bridge import present
        from kanoon_bridge.agent.memo import LawEntry, ResearchMemo
        from kanoon_bridge.drift.punishment import extract

        memo = ResearchMemo(question=query.text, route=rt.describe(), code_in_force=aq.code_in_force.value
                            if aq.code_in_force.value in ("ipc", "bns") else "")
        in_force = memo.code_in_force
        new_codes = {"bns", "bnss", "bsa"}
        for h in statutes[:2]:
            ref = present.statute_ref(self.engine, h.doc_id) or h.doc_id
            code = ref.split(":")[0]
            ok = not in_force or (code in new_codes) == (in_force == "bns")
            memo.law.append(LawEntry(ref, self._title(h.doc_id), ok, present.drift_note(ref, self.engine),
                                     extract(self._statute_text(h.doc_id)).describe()))
        return memo

    def _issues(self, query: Query, aq, out: AgentResult, memo, fused_prec: dict[str, float], options):
        from kanoon_bridge.agent.issues import decompose, find_evidence
        from kanoon_bridge.agent.plan import SubQuery, _variant

        acfg = self.cfg.agent
        analyze, whole = self._analyze(), self._whole()
        if analyze is None or whole is None or self.docs is None:
            out.trace.append(("issues", "skipped: no document store"))
            return fused_prec
        idf = whole.idf
        memo.issues = decompose(self._statute_text(out.statutes[0].doc_id), analyze, idf,
                                max_issues=acfg.get("max_issues", 5))
        if not memo.issues:
            out.trace.append(("issues", "the top statute has no ingredient clauses to check"))
            return fused_prec
        top = lambda: sorted(fused_prec, key=fused_prec.get, reverse=True)[:20]  # noqa: E731
        memo.coverage = find_evidence(memo.issues, top(), self.docs, analyze)
        out.trace.append(("issues", f"{len(memo.issues)} ingredient issues, coverage {memo.coverage:.0%}"))
        for rnd in range(acfg.get("issue_rounds", 1)):
            open_issues = memo.open_issues
            if memo.coverage >= acfg.get("min_coverage", 0.8) or not open_issues:
                break
            plan = Plan(prefix=f"i{rnd + 1}q")
            for issue in open_issues:
                plan.subqueries.append(SubQuery(f"i{rnd + 1}q{len(plan.subqueries)}", "issue",
                                                _variant(query, query.text + " " + " ".join(issue.terms)),
                                                weight=acfg.get("issue_weight", 0.6),
                                                rationale=f"evidence for: {issue.label}"))
            for sq in plan.subqueries:
                out.trace.append((f"issue round {rnd + 1} {sq.sq_id}", f"[issue] {sq.query.text}  - {sq.rationale}"))
            out.plans.append(plan)
            out.subresults += run_plan(self.engine, plan, options, depth=acfg.depth)
            fused_prec = fuse(out.subresults, "precedent", self.fusion, acfg.rrf_k)
            memo.coverage = find_evidence(memo.issues, top(), self.docs, analyze)
            out.trace.append((f"issue round {rnd + 1}", f"coverage now {memo.coverage:.0%}"))
        if memo.open_issues:
            memo.notes.append(f"{len(memo.open_issues)} issue(s) without evidence in the indexed judgments")
        return fused_prec


def make_agent(engine, cfg: Config | None = None, docs=None, **kw) -> ResearchAgent:
    """The agent named by configs/default.yaml agent.version (1 or 2)."""
    cfg = cfg or getattr(engine, "cfg", None) or load_config()
    cls = ResearchAgentV2 if int(cfg.agent.get("version", 2)) >= 2 else ResearchAgent
    return cls.load(engine, cfg=cfg, docs=docs, **kw)
