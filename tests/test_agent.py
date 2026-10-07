"""Layer 2: fusion maths, planner rules, reflection and the agent loop with a fake core engine."""

from types import SimpleNamespace

from kanoon_bridge.agent.fuse import combsum, reciprocal_rank_fusion
from kanoon_bridge.agent.plan import RulePlanner
from kanoon_bridge.agent.research import ResearchAgent
from kanoon_bridge.config import load_config
from kanoon_bridge.schema import AnalyzedQuery, DocType, Query, ScoredDoc, SearchResult


def test_rrf_formula():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert abs(fused["b"] - (1 / 62 + 1 / 61)) < 1e-12
    assert max(fused, key=fused.get) == "b"


def test_rrf_weights():
    fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[2.0, 1.0], k=60)
    assert fused["a"] > fused["b"]


class FakeEngine:
    """Returns a fixed ranking; stands in for search.SearchEngine."""

    def __init__(self):
        self.calls = 0
        self.analyzer = SimpleNamespace(analyze=lambda q: AnalyzedQuery(query=q, tokens=q.text.split()))

    def search(self, query, options):
        self.calls += 1
        hits = [ScoredDoc("p1", 2.0, DocType.PRECEDENT), ScoredDoc("p2", 1.0, DocType.PRECEDENT)]
        return SearchResult(query=AnalyzedQuery(query=query), precedents=hits,
                            statutes=[ScoredDoc("bns:103", 1.0, DocType.STATUTE)])


def test_agent_with_only_original_matches_core():
    engine = FakeEngine()
    out = ResearchAgent(engine=engine, cfg=load_config()).run(Query("murder knife"))
    assert [h.doc_id for h in out.precedents] == ["p1", "p2"]     # same order as the core
    assert out.statutes[0].doc_id == "bns:103"
    assert out.n_searches == engine.calls >= 1
    assert any("reflect" in step for step, _ in out.trace)


def test_planner_adds_facet_subquery_when_state_given():
    import pytest

    q = Query("bail dowry death", state="delhi")
    plan = RulePlanner().plan(q, AnalyzedQuery(query=q, tokens=["bail", "dowri", "death"]))
    if any(n.startswith("skipped facet") for n in plan.notes):
        pytest.skip("not implemented yet: TODO(agent): facet sub-query")
    assert "facet" in [sq.kind for sq in plan.subqueries]


def test_combsum_runs():
    assert combsum([], "precedent") == {}


def _sr(kind, target, ids, scores, weight=1.0):
    from kanoon_bridge.agent.executor import SubResult
    from kanoon_bridge.agent.plan import SubQuery

    hits = [ScoredDoc(d, s, DocType.PRECEDENT) for d, s in zip(ids, scores)]
    res = SearchResult(query=AnalyzedQuery(query=Query("x")), precedents=hits if target == "precedent" else [],
                       statutes=hits if target == "statute" else [])
    return SubResult(SubQuery("q", kind, Query("x"), target=target, weight=weight), res)


def test_combsum_minmax_and_targets():
    srs = [_sr("original", "precedent", ["a", "b", "c"], [10, 5, 0]),
           _sr("facet", "precedent", ["b", "c"], [2, 1]),
           _sr("statutes", "statute", ["s1"], [3])]
    fused = combsum(srs, "precedent")
    assert fused == {"a": 1.0, "b": 1.5, "c": 0.0}
    assert combsum(srs, "statute") == {"s1": 1.0}    # the statutes sub-query feeds only the statute list


def _aq(q, **kw):
    return AnalyzedQuery(query=q, **kw)


def test_planner_rules():
    from types import SimpleNamespace

    q = Query("BNS 103 knife stabbing", state="delhi", incident_date="2025-01-03")
    aq = _aq(q, tokens=["sec:bns:103", "knife", "stab", "off:murder_bns103"], sections=["bns:103"],
             offence_ids=["off:murder_bns103"], expanded_terms={"sec:ipc:302": 0.8})
    idf = {"knife": 1.2, "stab": 0.9, "off:murder_bns103": 0.5}
    planner = RulePlanner(normalizer=SimpleNamespace(label=lambda o: "Punishment for murder"), df=lambda t: 5)
    plan = planner.plan(q, aq, idf)
    by = {sq.kind: sq for sq in plan.subqueries}
    assert list(by) == ["original", "cross_code", "boolean", "facet", "statutes"]
    assert by["cross_code"].query.text == "knife stabbing IPC Section 302"
    assert by["boolean"].options["require_terms"][0][:2] == ["off:murder_bns103", "sec:bns:103"]
    assert ["knife"] in by["boolean"].options["require_terms"]
    assert by["facet"].options == {"restrict_states": ["delhi"]} and by["facet"].weight == 0.8
    assert by["statutes"].target == "statute" and "Punishment for murder" in by["statutes"].query.text
    assert plan.subqueries[0].query is q                       # original untouched


def test_boolean_skips_rare_terms_and_needs_two_clauses():
    q = Query("zygote")
    aq = _aq(q, tokens=["zygote", "knife"])
    plan = RulePlanner(df=lambda t: {"zygote": 1, "knife": 9}[t]).plan(q, aq, {"zygote": 3.0, "knife": 1.0})
    assert "boolean" not in [sq.kind for sq in plan.subqueries]


def test_reflect_decisions():
    from kanoon_bridge.agent import reflect

    aq = _aq(Query("x"), tokens=["knife"])
    many = {f"d{i}": 1.0 for i in range(20)}
    assert reflect.should_reformulate(many, aq, {"knife": 3.0}, 2, 2)[0] is False            # round cap
    assert reflect.should_reformulate({"a": 1}, aq, {"knife": 3.0}, 1, 2)[0] is True         # too few results
    assert reflect.should_reformulate(many, aq, {"knife": 3.0}, 1, 2, [10, 2, 1])[0] is False  # confident
    assert reflect.should_reformulate(many, aq, {"knife": 0.1}, 1, 2, [1.0, 0.99])[0] is True  # flat + vague


def test_reformulate_prf_terms():
    from kanoon_bridge.agent import reflect
    from kanoon_bridge.agent.plan import Plan
    from kanoon_bridge.schema import Code, Document, Paragraph

    docs = {"d1": Document("d1", DocType.PRECEDENT, paragraphs=[Paragraph("dagger wound dagger sec:ipc:302", "ratio")]),
            "d2": Document("d2", DocType.PRECEDENT, paragraphs=[Paragraph("dagger blood", "ratio")])}
    q = Query("knife stab attack fight")
    aq = _aq(q, tokens=["knife", "stab", "attack", "fight"], code_in_force=Code.BNS)
    plan = Plan()
    plan.add("original", q, "")
    idf = {"dagger": 1.0, "wound": 0.8, "blood": 0.7, "knife": 1.0, "stab": 0.9, "attack": 0.2, "fight": 0.5,
           "sec:ipc:302": 2.0}.get
    sqs = reflect.reformulate(aq, {"d1": 2.0, "d2": 1.0}, docs, plan, top_n=2, n_terms=2,
                              idf=lambda t: idf(t, 0.0), df=lambda t: 3, analyze=str.split)
    assert len(sqs) == 1 and sqs[0].kind == "reformulated" and sqs[0].sq_id == "r2q0"
    assert list(sqs[0].options["extra_terms"]) == ["dagger", "wound"]   # IPC token excluded under BNS
    assert sqs[0].options["drop_terms"] == ["attack"]
    assert reflect.reformulate(aq, {"d1": 1.0}, None, plan) == []


def test_llm_query_parsing():
    from kanoon_bridge.agent.plan import parse_llm_queries

    out = parse_llm_queries('Sure: ["murder knife BNS 103", "IPC 302 stabbing", "murder knife BNS 103", 5, ""]')
    assert out == ["murder knife BNS 103", "IPC 302 stabbing"]
    assert parse_llm_queries("no json") == []


def test_make_planner_kinds():
    from kanoon_bridge.agent.plan import HybridPlanner, LLMPlanner, make_planner

    assert isinstance(make_planner(load_config()), RulePlanner)
    assert isinstance(make_planner(load_config(overrides={"agent": {"planner": "llm"}})), LLMPlanner)
    assert isinstance(make_planner(load_config(overrides={"agent": {"planner": "hybrid"}})), HybridPlanner)


def test_hybrid_planner_with_fake_llm(monkeypatch):
    from kanoon_bridge.agent.plan import HybridPlanner, LLMPlanner
    from kanoon_bridge.rag import generate

    monkeypatch.setattr(generate, "call_claude", lambda *a, **k: '["murder by stabbing BNS 103", "knife injury intention"]')
    q = Query("chaku se maara", incident_date="2025-01-03")
    plan = HybridPlanner(RulePlanner(), LLMPlanner()).plan(q, AnalyzedQuery(query=q, tokens=["chaku", "maara"]), {})
    llm = [sq for sq in plan.subqueries if sq.kind == "llm"]
    assert [sq.query.text for sq in llm] == ["murder by stabbing BNS 103", "knife injury intention"]
    assert all(sq.query.incident_date == q.incident_date and sq.weight == 0.7 for sq in llm)

    def boom(*a, **k):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(generate, "call_claude", boom)
    plan = LLMPlanner().plan(q, AnalyzedQuery(query=q))
    assert [sq.kind for sq in plan.subqueries] == ["original"] and "llm planner skipped" in plan.notes[0]
