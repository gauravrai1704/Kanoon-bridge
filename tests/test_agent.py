"""Layer 2: fusion maths and the agent loop with a fake core engine.  [agent owner]"""

from types import SimpleNamespace

from conftest import todo

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


@todo
def test_combsum_runs():
    assert combsum([], "precedent") == {}
