"""Citation graph, PageRank authority, statute bridge, tiers, QPP.  [Gaurav]"""

from datetime import date

import networkx as nx
import pytest

from kanoon_bridge.config import load_config
from kanoon_bridge.index.facets import DocMeta, FacetIndex
from kanoon_bridge.rank import qpp
from kanoon_bridge.rank.authority import Authority, binding_status, pagerank
from kanoon_bridge.rank.citation_graph import build_graph, jurisdiction_subgraphs
from kanoon_bridge.rank.statute_bridge import StatuteBridge
from kanoon_bridge.rank.topk import net_score, top_k
from kanoon_bridge.schema import DocType, ScoredDoc
from kanoon_bridge.text.version_norm import VersionNormalizer


def meta(court, states):
    return DocMeta("precedent", court, "", states, date(2015, 1, 1), "ipc", [])


# --------------------------------------------------------------------------- binding rule


def test_binding_rules():
    assert binding_status(meta("supreme_court", ["*"]), "kerala") == "binding"
    assert binding_status(meta("high_court", ["delhi"]), "delhi") == "binding"
    assert binding_status(meta("high_court", ["delhi"]), "maharashtra") == "persuasive"
    assert binding_status(meta("high_court", ["delhi"]), None) == "unknown"


def test_top_k_and_net_score():
    assert top_k({"a": 1.0, "b": 3.0, "c": 2.0}, 2) == [("b", 3.0), ("c", 2.0)]
    assert net_score(1.0, 0.5, 0.2) == 1.1


# --------------------------------------------------------------------------- graph + PageRank


def test_pagerank_matches_networkx_python_impl():
    g = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a"), ("d", "c")])
    ours = pagerank(g, 0.85)
    ref = nx.pagerank(g, alpha=0.85) if _has_scipy() else None
    assert abs(sum(ours.values()) - 1.0) < 1e-9
    assert max(ours, key=ours.get) == "c"
    if ref:
        assert all(abs(ours[n] - ref[n]) < 1e-5 for n in g)      # both stop at tol 1e-6 (summed)


def _has_scipy():
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


def test_graph_uses_train_qrels_and_blocks_test(toy_docs):
    toy_docs[1].precedents_cited = ["p1"]
    g = build_graph(toy_docs, {"q1": {"p1": 1, "p3": 1}})
    assert g.has_edge("p2", "p1") and g.has_edge("q1", "p1") and g.nodes["q1"]["kind"] == "query"
    from kanoon_bridge.schema import Document

    leaked = Document("qt", DocType.QUERY_CASE, split="test")
    with pytest.raises(ValueError):
        build_graph(toy_docs, {}, [leaked])


def test_jurisdiction_subgraph_keeps_sc_and_own_hc(toy_docs):
    g = build_graph(toy_docs, {})
    sub = jurisdiction_subgraphs(g, ["delhi"])["delhi"]
    assert set(sub.nodes) == {"p1", "p2"}          # SC + Delhi HC, not the Bombay case


def test_state_changes_ranking():
    metas = {"delhi_hc": meta("high_court", ["delhi"]), "bom_hc": meta("high_court", ["maharashtra"])}
    a = Authority(cfg=load_config(), global_scores={"delhi_hc": 0.5, "bom_hc": 0.5}, metas=metas)
    assert a.score("delhi_hc", "delhi") > a.score("bom_hc", "delhi")
    assert a.score("bom_hc", "maharashtra") > a.score("delhi_hc", "maharashtra")
    assert a.score("delhi_hc", None) == a.score("bom_hc", None) == 0.5


def test_authority_compute_most_cited_wins(toy_docs):
    g = build_graph(toy_docs, {"q1": {"p1": 1}, "q2": {"p1": 1, "p2": 1}})
    metas = {d.doc_id: DocMeta("precedent", d.court.value, "", d.states, d.decision_date, "ipc", []) for d in toy_docs}
    a = Authority.compute(g, metas, load_config(), jurisdiction_subgraphs(g, ["delhi", "maharashtra"]))
    assert a.top(1)[0][0] == "p1" and a.global_scores["p1"] == 1.0
    assert "q1" not in a.global_scores                  # only precedents get scores


# --------------------------------------------------------------------------- statute bridge


def test_bridge_reaches_ipc_citing_precedents_from_a_bns_statute():
    facets = FacetIndex(by_section={"ipc:302": {"p_old_murder"}, "ipc:506": {"p_threat"}})
    norm = VersionNormalizer.load()
    off = norm.offences_for("bns:103")[0]
    bridge = StatuteBridge(facets=facets, statute_terms={"bns:103": ["sec:bns:103", off]}, normalizer=norm, boost=0.3)
    out = bridge.run([ScoredDoc("bns:103", 9.0, DocType.STATUTE)])
    assert out.boosts == {"p_old_murder": 0.3}
    assert out.expansion["sec:bns:103"] == 0.5 and off in out.expansion


def test_bridge_weights_by_statute_score():
    facets = FacetIndex(by_section={"ipc:302": {"a"}, "ipc:506": {"b"}})
    bridge = StatuteBridge(facets=facets, statute_terms={"s1": ["sec:ipc:302"], "s2": ["sec:ipc:506"]}, boost=0.3)
    out = bridge.run([ScoredDoc("s1", 10.0), ScoredDoc("s2", 5.0)])
    assert out.boosts["a"] == pytest.approx(0.3) and out.boosts["b"] == pytest.approx(0.15)


# --------------------------------------------------------------------------- tiers + QPP


def test_tiers_and_champions():
    from kanoon_bridge.index.tiers import TieredIndex
    from kanoon_bridge.index.zones import ZoneIndex

    z = ZoneIndex()
    for d, toks in {"a": ["bail", "bail"], "b": ["bail"], "c": ["murder"]}.items():
        for i, t in enumerate(toks):
            z.whole.postings[t].setdefault(d, []).append(i)
        z.whole.doc_len[d] = len(toks)
    t = TieredIndex.build(z, {"a": 0.9, "b": 0.1, "c": 0.0}, quantile=0.6, champion_size=1)
    assert t.tier1 == {"a"} and t.champions["bail"] == ["a"]
    assert t.candidates(["bail"], min_results=1, index=z.whole) == {"a"}
    assert t.candidates(["bail"], min_results=5, index=z.whole) == {"a", "b"}
    assert t.candidates(["bail", "murder"], use_champions=True) == {"a", "c"}


def test_qpp_features_and_alpha():
    f = qpp.pre_retrieval(["rare", "common"], {"rare": 2.5, "common": 0.1}, {"rare": 3, "common": 900}, 1000)
    f = qpp.post_retrieval(f, [10.0, 5.0, 4.0, 3.0])
    assert f.max_idf == 2.5 and f.score_gap == pytest.approx(0.5)
    assert 0.4 <= qpp.alpha_from_qpp(f) <= 0.9 and not qpp.should_abstain(f)
    flat = qpp.post_retrieval(qpp.QPPFeatures(max_idf=0.1), [1.0, 1.0, 1.0])
    assert qpp.should_abstain(flat)
