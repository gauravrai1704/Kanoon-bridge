"""All three layers wired together on a tiny in-memory corpus.

Builds the real pipeline (text pipeline with the committed crosswalk, zone + facet indexes,
statute bridge, citation graph, PageRank authority, tiers) with SearchEngine.from_components,
then checks that layer 1 (search), layer 2 (agent), layer 3 (RAG) and the evaluator all accept
each other's outputs. Until the index code is merged, the temporary reference index
(src/kanoon_bridge/dev/reference_index.py) fills in for it during THIS module only.
"""

from __future__ import annotations

from datetime import date

import pytest

from kanoon_bridge.config import load_config
from kanoon_bridge.schema import Code, Court, Document, DocType, Paragraph, Query


def _statute(doc_id, code, num, title, text, offences):
    return Document(doc_id, DocType.STATUTE, title=title, code=code, section=num, offence_ids=offences,
                    decision_date=date(2024, 7, 1) if code == Code.BNS else date(1860, 10, 6),
                    meta={"ref": f"{code.value}:{num}"},
                    paragraphs=[Paragraph(f"Section {num} {code.value.upper()}. {title}.", "statute", -1),
                                Paragraph(text, "statute", 0)])


def _prec(doc_id, court, states, d, facts, ratio, cites, title):
    return Document(doc_id, DocType.PRECEDENT, title=title, court=court, states=states, decision_date=d,
                    code=Code.IPC if d < date(2024, 7, 1) else Code.BNS, statutes_cited=cites,
                    paragraphs=[Paragraph(facts, "facts", 0), Paragraph(ratio, "ratio", 1)])


DOCS = [
    _statute("bns:103", Code.BNS, "103", "Punishment for murder",
             "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.",
             ["off:murder_bns103"]),
    _statute("S302", Code.IPC, "302", "Punishment for murder",
             "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
             ["off:murder_bns103"]),
    _statute("bns:303", Code.BNS, "303", "Theft",
             "Whoever intending to take dishonestly any movable property out of the possession of any person commits theft.",
             []),
    _prec("P1", Court.SUPREME_COURT, ["*"], date(2015, 3, 1),
          "The accused stabbed the deceased with a knife after a quarrel over land.",
          "We hold that the knife injuries show an intention to cause death; the conviction for murder under Section 302 IPC is upheld.",
          ["ipc:302"], "State v. Arjun"),
    _prec("P2", Court.HIGH_COURT, ["delhi"], date(2019, 8, 9),
          "The appellant stabbed his neighbour with a knife in Delhi.",
          "Murder under Section 302 IPC is made out; the knife wound was sufficient to cause death.",
          ["ipc:302"], "Ravi v. State (NCT of Delhi)"),
    _prec("P3", Court.HIGH_COURT, ["maharashtra"], date(2020, 1, 15),
          "The accused attacked the victim with a knife at a market.",
          "The injuries prove murder under Section 302 IPC; appeal dismissed.",
          ["ipc:302"], "Prakash v. State of Maharashtra"),
    _prec("P4", Court.HIGH_COURT, ["delhi"], date(2025, 2, 2),
          "The accused was found with stolen jewellery.",
          "The offence of theft under Section 303 BNS is proved.",
          ["bns:303"], "State v. Mohan"),
]
for d in DOCS:                                  # citations between precedents feed PageRank
    d.precedents_cited = {"P2": ["P1"], "P3": ["P1"], "P4": []}.get(d.doc_id, [])


@pytest.fixture(scope="module")
def engine():
    from kanoon_bridge.dev import reference_index

    reference_index.install(quiet=True)
    try:
        from kanoon_bridge.index.facets import FacetIndex
        from kanoon_bridge.index.tiers import TieredIndex
        from kanoon_bridge.index.zones import ZoneIndex
        from kanoon_bridge.rank.authority import Authority
        from kanoon_bridge.rank.citation_graph import build_graph, jurisdiction_subgraphs
        from kanoon_bridge.search import SearchEngine
        from kanoon_bridge.text.pipeline import TextResources, analyze_text

        cfg = load_config()
        res = TextResources.load(cfg)

        def analyze(text, doc):
            return analyze_text(text, res, lang=doc.lang, date=doc.decision_date)

        statutes = [d for d in DOCS if d.doc_type == DocType.STATUTE]
        precs = [d for d in DOCS if d.doc_type == DocType.PRECEDENT]
        sz, pz, facets = ZoneIndex.build(statutes, analyze), ZoneIndex.build(precs, analyze), FacetIndex.build(DOCS)
        terms = {d.doc_id: [f"sec:{d.meta['ref']}"] + d.offence_ids for d in statutes}
        graph = build_graph(precs, {}, [])
        auth = Authority.compute(graph, facets.metas, cfg, jurisdiction_subgraphs(graph, ["delhi", "maharashtra"]))
        tiers = TieredIndex.build(pz, auth.global_scores, 0.5, 10)
        yield SearchEngine.from_components(cfg, sz, pz, facets, terms, auth, tiers)
    finally:
        reference_index.uninstall()


@pytest.fixture(scope="module")
def docs():
    return {d.doc_id: d for d in DOCS}


Q = dict(state="delhi", incident_date="2025-01-03")


def test_layer1_cross_code_search(engine):
    res = engine.search(Query("BNS 103 knife stabbing", **Q))
    assert res.statutes[0].doc_id == "bns:103"                       # code in force
    assert "S302" not in [h.doc_id for h in res.statutes]              # superseded code filtered
    assert {"P1", "P2", "P3"} <= {h.doc_id for h in res.precedents}    # IPC-era precedents via the bridge


def test_layer1_agent_hooks(engine):
    from kanoon_bridge.search import SearchOptions

    binding = engine.search(Query("knife murder", **Q), SearchOptions(restrict_states=["delhi"]))
    assert {h.doc_id for h in binding.precedents} <= {"P1", "P2", "P4"}
    boolean = engine.search(Query("knife murder", **Q), SearchOptions(require_terms=[["knife"], ["market"]]))
    assert [h.doc_id for h in boolean.precedents] == ["P3"]
    assert engine.boolean_candidates([["zzz"], ["knife"]]) == set()


def test_layer2_agent_plan_and_fusion(engine, docs):
    from kanoon_bridge.agent.research import ResearchAgent
    from kanoon_bridge.search import SearchOptions

    agent = ResearchAgent.load(engine, docs=docs)
    out = agent.run(Query("BNS 103 knife stabbing", **Q), SearchOptions(top_k=5))
    kinds = [sq.kind for sq in out.plans[0].subqueries]
    assert kinds[0] == "original" and {"cross_code", "boolean", "facet", "statutes"} <= set(kinds)
    cross = next(sq for sq in out.plans[0].subqueries if sq.kind == "cross_code")
    assert "IPC Section 302" in cross.query.text
    assert out.statutes[0].doc_id == "bns:103"
    assert out.precedents and out.precedents[0].doc_id in {"P1", "P2"}  # binding in Delhi ranks first
    assert out.found_by(out.precedents[0].doc_id)
    assert out.n_searches == len(out.subresults) >= 5
    assert any(step.endswith("reflect") for step, _ in out.trace)
    assert len(out.plans) == 2 and out.plans[1].subqueries[0].kind == "reformulated"   # < 10 results -> PRF round


def test_layer2_combsum_agent(engine, docs):
    from kanoon_bridge.agent.research import ResearchAgent

    out = ResearchAgent.load(engine, docs=docs, fusion="combsum").run(Query("knife murder", **Q))
    assert out.precedents and "combsum" in out.precedents[0].components


def test_layer3_on_layer1_and_layer2(engine, docs):
    from kanoon_bridge.agent.research import ResearchAgent
    from kanoon_bridge.rag.answer import RagPipeline, render

    rag = RagPipeline.load(engine, docs=docs)
    rag.cfg = load_config(overrides={"rag": {"abstain_min_idf": 0.05}})   # 4-doc corpus: every idf is tiny
    core = engine.search(Query("punishment for murder with a knife", **Q))
    a1 = rag.answer(core, generator="extractive")
    assert not a1.abstained and a1.sentences and a1.chunks[0].title.startswith("BNS Section 103")
    out = ResearchAgent.load(engine, docs=docs).run(Query("punishment for murder with a knife", **Q))
    a2 = rag.answer(out, generator="extractive")
    assert not a2.abstained and a2.sentences
    assert "Sources:" in render(a2)


def test_evaluator_runs_core_and_agent(engine, docs):
    from kanoon_bridge.agent.research import ResearchAgent
    from kanoon_bridge.eval.run_eval import TestSet, evaluate_set
    from kanoon_bridge.search import SearchOptions

    ev = load_config("eval.yaml")
    ts = TestSet("e1_toy", [Query("knife murder stab", query_id="t1", **Q)], {"t1": {"P1": 1, "P2": 1}}, "precedent")
    core = evaluate_set(engine, ts, SearchOptions(), "full", ev)
    agent = evaluate_set(engine, ts, SearchOptions(), "agent", ev, agent=ResearchAgent.load(engine, docs=docs))
    assert core["MAP"] > 0 and agent["MAP"] > 0
