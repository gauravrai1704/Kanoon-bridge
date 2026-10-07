"""All three layers wired together on a tiny in-memory corpus.

Builds the real pipeline (text pipeline with the committed crosswalk, zone + facet indexes,
statute bridge, citation graph, PageRank authority, tiers) with SearchEngine.from_components,
then checks that layer 1 (search), layer 2 (agent), layer 3 (RAG) and the evaluator all accept
each other's outputs.
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


def test_layer1_boolean_query_syntax(engine):
    res = engine.search(Query('knife AND market', **Q))
    assert [h.doc_id for h in res.precedents] == ["P3"]
    res = engine.search(Query('"knife injuries" OR jewellery', **Q))
    assert {h.doc_id for h in res.precedents} == {"P1", "P4"}
    res = engine.search(Query("BNS 103 AND NOT Delhi", **Q))       # section = offence in either code
    assert {h.doc_id for h in res.precedents} == {"P1", "P3"}
    assert any(step == "boolean" for step, _ in res.query.trace)


def test_layer1_hindi_and_hinglish_reach_the_same_statute(engine):
    hi = engine.search(Query("चाकू से हत्या", **Q))
    hinglish = engine.search(Query("chaku se hatya kar di", **Q))
    assert hi.query.detected_lang == "hi" and hinglish.query.detected_lang == "hinglish"
    assert hi.statutes[0].doc_id == hinglish.statutes[0].doc_id == "bns:103"
    assert "murder" in hi.query.expanded_terms


def test_layer1_dense_fusion_with_fake_encoder(engine, tmp_path):
    import copy

    import numpy as np

    from kanoon_bridge.rank.dense import DenseRetriever
    from kanoon_bridge.search import SearchOptions

    class Fake:
        def encode(self, texts, **kw):
            return np.array([[t.count("knife") + 0.1, t.count("theft") + 0.1] for t in texts], dtype=np.float32)

    dense = DenseRetriever(cfg=load_config(overrides={"paths": {"embeddings": str(tmp_path)}}), model=Fake())
    dense.encode_corpus([d for d in DOCS if d.doc_type == DocType.PRECEDENT])
    eng = copy.copy(engine)
    eng.dense = dense
    res = eng.search(Query("knife", **Q), SearchOptions(dense=True, qpp=True))
    assert all("dense" in h.components for h in res.precedents)
    fusion = [v for step, v in res.query.trace if step == "fusion"]           # typed question: 3-way QPP gate
    assert fusion and "dense" in fusion[0] and "QPP confidence" in fusion[0]
    long_q = " ".join(["the accused stabbed with a knife in the murder"] * 8)  # a pasted judgment
    res = eng.search(Query(long_q, **Q), SearchOptions(dense=True, qpp=True, ngram=True))
    assert any(step == "alpha" for step, _ in res.query.trace)


# ---------------------------------------------------------------- tolerant retrieval + presentation
def test_spelling_correction_and_did_you_mean(engine):
    res = engine.search(Query("murdr with a knfe", **Q))
    fixed = {c.word: c.display for c in res.query.corrections}
    assert fixed == {"murdr": "murder", "knfe": "knife"}
    assert res.query.suggestion == "murder with a knife"
    assert res.precedents and res.precedents[0].doc_id in {"P1", "P2", "P3"}


def test_wildcards_in_free_text_and_boolean(engine):
    res = engine.search(Query("stab* knife", **Q))
    assert "stab" in res.query.wildcards["stab*"]
    res = engine.search(Query("jewel* AND theft", **Q))
    assert [h.doc_id for h in res.precedents] == ["P4"]


def test_snippet_highlight_and_reasons(engine, docs):
    from kanoon_bridge import present

    res = engine.search(Query("BNS 103 knife", **Q))
    top = res.precedents[0]
    snip = present.snippet(docs[top.doc_id], res.query, engine.analyzer.text_res)
    assert "**knife" in snip or "**Section 302 IPC**" in snip
    why = present.explain(top, res.query, engine)
    assert any("IPC 302 = BNS 103" in r for r in why)
    assert any(r.startswith("binding") for r in why)
    assert present.version_note("bns:103", engine.analyzer.text_res.normalizer).startswith("BNS 103 <- IPC 302")
    counts = present.facet_counts(res.precedents, engine)
    assert sum(counts["court"].values()) == len(res.precedents)


def test_similar_cases_version_aware_coupling(engine, docs):
    from kanoon_bridge.rank.similar import SimilarCases

    sim = SimilarCases(engine=engine, docs=docs)
    found = sim.find("P2", k=3)
    assert found and found[0][0] in {"P1", "P3"}
    assert found[0][2]["coupling"] == 1.0                      # both cite the murder offence
    assert "P2" not in [d for d, _, _ in found]


def test_relevance_feedback_and_date_filters(engine, docs):
    from kanoon_bridge.rank.feedback import rocchio_options
    from kanoon_bridge.search import SearchOptions

    first = engine.search(Query("knife", **Q))
    fb = rocchio_options(engine, first.query, docs, ["P3"], ["P1"])
    assert fb["extra_terms"] and all(w == 0.75 for w in fb["extra_terms"].values())
    res = engine.search(Query("knife", **Q), SearchOptions(**fb))
    assert any(step == "feedback" for step, _ in res.query.trace)
    dated = engine.search(Query("knife after:2016 before:2020", **Q))
    assert {h.doc_id for h in dated.precedents} <= {"P2", "P3"}


def test_ltr_rerank_and_duplicate_collapse(engine):
    import copy

    import numpy as np

    from kanoon_bridge.rank.ltr import FEATURES, LinearRanker
    from kanoon_bridge.search import SearchOptions

    eng = copy.copy(engine)
    w = np.zeros(len(FEATURES))
    w[FEATURES.index("binding")] = 1.0                            # a ranker that only likes binding cases
    eng.ltr = LinearRanker(weights=w)
    case_query = " ".join(["the accused stabbed with a knife in the murder"] * 8)   # LTR only re-ranks case-as-query input
    res = eng.search(Query(case_query, **Q), SearchOptions(ltr=True))
    assert "ltr" in res.precedents[0].components and res.precedents[0].components["ltr"] == 1.0   # a binding case on top
    short = eng.search(Query("knife murder", **Q), SearchOptions(ltr=True))
    assert all("ltr" not in h.components for h in short.precedents)
    # the short-query model: only when asked for, and never for a query citing a section
    eng.ltr_short = LinearRanker(weights=w)
    typed = eng.search(Query("knife murder", **Q), SearchOptions(ltr=True, ltr_short=True))
    assert any(step == "ltr" and "short-query" in v for step, v in typed.query.trace)
    cited = eng.search(Query("cases under section 302 IPC", **Q), SearchOptions(ltr=True, ltr_short=True))
    assert not any(step == "ltr" for step, _ in cited.query.trace)
    eng.near_dups = {"P1": "P1", "P2": "P1"}
    res = eng.search(Query("knife murder", **Q), SearchOptions(collapse_duplicates=True))
    ids = [h.doc_id for h in res.precedents]
    assert not ({"P1", "P2"} <= set(ids))


def test_web_backend_payload(engine, docs):
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("web_server", Path(__file__).parents[1] / "app" / "web_server.py")
    web = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(web)
    b = web.Backend(engine=engine, docs=docs)
    out = b.search({"q": "BNS 103 knfe", "state": "delhi", "date": "2025-01-03", "agent": True, "answer": True,
                    "generator": "extractive"})
    json.dumps(out)                                              # everything is JSON-serialisable
    u = out["understanding"]
    assert u["crossings"][0] == {"from": "BNS 103", "to": "IPC 302", "offence": u["crossings"][0]["offence"]}
    assert u["did_you_mean"] == "BNS 103 knife"
    assert out["statutes"][0]["ref"] == "BNS 103" and out["statutes"][0]["in_force"]
    assert out["precedents"] and "<mark>" in "".join(p["snippet"] for p in out["precedents"])
    assert out["agent"]["searches"] >= 5 and out["answer"] is not None
    groups = [s["group"] for s in out["pipeline"]]
    assert groups[0] == "Understand" and "Research agent" in groups and groups[-1] == "Answer"
    assert b.search({"q": ""})["error"]
