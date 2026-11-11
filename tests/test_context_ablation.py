"""E25 context ablation (eval/context_ablation.py) on hand-made inputs: no corpus, no LLM."""

from __future__ import annotations

from types import SimpleNamespace

from kanoon_bridge.eval import context_ablation as A
from kanoon_bridge.eval.significance import mcnemar_exact
from kanoon_bridge.rag.chunker import Chunk
from kanoon_bridge.schema import AnalyzedQuery, Query, ScoredDoc, SearchResult


class _Engine:
    """statute_ref reads bridge.statute_terms; ablate reads facets.metas."""

    def __init__(self):
        self.bridge = SimpleNamespace(statute_terms={"S1": ["sec:ipc:302"], "S2": ["sec:bns:103"]})
        self.facets = SimpleNamespace(metas={"P_old": SimpleNamespace(statutes_cited=["ipc:302"]),
                                             "P_new": SimpleNamespace(statutes_cited=["bns:103"]),
                                             "P_none": SimpleNamespace(statutes_cited=[])})


def test_wrong_code_follows_incident_date():
    assert A.wrong_code({"set": "e7_temporal", "expected_code": "bns", "incident_date": "2025-01-01"}) == "ipc"
    assert A.wrong_code({"set": "e7_temporal", "expected_code": "ipc", "incident_date": "2023-01-01"}) == "bns"
    assert A.wrong_code({"set": "e9_procedure_temporal", "expected_code": "bnss", "incident_date": "2025-01-01"}) == "crpc"
    assert A.wrong_code({"set": "e2_collision", "expected": "bns:84", "incident_date": "2025-01-01"}) == "ipc"


def test_ablate_drops_wrong_code_statutes_and_judgments_only():
    eng = _Engine()
    res = SearchResult(AnalyzedQuery(Query("murder")),
                       statutes=[ScoredDoc("S1", 2.0), ScoredDoc("S2", 1.0)],
                       precedents=[ScoredDoc("P_old", 3.0), ScoredDoc("P_new", 2.0), ScoredDoc("P_none", 1.0)])
    out = A.ablate(eng, res, "ipc")
    assert [h.doc_id for h in out.statutes] == ["S2"]
    assert [h.doc_id for h in out.precedents] == ["P_new", "P_none"]
    assert [h.doc_id for h in res.statutes] == ["S1", "S2"]          # the original is untouched


def test_diagnose_finds_wrong_code_text_and_the_copied_section():
    eng = _Engine()
    chunks = [Chunk(1, "S2", "statute", "Whoever commits murder shall be punished.", 1, 1.0, "BNS Section 103"),
              Chunk(2, "P_old", "ratio", "The conviction under Section 302 IPC is upheld.", 1, 1.0, "State v X")]
    row = {"set": "e7_temporal", "expected_code": "bns", "incident_date": "2025-01-01"}
    d = A.diagnose(eng, chunks, "[1] BNS Section 103 (statute). Also IPC Section 302 applies.", row, "ipc")
    assert d["wrong_code_chunks"] == 1
    assert d["first_section_in_context"] == 1.0                      # bns:103 is chunk 1
    assert (d["headers"], d["headers_matching"]) == (1, 1)
    d2 = A.diagnose(eng, chunks[:1], "IPC Section 302 applies.", row, "ipc")
    assert d2["wrong_code_chunks"] == 0 and d2["first_section_in_context"] == 0.0


def test_mislabel_is_the_right_number_under_the_old_name():
    eng = _Engine()
    norm = SimpleNamespace(offences_for=lambda ref: {"off:murder"} if ref == "bns:103" else set())
    row = {"set": "e7_temporal", "expected_code": "bns", "offence": "off:murder", "incident_date": "2025-01-01"}
    assert A.diagnose(eng, [], "Section 103 of the Indian Penal Code applies.", row, "ipc", norm)["mislabel"] == 1.0
    assert A.diagnose(eng, [], "Section 302 of the Indian Penal Code applies.", row, "ipc", norm)["mislabel"] == 0.0
    e2 = {"set": "e2_collision", "expected": "bns:103", "incident_date": "2025-01-01"}
    assert A.diagnose(eng, [], "Section 103 IPC applies.", e2, "ipc", norm)["mislabel"] == 0.0


def test_ablate_with_context_removes_unlisted_citations():
    eng = _Engine()
    eng.facets.metas["P_hidden"] = SimpleNamespace(statutes_cited=[])
    texts = {"P_hidden": "Convicted under Section 302 IPC.", "P_none": "Bail is the rule."}
    rag = SimpleNamespace()
    import kanoon_bridge.eval.context_ablation as mod

    orig = mod.chunks_of
    mod.chunks_of = lambda _rag, res: [Chunk(i + 1, h.doc_id, "ratio", texts[h.doc_id], 1, 1.0, h.doc_id)
                                       for i, h in enumerate(res.precedents)]
    try:
        res = SearchResult(AnalyzedQuery(Query("murder")), statutes=[],
                           precedents=[ScoredDoc("P_hidden", 2.0), ScoredDoc("P_none", 1.0)])
        assert [h.doc_id for h in A.ablate(eng, res, "ipc", rag).precedents] == ["P_none"]
    finally:
        mod.chunks_of = orig


def test_classify_and_summary():
    assert A.classify(1.0, 0.0, 0) == "context-induced"
    assert A.classify(1.0, 1.0, 0) == "parametric"
    assert A.classify(1.0, 1.0, 2).startswith("unresolved")
    assert A.classify(0.0, 1.0, 0) == "not misgrounded"
    assert A.classify(1.0, 1.0, 0, mislabel=1.0) == "mislabel"
    base = {"set": "e7_temporal", "condition": "naive RAG", "e20_right@1": 0.0, "n_chunks": 4, "headers": 0,
            "headers_matching": 0, "ablated_wrong_code_chunks": 0, "control_right@1": 0.0, "ablated_right@1": 1.0,
            "ablated_no_section": 0.0, "mislabel": 0.0}
    rows = [{**base, "id": "a", "e20_misgrounded@1": 1.0, "wrong_code_chunks": 2, "first_section_in_context": 1.0,
             "control_misgrounded@1": 1.0, "ablated_misgrounded@1": 0.0, "cause": "context-induced"},
            {**base, "id": "b", "e20_misgrounded@1": 1.0, "wrong_code_chunks": 0, "first_section_in_context": 0.0,
             "control_misgrounded@1": 1.0, "ablated_misgrounded@1": 1.0, "cause": "parametric"}]
    t = next(t for t in A.summarize(rows) if t["set"] == "all")
    assert t["context-induced"] == 0.5 and t["parametric"] == 0.5
    assert t["misgrounded_no_wrong_code_in_context"] == 0.5
    sig = A.significance(rows)
    assert {s["metric"] for s in sig} == {"misgrounded@1", "right@1"}


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(10, 0) - 2 / 2 ** 10) < 1e-12
    assert mcnemar_exact(3, 3) == 1.0
