"""Shared plumbing: schema round-trip, config, text pipeline, analyzer.  [all]"""

from datetime import date

from kanoon_bridge.config import load_config
from kanoon_bridge.query.analyzer import QueryAnalyzer
from kanoon_bridge.schema import Document, Query, read_documents, write_documents
from kanoon_bridge.text.pipeline import analyze_text


def test_document_roundtrip(tmp_path, toy_docs):
    path = tmp_path / "docs.jsonl"
    write_documents(toy_docs, path)
    back = list(read_documents(path))
    assert [d.doc_id for d in back] == ["p1", "p2", "p3"]
    assert back[0].decision_date == date(2010, 5, 1)
    assert back[0].zone_text("ratio").startswith("We hold")


def test_config_attribute_access():
    cfg = load_config()
    assert cfg.bm25.k1 == 1.6
    assert cfg.authority["lambda"] == 0.2


def test_pipeline_keeps_sections_and_drops_stopwords():
    toks = analyze_text("The appellant was convicted under Section 302 IPC")
    assert "sec:ipc:302" in toks
    assert "the" not in toks and "appellant" not in toks


def test_query_from_dict_and_dates():
    q = Query.from_dict({"id": "E7-1", "text": "x", "incident_date": "2024-09-01", "notes": "ignored"})
    assert q.query_id == "E7-1" and q.incident_date == date(2024, 9, 1)


def test_analyzer_runs_end_to_end_with_stubs():
    an = QueryAnalyzer.load()
    aq = an.analyze(Query("section 302 murder state:delhi", incident_date="2023-01-01"))
    assert aq.query.state == "delhi"
    assert aq.code_in_force.value == "ipc"
    assert any(step == "tokens" for step, _ in aq.trace)
