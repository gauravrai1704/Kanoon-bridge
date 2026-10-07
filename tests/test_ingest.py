"""Ingest on the synthetic IL-PCSR sample (same schema as the real dataset).  [Gaurav]"""

from datetime import date
from pathlib import Path

import pytest

from kanoon_bridge.config import load_config
from kanoon_bridge.ingest import load_ilpcsr, metadata, segment
from kanoon_bridge.ingest.load_ilpcsr import parse_provision
from kanoon_bridge.schema import Code, Court, DocType

SAMPLE = Path(__file__).resolve().parent / "data" / "ilpcsr_sample"


@pytest.fixture(scope="module")
def cfg():
    if not (SAMPLE / "queries" / "train_queries.jsonl").exists():
        import runpy

        runpy.run_path(str(SAMPLE.parent / "make_ilpcsr_sample.py"), run_name="__main__")
    return load_config(overrides={"paths": {"ilpcsr_dir": str(SAMPLE)}})


def test_parse_provision_shapes():
    assert parse_provision("Section 302 in The Indian Penal Code, 1860")[:2] == ("ipc", "302")
    assert parse_provision("Indian Penal Code, 1860_Section 498A")[:2] == ("ipc", "498a")
    assert parse_provision("Section 438 in The Code Of Criminal Procedure, 1973")[:2] == ("crpc", "438")
    assert parse_provision("Article 21 in Constitution of India")[:2] == ("constitution", "21")
    assert parse_provision("Section 138 in The Negotiable Instruments Act, 1881")[:2] == ("nia", "138")
    assert parse_provision("Section 5 in The Foo Bar Act, 1999")[0] == "foo_bar_act"


def test_loader_reads_all_configs(cfg):
    assert [len(load_ilpcsr.load_queries(s, cfg)) for s in ("train", "val", "test")] == [3, 1, 2]
    precs = load_ilpcsr.load_precedents(cfg)
    stats = {s.doc_id: s for s in load_ilpcsr.load_statute_candidates(cfg)}
    assert len(precs) == 8 and stats["S302"].code == Code.IPC and stats["S302"].section == "302"
    assert stats["S438"].code == Code.CRPC and stats["A21"].meta["ref"] == "constitution:21"


def test_statute_ids_become_section_refs(cfg):
    p02 = {d.doc_id: d for d in load_ilpcsr.load_precedents(cfg)}["P02"]
    assert p02.statutes_cited == ["ipc:304b", "ipc:498a"] and p02.precedents_cited == ["P01"]
    assert p02.decision_date == date(2005, 7, 1) and p02.meta["jurisdiction"] == "Delhi High Court"


def test_qrels_by_split_and_target(cfg):
    assert load_ilpcsr.load_qrels("train", "precedent", cfg)["Q01"] == {"P01": 1, "P06": 1}
    assert load_ilpcsr.load_qrels("test", "statute", cfg)["Q05"] == {"S438": 1, "A21": 1}


def test_roles_become_zones():
    paras = segment.segment_judgment(["a", "b", "c", "d"], ["Facts", "PetArg", "CourtRes", "Conclusion"])
    assert [p.zone for p in paras] == ["facts", "arguments", "ratio", "decision"]
    unlabelled = segment.segment_judgment(["Learned counsel submitted that ...", "We hold that ..."], None)
    assert [p.zone for p in unlabelled] == ["arguments", "ratio"]


def test_metadata_from_jurisdiction(cfg):
    table = metadata.CourtTable.load(cfg)
    doc = {d.doc_id: d for d in load_ilpcsr.load_precedents(cfg)}["P03"]
    metadata.enrich(doc, table)
    assert doc.court == Court.HIGH_COURT and "maharashtra" in doc.states and doc.code == Code.IPC
    sc = {d.doc_id: d for d in load_ilpcsr.load_precedents(cfg)}["P01"]
    metadata.enrich(sc, table)
    assert sc.court == Court.SUPREME_COURT and sc.states == ["*"]


def test_find_date_formats():
    assert metadata.find_date("DATED: 04.03.2016 ... FIR of 12.01.2014") == date(2016, 3, 4)
    assert metadata.find_date("Judgment delivered on 4th March, 2016") == date(2016, 3, 4)
    assert metadata.find_date("decided on March 4, 2016") == date(2016, 3, 4)
    assert metadata.find_date("no date here") is None
    assert metadata.code_in_force(date(2024, 7, 1)) == Code.BNS


def test_bns_sections_load_when_source_present():
    from kanoon_bridge.config import project_path
    from kanoon_bridge.ingest.load_statutes import parse_bns

    root = project_path(load_config().paths.bns_dir)
    if not (root / "data" / "sections").exists():
        pytest.skip("BNS source not fetched (scripts/00_fetch_data.py --only bns)")
    docs = {d.doc_id: d for d in parse_bns(root)}
    assert len(docs) == 358 and docs["bns:103"].title == "Punishment for murder"
    assert docs["bns:103"].doc_type == DocType.STATUTE and docs["bns:103"].meta["ipc_reference"].startswith("302")
