"""Raw data -> data/processed/docs.jsonl.  [owner: Gaurav — working]

Steps:
    crosswalk CSVs regenerated from the BNS source (if present)   ingest/parse_crosswalk.py
    IL-PCSR queries (train/val/test), precedents, statutes         ingest/load_ilpcsr.py
    358 BNS sections                                               ingest/load_statutes.py
    zones (role labels -> zones)                                   ingest/segment.py
    court, states, date, code in force                             ingest/metadata.py
    statute offence ids (IPC 302 -> off:murder_bns103)              text/version_norm.py

    python scripts/01_build_corpus.py               # real data from data/raw/
    python scripts/01_build_corpus.py --sample      # tiny synthetic sample in tests/data/
"""

from __future__ import annotations

import argparse
from collections import Counter

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.ingest import load_ilpcsr, load_statutes, metadata, parse_crosswalk, segment
from kanoon_bridge.schema import Code, DocType, Paragraph, write_documents
from kanoon_bridge.text.version_norm import VersionNormalizer

SAMPLE_OVERRIDES = {"paths": {"ilpcsr_dir": "tests/data/ilpcsr_sample", "bns_dir": "data/raw/bns-study-platform"}}

_CODE_LABEL = {"ipc": "IPC", "bns": "BNS", "crpc": "CrPC", "bnss": "BNSS"}


def heading(doc) -> str:
    """Normalised first line so the tokenizer emits the right section token ('Section 302 IPC.')."""
    code = doc.meta.get("act_code", doc.code.value)
    if doc.section and code in _CODE_LABEL:
        return f"Section {doc.section} {_CODE_LABEL[code]}. {doc.title}."
    return f"{doc.title}."


def build(cfg) -> list:
    bns_dir = project_path(cfg.paths.bns_dir)
    if (bns_dir / "data" / "sections").exists():
        rows = parse_crosswalk.extract_rows_from_bns_json(bns_dir)
        parse_crosswalk.write_crosswalk(rows, project_path(cfg.paths.crosswalk))
        parse_crosswalk.build_offence_ids(rows, project_path(cfg.paths.offence_ids))
        print(f"crosswalk: {sum(1 for r in rows if r['ipc_section'])} IPC-BNS pairs")
    normalizer = VersionNormalizer.load(cfg)
    table = metadata.CourtTable.load(cfg)

    docs = list(load_ilpcsr.iter_all(cfg))
    if (bns_dir / "data" / "sections").exists():
        docs += load_statutes.parse_bns(bns_dir)
    else:
        print(f"warning: {bns_dir} missing - BNS statutes not added (run scripts/00_fetch_data.py)")

    for doc in docs:
        if doc.doc_type == DocType.STATUTE:
            if doc.meta.get("source") == "il-pcsr":
                segment.segment_statute(doc)
            doc.paragraphs.insert(0, Paragraph(heading(doc), "statute", -1))
            ref = doc.meta.get("ref", "")
            doc.offence_ids = normalizer.offences_for(ref) if doc.code in (Code.IPC, Code.BNS) else []
        else:
            metadata.enrich(doc, table)
    return docs


def report(docs) -> None:
    print("by type:", dict(Counter(d.doc_type.value for d in docs)))
    print("query splits:", dict(Counter(d.split for d in docs if d.doc_type == DocType.QUERY_CASE)))
    judg = [d for d in docs if d.doc_type != DocType.STATUTE]
    print("judgments missing court:", sum(1 for d in judg if d.court.value == "unknown"),
          "| missing states:", sum(1 for d in judg if not d.states),
          "| missing date:", sum(1 for d in judg if d.decision_date is None))
    print("courts:", Counter(d.court_name for d in judg).most_common(8))
    print("zones:", dict(Counter(p.zone for d in judg for p in d.paragraphs)))
    stat = [d for d in docs if d.doc_type == DocType.STATUTE]
    print("statutes by code:", dict(Counter(d.code.value for d in stat)),
          "| with offence ids:", sum(1 for d in stat if d.offence_ids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="use the synthetic sample (tests/data)")
    args = ap.parse_args()
    cfg = load_config(overrides=SAMPLE_OVERRIDES if args.sample else None)
    docs = build(cfg)
    out = project_path(cfg.paths.docs)
    n = write_documents(docs, out)
    print(f"wrote {n} documents to {out}")
    report(docs)


if __name__ == "__main__":
    main()
