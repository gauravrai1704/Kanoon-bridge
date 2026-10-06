"""Raw data -> data/processed/docs.jsonl.  [owner: A, wiring working]

Steps:
    IL-PCSR queries (all splits), precedents, statutes  ingest/load_ilpcsr.py
    BNS sections                                        ingest/load_statutes.py
    zones                                               ingest/segment.py
    court, states, date, code                           ingest/metadata.py

    python scripts/01_build_corpus.py
"""

from __future__ import annotations

from collections import Counter

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.ingest import load_ilpcsr, load_statutes, metadata, segment
from kanoon_bridge.schema import DocType, write_documents


def main() -> None:
    cfg = load_config()
    table = metadata.CourtTable.load(cfg)

    docs = []
    for split in ("train", "val", "test"):
        docs += load_ilpcsr.load_queries(split)
    docs += load_ilpcsr.load_precedents()
    docs += load_ilpcsr.load_statute_candidates()
    bns_dir = project_path(cfg.paths.raw) / "bns"
    for path in sorted(bns_dir.glob("*")):
        docs += load_statutes.parse_bns(path)

    for doc in docs:
        if doc.doc_type == DocType.STATUTE:
            segment.segment_statute(doc)
        else:
            metadata.enrich(doc, table)

    out = project_path(cfg.paths.docs)
    n = write_documents(docs, out)
    print(f"wrote {n} documents to {out}")
    print("by type:", Counter(d.doc_type.value for d in docs))
    missing_court = sum(1 for d in docs if d.doc_type != DocType.STATUTE and not d.court_name)
    missing_date = sum(1 for d in docs if d.doc_type != DocType.STATUTE and d.decision_date is None)
    print(f"judgments missing court: {missing_court}, missing date: {missing_date}")


if __name__ == "__main__":
    main()
