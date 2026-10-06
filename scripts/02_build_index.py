"""Build zone indexes (statutes, precedents), facet index and statute terms.  [owner: A, wiring working]

Reads data/processed/docs.jsonl, runs every document through text.pipeline.analyze_text
(the same pipeline queries use), and saves:

    data/processed/index/statutes_zone.pkl
    data/processed/index/precedents_zone.pkl
    data/processed/index/facets.pkl
    data/processed/index/statute_terms.json     statute doc_id -> its section + offence tokens

Query cases (IL-PCSR queries) are NOT indexed: they are only used as queries.

    python scripts/02_build_index.py
"""

from __future__ import annotations

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.index import store
from kanoon_bridge.index.facets import FacetIndex
from kanoon_bridge.index.zones import ZoneIndex
from kanoon_bridge.schema import DocType, read_documents
from kanoon_bridge.text.pipeline import TextResources, analyze_text


def main() -> None:
    cfg = load_config()
    res = TextResources.load(cfg)
    docs = list(read_documents(project_path(cfg.paths.docs)))
    statutes = [d for d in docs if d.doc_type == DocType.STATUTE]
    precedents = [d for d in docs if d.doc_type == DocType.PRECEDENT]
    print(f"{len(statutes)} statutes, {len(precedents)} precedents")

    def analyze(text, doc):
        return analyze_text(text, res, lang=doc.lang, date=doc.decision_date)

    store.save(ZoneIndex.build(statutes, analyze), "statutes_zone")
    store.save(ZoneIndex.build(precedents, analyze), "precedents_zone")
    store.save(FacetIndex.build(statutes + precedents), "facets")

    statute_terms = {}
    for d in statutes:
        terms = analyze_text(f"section {d.section} {d.code.value}", res) if d.section else []
        statute_terms[d.doc_id] = [t for t in terms if t.startswith(("sec:", "off:"))] + [
            f"off:{o.removeprefix('off:')}" for o in d.offence_ids
        ]
    store.save(statute_terms, "statute_terms", "json")
    print("saved indexes to", store.index_dir())


if __name__ == "__main__":
    main()
