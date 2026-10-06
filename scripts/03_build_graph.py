"""Citation graph, authority scores and tiered index.  [owner: Gaurav — working]

Uses TRAIN qrels only (rule 3) plus precedent-to-precedent citations. Saves:
    data/processed/citation_graph.json
    data/processed/index/authority.json
    data/processed/index/tiers.pkl

    python scripts/03_build_graph.py            # real data
    python scripts/03_build_graph.py --sample   # synthetic sample (after 01 --sample and 02)
"""

from __future__ import annotations

import argparse

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.index import store
from kanoon_bridge.index.tiers import TieredIndex
from kanoon_bridge.ingest import load_ilpcsr
from kanoon_bridge.rank.authority import Authority
from kanoon_bridge.rank.citation_graph import build_graph, jurisdiction_subgraphs, save_graph, stats
from kanoon_bridge.schema import DocType, read_documents

SAMPLE_OVERRIDES = {"paths": {"ilpcsr_dir": "tests/data/ilpcsr_sample"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    cfg = load_config(overrides=SAMPLE_OVERRIDES if args.sample else None)

    docs = list(read_documents(project_path(cfg.paths.docs)))
    precedents = [d for d in docs if d.doc_type == DocType.PRECEDENT]
    train_queries = [d for d in docs if d.doc_type == DocType.QUERY_CASE and d.split == "train"]
    train_qrels = load_ilpcsr.load_qrels("train", "precedent", cfg)

    graph = build_graph(precedents, train_qrels, train_queries)
    save_graph(graph, project_path(cfg.paths.graph))
    print("graph:", stats(graph))

    facets = store.load("facets")
    states = sorted({s for m in facets.metas.values() for s in m.states if s != "*"})
    subgraphs = jurisdiction_subgraphs(graph, states)
    authority = Authority.compute(graph, facets.metas, cfg, subgraphs)
    store.save(authority.to_dict(), "authority", "json")
    print(f"authority: {len(authority.global_scores)} precedents, {len(authority.state_scores)} jurisdictions")
    print("top 5:", [(d, round(s, 3)) for d, s in authority.top(5)])

    tiers = TieredIndex.build(store.load("precedents_zone"), authority.global_scores,
                              cfg.tiers.tier1_authority_quantile, cfg.tiers.champion_list_size)
    store.save(tiers, "tiers")
    print(f"tiers: tier1={len(tiers.tier1)} tier2={len(tiers.tier2)} champion lists={len(tiers.champions)}")


if __name__ == "__main__":
    main()
