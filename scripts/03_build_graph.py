"""Citation graph, authority scores and tiered index.  [owner: D, wiring working]

Uses TRAIN qrels only (rule 3). Saves:
    data/processed/citation_graph.json
    data/processed/index/authority.json
    data/processed/index/tiers.pkl

    python scripts/03_build_graph.py
"""

from __future__ import annotations

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.index import store
from kanoon_bridge.index.tiers import TieredIndex
from kanoon_bridge.ingest import load_ilpcsr
from kanoon_bridge.rank.authority import Authority
from kanoon_bridge.rank.citation_graph import build_graph, jurisdiction_subgraphs, save_graph
from kanoon_bridge.schema import DocType, read_documents


def main() -> None:
    cfg = load_config()
    docs = list(read_documents(project_path(cfg.paths.docs)))
    precedents = [d for d in docs if d.doc_type == DocType.PRECEDENT]
    train_queries = [d for d in docs if d.doc_type == DocType.QUERY_CASE and d.split == "train"]
    train_qrels = load_ilpcsr.load_qrels("train", "precedent")

    graph = build_graph(precedents, train_qrels, train_queries)
    save_graph(graph, project_path(cfg.paths.graph))
    print(f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    facets = store.load("facets")
    states = {s for m in facets.metas.values() for s in m.states if s != "*"}
    subgraphs = jurisdiction_subgraphs(graph, states)
    authority = Authority.compute(graph, facets.metas, cfg, subgraphs)
    store.save(authority.to_dict(), "authority", "json")

    tiers = TieredIndex.build(store.load("precedents_zone"), authority.global_scores,
                              cfg.tiers.tier1_authority_quantile, cfg.tiers.champion_list_size)
    store.save(tiers, "tiers")
    print("saved authority + tiers")


if __name__ == "__main__":
    main()
