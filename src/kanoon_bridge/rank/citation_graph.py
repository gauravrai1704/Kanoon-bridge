"""Citation graph over precedents.  [owner: Gaurav — working]

Edges u -> v mean "u cites v" (weight = number of times seen). Sources, with NO test leakage
(team rule 3):
  * TRAIN-split IL-PCSR qrels: query case -> cited precedent
  * precedent documents (unmasked): precedent -> precedent it cites (relevant_precedent_ids),
    kept only when the cited id is in the precedent pool

Node attributes: kind ('precedent' | 'query'), court, states (list), date (ISO or "").

jurisdiction_subgraphs() keeps, per state, the nodes whose rulings bind there (the Supreme Court
plus that state's High Court) and the edges among them — graph mining per jurisdiction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import networkx as nx

from kanoon_bridge.schema import Document

SUPREME = "*"


def _attrs(doc: Document, kind: str) -> dict:
    return {"kind": kind, "court": doc.court.value, "states": list(doc.states),
            "date": doc.decision_date.isoformat() if doc.decision_date else ""}


def _add_edge(g: nx.DiGraph, u: str, v: str) -> None:
    if u == v:
        return
    if g.has_edge(u, v):
        g[u][v]["weight"] += 1
    else:
        g.add_edge(u, v, weight=1)


def build_graph(precedents: Iterable[Document], train_qrels: dict[str, dict[str, int]],
                queries: Iterable[Document] = ()) -> nx.DiGraph:
    """Precedent nodes + train-query nodes; edges from train qrels and precedent citations.

    Raises ValueError if a query document from the val/test split is passed in (leakage guard).
    """
    g = nx.DiGraph()
    precedents = list(precedents)
    pool = {p.doc_id for p in precedents}
    for p in precedents:
        g.add_node(p.doc_id, **_attrs(p, "precedent"))
    for p in precedents:
        for cited in p.precedents_cited:
            if cited in pool:
                _add_edge(g, p.doc_id, cited)

    query_docs = {q.doc_id: q for q in queries}
    for q in query_docs.values():
        if q.split not in (None, "train"):
            raise ValueError(f"leakage guard: query {q.doc_id} is in split '{q.split}', only train may build the graph")
    for qid, rels in train_qrels.items():
        if qid not in g:
            q = query_docs.get(qid)
            g.add_node(qid, **(_attrs(q, "query") if q else {"kind": "query", "court": "unknown", "states": [], "date": ""}))
        for cited, grade in rels.items():
            if grade > 0 and cited in pool:
                _add_edge(g, qid, cited)
    return g


def binds_in(states: list[str], state: str) -> bool:
    return SUPREME in states or state in states


def jurisdiction_subgraphs(graph: nx.DiGraph, states: Iterable[str]) -> dict[str, nx.DiGraph]:
    """state -> subgraph of the nodes that bind in that state (SC + that state's High Court)."""
    out: dict[str, nx.DiGraph] = {}
    for state in states:
        nodes = [n for n, a in graph.nodes(data=True) if binds_in(a.get("states") or [], state)]
        if nodes:
            out[state] = graph.subgraph(nodes).copy()
    return out


def stats(graph: nx.DiGraph) -> dict[str, float]:
    """Numbers for the report: nodes, edges, cited precedents, mean in-degree of precedents."""
    prec = [n for n, a in graph.nodes(data=True) if a.get("kind") == "precedent"]
    indeg = [graph.in_degree(n) for n in prec]
    return {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "precedents": len(prec),
            "cited_precedents": sum(1 for d in indeg if d > 0),
            "mean_in_degree": (sum(indeg) / len(indeg)) if indeg else 0.0}


def save_graph(graph: nx.DiGraph, path: str | Path) -> None:
    """Save as node-link JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(graph, edges="links"), f)


def load_graph(path: str | Path) -> nx.DiGraph:
    """Load a graph saved by save_graph."""
    with open(path, encoding="utf-8") as f:
        return nx.node_link_graph(json.load(f), directed=True, edges="links")
