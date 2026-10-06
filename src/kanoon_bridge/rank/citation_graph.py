"""Citation graph over precedents.  [owner: D]

Edges u -> v mean "u cites v". Sources, with NO test leakage (team rule 3):
  * train-split IL-PCSR qrels: query case -> cited precedent
  * precedent texts (unmasked): precedent -> precedent it cites, when resolvable to an id

jurisdiction_subgraphs() keeps, per state, the edges among documents that bind there
(the state's own High Court plus the Supreme Court) — graph mining per jurisdiction, as
in the proposal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import networkx as nx

from kanoon_bridge.schema import Document


def build_graph(precedents: Iterable[Document], train_qrels: dict[str, dict[str, int]],
                queries: Iterable[Document] = ()) -> nx.DiGraph:
    """TODO(D): add every precedent as a node (with court/states attributes), add edges from
    train qrels and from Document.precedents_cited. Assert no test query id appears."""
    raise NotImplementedError("TODO(D): build citation graph")


def jurisdiction_subgraphs(graph: nx.DiGraph, states: Iterable[str]) -> dict[str, nx.DiGraph]:
    """state -> subgraph of nodes binding in that state. TODO(D)."""
    raise NotImplementedError("TODO(D): per-jurisdiction subgraphs")


def save_graph(graph: nx.DiGraph, path: str | Path) -> None:
    """Save as node-link JSON (working)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(graph, edges="links"), f)


def load_graph(path: str | Path) -> nx.DiGraph:
    """Load a graph saved by save_graph (working)."""
    with open(path, encoding="utf-8") as f:
        return nx.node_link_graph(json.load(f), directed=True, edges="links")
