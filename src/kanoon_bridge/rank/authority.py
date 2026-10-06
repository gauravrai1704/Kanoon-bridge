"""Static quality g(d) and jurisdiction-aware authority g(d | state).  [owner: D]

    net score = relevance + lambda * g(d | state)            (configs: authority.lambda)

g(d)          PageRank on the citation graph, min-max scaled to [0, 1]
g(d | state)  g(d) x weight, where weight is
                  binding_weight     (1.0) for the Supreme Court or the user's own High Court
                  persuasive_weight  (0.4) for other High Courts
              optionally blended with PageRank on that state's subgraph

`binding_status` is a working baseline rule, also used by eval/qrels.py to grade E6.
Its simplifications (bench size, overruling) are listed in the proposal's limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from kanoon_bridge.config import Config, load_config
from kanoon_bridge.index.facets import DocMeta

SUPREME_COURT_STATE = "*"


def binding_status(meta: DocMeta, user_state: str | None) -> str:
    """'binding' | 'persuasive' | 'unknown' for a precedent, from the user's state (working)."""
    if meta.court == "supreme_court" or SUPREME_COURT_STATE in meta.states:
        return "binding"
    if user_state is None or not meta.states:
        return "unknown"
    return "binding" if user_state.lower() in meta.states else "persuasive"


@dataclass
class Authority:
    cfg: Config
    global_scores: dict[str, float] = field(default_factory=dict)
    state_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    metas: dict[str, DocMeta] = field(default_factory=dict)

    @classmethod
    def compute(cls, graph: nx.DiGraph, metas: dict[str, DocMeta], cfg: Config | None = None,
                subgraphs: dict[str, nx.DiGraph] | None = None) -> "Authority":
        """TODO(D): nx.pagerank(graph, alpha=cfg.authority.pagerank_damping), min-max scale;
        same per subgraph into state_scores. Docs missing from the graph get 0."""
        raise NotImplementedError("TODO(D): PageRank authority")

    def score(self, doc_id: str, state: str | None = None, jurisdiction: bool = True) -> float:
        """g(d) when jurisdiction is False or state is None, else g(d | state).

        TODO(D): use binding_status + cfg.authority weights; blend state_scores if present.
        """
        raise NotImplementedError("TODO(D): jurisdiction-aware authority")

    # Saving/loading as plain dicts (working)
    def to_dict(self) -> dict:
        return {"global": self.global_scores, "state": self.state_scores}

    @classmethod
    def from_dict(cls, d: dict, metas: dict[str, DocMeta], cfg: Config | None = None) -> "Authority":
        return cls(cfg=cfg or load_config(), global_scores=d["global"], state_scores=d["state"], metas=metas)
