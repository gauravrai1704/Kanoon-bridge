"""Static quality g(d) and jurisdiction-aware authority g(d | state).  [owner: Gaurav — working]

    net score = normalised relevance + lambda * g(d | state)          (configs: authority.lambda)

g(d)          weighted PageRank on the citation graph, min-max scaled to [0, 1] over precedents
g(d | state)  weight(d, state) * blend(g(d), g_state(d)), where
                  weight = binding_weight (1.0)    Supreme Court, or the user's own High Court
                           persuasive_weight (0.4) another state's High Court
                           their mean               court unknown
                  blend  = 0.5 g(d) + 0.5 g_state(d) when d is in that state's subgraph
                           (PageRank inside the jurisdiction), else g(d)

`binding_status` is also used by eval/qrels.py to grade E6. Simplifications (bench size,
overruling) are listed in the proposal's limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from kanoon_bridge.config import Config, load_config
from kanoon_bridge.index.facets import DocMeta

SUPREME_COURT_STATE = "*"


def binding_status(meta: DocMeta, user_state: str | None) -> str:
    """'binding' | 'persuasive' | 'unknown' for a precedent, from the user's state."""
    if meta.court == "supreme_court" or SUPREME_COURT_STATE in meta.states:
        return "binding"
    if user_state is None or not meta.states:
        return "unknown"
    return "binding" if user_state.lower() in meta.states else "persuasive"


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {d: (1.0 if hi > 0 else 0.0) for d in scores}
    return {d: (s - lo) / (hi - lo) for d, s in scores.items()}


def pagerank(graph: nx.DiGraph, damping: float = 0.85, max_iter: int = 200, tol: float = 1e-10) -> dict[str, float]:
    """Weighted PageRank by power iteration (no SciPy needed).

        PR(v) = (1 - d) / N  +  d * ( sum_{u -> v} PR(u) * w(u, v) / W_out(u)  +  dangling / N )

    `dangling` is the rank of nodes with no out-links (most precedents cite nothing in the pool),
    spread uniformly so ranks keep summing to 1.
    """
    nodes = list(graph.nodes)
    n = len(nodes)
    if n == 0:
        return {}
    out_w = {u: sum(d.get("weight", 1) for _, _, d in graph.out_edges(u, data=True)) for u in nodes}
    rank = {v: 1.0 / n for v in nodes}
    for _ in range(max_iter):
        dangling = sum(rank[u] for u in nodes if out_w[u] == 0)
        new = {v: (1 - damping) / n + damping * dangling / n for v in nodes}
        for u, v, data in graph.edges(data=True):
            new[v] += damping * rank[u] * data.get("weight", 1) / out_w[u]
        delta = sum(abs(new[v] - rank[v]) for v in nodes)
        rank = new
        if delta < tol * n:
            break
    return rank


def _pagerank(graph: nx.DiGraph, damping: float) -> dict[str, float]:
    return pagerank(graph, damping)


@dataclass
class Authority:
    cfg: Config
    global_scores: dict[str, float] = field(default_factory=dict)
    state_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    metas: dict[str, DocMeta] = field(default_factory=dict)

    @classmethod
    def compute(cls, graph: nx.DiGraph, metas: dict[str, DocMeta], cfg: Config | None = None,
                subgraphs: dict[str, nx.DiGraph] | None = None) -> "Authority":
        """PageRank on the whole graph and on each jurisdiction subgraph; scores kept for precedents only."""
        cfg = cfg or load_config()
        damping = cfg.authority.pagerank_damping

        def precedent_scores(g: nx.DiGraph) -> dict[str, float]:
            pr = _pagerank(g, damping)
            return _minmax({n: s for n, s in pr.items() if g.nodes[n].get("kind") == "precedent"})

        auth = cls(cfg=cfg, global_scores=precedent_scores(graph), metas=metas)
        for state, sub in (subgraphs or {}).items():
            scores = precedent_scores(sub)
            if scores:
                auth.state_scores[state] = scores
        return auth

    def weight(self, doc_id: str, state: str | None) -> float:
        a = self.cfg.authority
        meta = self.metas.get(doc_id)
        status = binding_status(meta, state) if meta else "unknown"
        if status == "binding":
            return a.binding_weight
        if status == "persuasive":
            return a.persuasive_weight
        return (a.binding_weight + a.persuasive_weight) / 2

    def score(self, doc_id: str, state: str | None = None, jurisdiction: bool = True) -> float:
        """g(d) when jurisdiction is False or no state is given, else g(d | state). 0 if unseen."""
        g = self.global_scores.get(doc_id, 0.0)
        if not jurisdiction or state is None:
            return g
        local = self.state_scores.get(state.lower(), {}).get(doc_id)
        blended = 0.5 * g + 0.5 * local if local is not None else g
        return self.weight(doc_id, state.lower()) * blended

    def top(self, k: int = 10, state: str | None = None) -> list[tuple[str, float]]:
        """Most authoritative precedents overall or for a state (for the report and demo)."""
        ids = self.global_scores if state is None else set(self.global_scores) | set(self.state_scores.get(state, {}))
        return sorted(((d, self.score(d, state)) for d in ids), key=lambda x: -x[1])[:k]

    def to_dict(self) -> dict:
        return {"global": self.global_scores, "state": self.state_scores}

    @classmethod
    def from_dict(cls, d: dict, metas: dict[str, DocMeta], cfg: Config | None = None) -> "Authority":
        return cls(cfg=cfg or load_config(), global_scores=d["global"], state_scores=d["state"], metas=metas)
