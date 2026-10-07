"""Similar cases ("more like this") for a precedent.  [owner: Gaurav — working]

Lawyers who find one good judgment want its neighbours. Three signals, each in [0, 1]:

    text       the case's own most characteristic terms (top tf-idf terms of its facts + reasoning
               zones) used as a BM25F query over the precedents, normalised by the best score
               (relevance-feedback style "more like this")
    coupling   VERSION-AWARE bibliographic coupling (Kessler, "Bibliographic coupling between
               scientific papers", 1963): Jaccard overlap of the OFFENCES the two judgments
               cite, so a 2015 judgment under IPC 302 and a 2025 one under BNS 103 couple
               (plain section-level coupling would score them 0)
    cocitation co-citation (Small, JASIS 1973): how often the two are cited together by the
               same case in the citation graph (train-split edges only), cosine-normalised

    sim = w_text * text + w_coupling * coupling + w_cocitation * cocitation   (configs: similar.*)

Combining text and citation similarity for Indian judgments follows Kumar et al. (IIT
Kharagpur, 2011, "Similarity analysis of legal judgments") and the statute-network view of
Hier-SPCNet (Bhattacharya et al., SIGIR 2020).

    from kanoon_bridge.rank.similar import SimilarCases
    sim = SimilarCases.load(engine, docs)
    sim.find("P01", k=10)          # [(doc_id, score, {"text":..., "coupling":..., "cocitation":...})]
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class SimilarCases:
    engine: object
    docs: object                                   # doc_id -> Document (index/docstore.py)
    graph: object | None = None                    # networkx DiGraph (rank/citation_graph.py)
    weights: dict = field(default_factory=lambda: {"text": 0.6, "coupling": 0.25, "cocitation": 0.15})
    n_terms: int = 40

    @classmethod
    def load(cls, engine, docs=None) -> "SimilarCases":
        from kanoon_bridge.config import project_path

        cfg = engine.cfg
        if docs is None:
            from kanoon_bridge.index.docstore import DocStore

            docs = DocStore.load(cfg)
        graph = None
        path = project_path(cfg.paths.graph)
        if path.exists():
            from kanoon_bridge.rank.citation_graph import load_graph

            graph = load_graph(path)
        scfg = cfg.get("similar", {}) or {}
        weights = {k: float(scfg.get(f"w_{k}", v)) for k, v in
                   {"text": 0.6, "coupling": 0.25, "cocitation": 0.15}.items()}
        return cls(engine=engine, docs=docs, graph=graph, weights=weights, n_terms=int(scfg.get("n_terms", 40)))

    # ------------------------------------------------------------------ signals
    def profile_terms(self, doc_id: str) -> dict[str, float]:
        """The case's top tf-idf terms (facts + reasoning zones), weights scaled to max 1."""
        from kanoon_bridge.text.pipeline import analyze_text

        doc = self.docs.get(doc_id)
        if doc is None:
            return {}
        text = "\n".join(p.text for p in doc.paragraphs if p.zone in ("facts", "ratio", "decision")) or doc.text
        whole = self.engine.precedent_index.whole
        tf = Counter(analyze_text(text[:20000], self.engine.analyzer.text_res, date=doc.decision_date))
        scored = {t: (1 + math.log10(c)) * whole.idf(t) for t, c in tf.items() if whole.df(t) > 1}
        top = sorted(scored.items(), key=lambda kv: -kv[1])[: self.n_terms]
        hi = top[0][1] if top else 1.0
        return {t: w / hi for t, w in top}

    def text_scores(self, doc_id: str) -> dict[str, float]:
        from kanoon_bridge.search import SearchOptions

        terms = self.profile_terms(doc_id)
        if not terms:
            return {}
        scorer = self.engine._scorer("precedent", SearchOptions(max_query_terms=self.n_terms))
        scores = scorer.score(terms, None)
        scores.pop(doc_id, None)
        top = max(scores.values(), default=0.0) or 1.0
        return {d: s / top for d, s in scores.items()}

    def _offences(self, doc_id: str) -> set[str]:
        meta = self.engine.facets.metas.get(doc_id)
        norm = self.engine.analyzer.text_res.normalizer
        if meta is None:
            return set()
        out = set()
        for ref in meta.statutes_cited:
            offs = norm.offences_for(ref) if norm is not None else []
            out.update(offs or [ref])                     # non-penal Acts (CrPC, ...) stay as sections
        return out

    def coupling_scores(self, doc_id: str) -> dict[str, float]:
        mine = self._offences(doc_id)
        if not mine:
            return {}
        fx = self.engine.facets
        norm = self.engine.analyzer.text_res.normalizer
        refs = set()
        for o in mine:                                    # every section, in either code, of my offences
            if o.startswith("off:") and norm is not None:
                refs.update(norm.sections_of(o, "ipc") + norm.sections_of(o, "bns"))
            else:
                refs.add(o)
        cands = fx.filter(doc_type="precedent", cites_any=sorted(refs)) - {doc_id}
        out = {}
        for d in cands:
            other = self._offences(d)
            if other:
                out[d] = len(mine & other) / len(mine | other)
        return out

    def cocitation_scores(self, doc_id: str) -> dict[str, float]:
        g = self.graph
        if g is None or doc_id not in g:
            return {}
        citers = set(g.predecessors(doc_id))
        if not citers:
            return {}
        counts: Counter = Counter()
        for c in citers:
            for d in g.successors(c):
                if d != doc_id:
                    counts[d] += 1
        return {d: n / math.sqrt(len(citers) * max(1, g.in_degree(d))) for d, n in counts.items()}

    # ------------------------------------------------------------------ combined
    def find(self, doc_id: str, k: int = 10) -> list[tuple[str, float, dict[str, float]]]:
        parts = {"text": self.text_scores(doc_id), "coupling": self.coupling_scores(doc_id),
                 "cocitation": self.cocitation_scores(doc_id)}
        precedents = self.engine.facets.filter(doc_type="precedent")
        total: dict[str, float] = {}
        for name, scores in parts.items():
            w = self.weights.get(name, 0.0)
            for d, s in scores.items():
                if d in precedents and d != doc_id:
                    total[d] = total.get(d, 0.0) + w * s
        ranked = sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(d, s, {n: round(parts[n].get(d, 0.0), 4) for n in parts}) for d, s in ranked]
