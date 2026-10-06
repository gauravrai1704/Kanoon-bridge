"""Statute bridge: use the best statutes for a query to improve precedent ranking.  [owner: Gaurav — working]

IL-PCSR's best result conditions a GPT-4.1 re-ranker on statutes. We get the same cross-task
signal with IR alone:

  1. take the top-N statutes for the query (already scored by BM25F over statutes); weight each
     by its score relative to the best one
  2. EXPAND the precedent query with those statutes' section and offence tokens
  3. BOOST precedents that cite any of those sections. Through the offence ids, a BNS statute
     also reaches precedents that cite its IPC counterpart (BNS 103 -> precedents citing IPC 302)
  4. scale boosts to [0, boost] so they add to normalised relevance in search.py

    out = bridge.run(statute_hits)
    out.expansion   {"sec:bns:103": 1.0, "off:murder_bns103": 1.0, ...}
    out.boosts      {precedent_id: 0.3, ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kanoon_bridge.index.facets import FacetIndex
from kanoon_bridge.schema import ScoredDoc


@dataclass
class BridgeOutput:
    expansion: dict[str, float] = field(default_factory=dict)
    boosts: dict[str, float] = field(default_factory=dict)
    statutes_used: list[str] = field(default_factory=list)


@dataclass
class StatuteBridge:
    facets: FacetIndex
    statute_terms: dict[str, list[str]] = field(default_factory=dict)  # statute doc_id -> ["sec:bns:103", "off:..."]
    top_n: int = 10
    boost: float = 0.3
    normalizer: object | None = None       # text.version_norm.VersionNormalizer (cross-code reach)
    expansion_weight: float = 0.5           # query weight of added tokens, relative to typed terms

    def _section_refs(self, tokens: list[str]) -> set[str]:
        refs = {t[4:] for t in tokens if t.startswith("sec:")}
        if self.normalizer is not None:
            for off in (t for t in tokens if t.startswith("off:")):
                for code in ("ipc", "bns"):
                    refs.update(self.normalizer.sections_of(off, code))
            for ref in list(refs):
                refs.update(self.normalizer.equivalents(ref))
        return refs

    def run(self, statute_hits: list[ScoredDoc]) -> BridgeOutput:
        out = BridgeOutput()
        hits = [h for h in statute_hits[: self.top_n] if h.score > 0]
        if not hits:
            return out
        best = hits[0].score
        raw: dict[str, float] = {}
        for h in hits:
            w = h.score / best
            tokens = self.statute_terms.get(h.doc_id, [])
            for t in tokens:
                out.expansion[t] = max(out.expansion.get(t, 0.0), w * self.expansion_weight)
            for ref in self._section_refs(tokens):
                for prec in self.facets.by_section.get(ref, ()):
                    raw[prec] = raw.get(prec, 0.0) + w
            out.statutes_used.append(h.doc_id)
        if raw:
            top = max(raw.values())
            out.boosts = {d: self.boost * v / top for d, v in raw.items()}
        return out
