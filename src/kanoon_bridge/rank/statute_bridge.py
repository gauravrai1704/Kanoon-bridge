"""Statute bridge: use the best statutes for a query to improve precedent ranking.  [owner: D]

IL-PCSR's best result conditions a GPT-4.1 re-ranker on statutes. We get the same
cross-task signal with IR alone:

  1. take the top-N statutes for the query (already scored by BM25F over statutes)
  2. EXPAND the precedent query with those statutes' section refs and offence ids
     (weight = normalised statute score)
  3. BOOST precedents that cite any of those statutes (FacetIndex.by_section),
     proportional to how many and how highly ranked

    out = bridge.run(statute_hits)
    out.expansion   {"sec:ipc:302": 0.9, "off:murder": 0.9, ...}
    out.boosts      {precedent_id: 0.42, ...}
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
    statute_terms: dict[str, list[str]] = field(default_factory=dict)  # statute doc_id -> ["sec:bns:103", "off:murder"]
    top_n: int = 10
    boost: float = 0.3

    def run(self, statute_hits: list[ScoredDoc]) -> BridgeOutput:
        """TODO(D): implement steps 1-3; min-max normalise boosts to [0, boost]."""
        raise NotImplementedError("TODO(D): statute bridge")
