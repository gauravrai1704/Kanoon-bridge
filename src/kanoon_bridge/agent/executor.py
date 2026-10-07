"""Run every sub-query through the core retriever.  [agent owner — working]

Each sub-query goes through search.SearchEngine.search, exactly like a user query (rule 4),
with its own SearchOptions overrides. Results keep the sub-query they came from, so the
trace can show "q2 (boolean) found p17 at rank 3".
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from kanoon_bridge.agent.plan import Plan, SubQuery
from kanoon_bridge.schema import SearchResult


@dataclass
class SubResult:
    subquery: SubQuery
    result: SearchResult

    def ranking(self, target: str | None = None) -> list[str]:
        target = target or self.subquery.target
        hits = self.result.precedents if target == "precedent" else self.result.statutes
        return [h.doc_id for h in hits]

    def scores(self, target: str | None = None) -> list[float]:
        target = target or self.subquery.target
        hits = self.result.precedents if target == "precedent" else self.result.statutes
        return [h.score for h in hits]


def run_plan(engine, plan: Plan, base_options=None, depth: int = 50) -> list[SubResult]:
    """Search each sub-query; `depth` results per list go into fusion."""
    from kanoon_bridge.search import SearchOptions

    out: list[SubResult] = []
    for sq in plan.subqueries:
        opt = copy.deepcopy(base_options) if base_options is not None else SearchOptions()
        for key, value in sq.options.items():
            setattr(opt, key, value)
        opt.top_k = depth
        out.append(SubResult(subquery=sq, result=engine.search(copy.deepcopy(sq.query), opt)))
    return out
