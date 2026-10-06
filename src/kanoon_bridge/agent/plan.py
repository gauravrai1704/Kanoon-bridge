"""Query planning: one user question -> several sub-queries.  [agent owner]

Sub-query kinds (each maps to an IR technique you can name in the report):

    original     the user's query as typed                         (always; working)
    cross_code   section numbers replaced by their other-code      version normalisation
                 equivalents ("BNS 103" -> "IPC 302")
    boolean      AND of the 2-3 highest-idf terms, OR over their    Boolean retrieval,
                 lexicon synonyms                                   query optimisation
    facet        same text, restricted to precedents binding in     parametric index
                 the user's state (court + states facets)
    statutes     statute-only search, feeds the statute bridge      zone/parametric search

RulePlanner is the default (no LLM, deterministic, easy to evaluate). LLMPlanner is an
optional variant: the LLM only PROPOSES sub-query text; it never ranks documents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kanoon_bridge.schema import AnalyzedQuery, Query


@dataclass
class SubQuery:
    sq_id: str                         # "q0", "q1", ... (stable order for the trace)
    kind: str                          # original | cross_code | boolean | facet | statutes | reformulated
    query: Query                       # what gets passed to SearchEngine.search
    target: str = "precedent"          # which result list this sub-query contributes to
    weight: float = 1.0                # fusion weight (agent/fuse.py)
    options: dict[str, Any] = field(default_factory=dict)   # SearchOptions overrides, e.g. {"jurisdiction": False}
    rationale: str = ""                # one line, shown in --debug and the video


@dataclass
class Plan:
    subqueries: list[SubQuery] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, kind: str, query: Query, rationale: str, **kw) -> SubQuery:
        sq = SubQuery(sq_id=f"q{len(self.subqueries)}", kind=kind, query=query, rationale=rationale, **kw)
        self.subqueries.append(sq)
        return sq


@dataclass
class RulePlanner:
    """Deterministic planner. Needs the analyzer's output and index idf for the Boolean rule."""

    max_boolean_terms: int = 3

    def plan(self, query: Query, aq: AnalyzedQuery, idf: dict[str, float] | None = None) -> Plan:
        p = Plan()
        p.add("original", query, "the question as asked")  # working: always present
        for rule in (self._cross_code, self._boolean, self._facet, self._statutes):
            try:
                rule(p, query, aq, idf or {})
            except NotImplementedError as err:
                p.notes.append(f"skipped {rule.__name__.strip('_')}: {err}")
        return p

    def _cross_code(self, p: Plan, query: Query, aq: AnalyzedQuery, idf: dict[str, float]) -> None:
        """TODO: if aq.expanded_terms holds cross-code sections (added by the analyzer from
        VersionNormalizer.equivalents), add a sub-query whose text names the other-code sections."""
        raise NotImplementedError("TODO(agent): cross-code sub-query")

    def _boolean(self, p: Plan, query: Query, aq: AnalyzedQuery, idf: dict[str, float]) -> None:
        """TODO: pick the max_boolean_terms highest-idf analysed terms; build
        '(t1 OR syn1) AND (t2 OR syn2)' using aq.expanded_terms as synonyms. Needs the Boolean
        parser (query/parser.py) to be implemented, else skip."""
        raise NotImplementedError("TODO(agent): Boolean sub-query")

    def _facet(self, p: Plan, query: Query, aq: AnalyzedQuery, idf: dict[str, float]) -> None:
        """TODO: when query.state is set, add the same text with filters restricting precedents
        to the Supreme Court + the user's High Court (binding only), weight 0.8."""
        raise NotImplementedError("TODO(agent): facet sub-query")

    def _statutes(self, p: Plan, query: Query, aq: AnalyzedQuery, idf: dict[str, float]) -> None:
        """TODO: statute-targeted sub-query (target='statute'), code restricted to aq.code_in_force."""
        raise NotImplementedError("TODO(agent): statutes-first sub-query")


@dataclass
class LLMPlanner:
    """Optional: ask an LLM for 2-4 sub-query strings, then validate them like any user query.

    TODO(agent, optional): prompt with the question + analyzer trace, parse a JSON list,
    keep at most 4, label kind='llm'. Compare against RulePlanner in eval/agent_eval.py.
    """

    model: str = ""

    def plan(self, query: Query, aq: AnalyzedQuery, idf: dict[str, float] | None = None) -> Plan:
        raise NotImplementedError("TODO(agent): LLM planner (optional)")
