"""Query planning: one user question -> several sub-queries.  [layer 2 — working]

Sub-query kinds (each maps to an IR technique you can name in the report):

    original     the user's query as typed                         (always)
    cross_code   section numbers replaced by their other-code      version normalisation
                 equivalents ("BNS 103" -> "IPC 302")
    boolean      AND of OR-clauses: the offence clause (offence    Boolean retrieval over
                 id OR its IPC/BNS sections) AND the most salient  postings, query
                 content terms; executed as a candidate filter     optimisation (rarest
                 (SearchOptions.require_terms), then ranked        clause first)
    facet        same text, precedents restricted to those         parametric index
                 binding in the user's state (SC + its HC)
    statutes     statute-only search with the offence names and    zone/parametric search
                 lexicon expansions spelled out
    reformulated (round 2, agent/reflect.py) pseudo-relevance feedback
    llm          optional: sub-query text proposed by Claude (LLMPlanner / HybridPlanner)

RulePlanner is the default (no LLM, deterministic, easy to evaluate). The LLM planners only
PROPOSE sub-query text; ranking is always the core retriever plus fusion.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from kanoon_bridge.schema import AnalyzedQuery, Query
from kanoon_bridge.text.tokenize import extract_sections


@dataclass
class SubQuery:
    sq_id: str                         # "q0", "q1", ... ("r2q0" for round 2)
    kind: str                          # original | cross_code | boolean | facet | statutes | reformulated | llm
    query: Query                       # what gets passed to SearchEngine.search
    target: str = "precedent"          # which result list this sub-query contributes to
    weight: float = 1.0                # fusion weight (agent/fuse.py)
    options: dict[str, Any] = field(default_factory=dict)   # SearchOptions overrides, e.g. {"restrict_states": ["delhi"]}
    rationale: str = ""                # one line, shown in --debug and the video


@dataclass
class Plan:
    subqueries: list[SubQuery] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    prefix: str = "q"

    def add(self, kind: str, query: Query, rationale: str, **kw) -> SubQuery:
        sq = SubQuery(sq_id=f"{self.prefix}{len(self.subqueries)}", kind=kind, query=query, rationale=rationale, **kw)
        self.subqueries.append(sq)
        return sq


def _variant(query: Query, text: str) -> Query:
    q = copy.deepcopy(query)
    q.text = text
    q.query_id = None
    return q


def _is_content(t: str) -> bool:
    return not (t.startswith(("sec:", "off:")) or t.isdigit() or len(t) < 3)


def _section_text(ref: str) -> str:
    code, _, num = ref.partition(":")
    return f"{code.upper()} Section {num}" if code in ("ipc", "bns") else f"Section {num}"


@dataclass
class RulePlanner:
    """Deterministic planner. `df` (term -> document frequency in the precedent index) keeps
    the Boolean rule away from terms too rare to intersect; `normalizer` names offences."""

    max_boolean_terms: int = 3
    min_df: int = 2
    facet_weight: float = 0.8
    statute_weight: float = 1.0
    normalizer: object | None = None
    df: Callable[[str], int] | None = None

    def plan(self, query: Query, aq: AnalyzedQuery, idf: dict[str, float] | None = None) -> Plan:
        p = Plan()
        p.add("original", query, "the question as asked")
        for rule in (self._cross_code, self._boolean, self._facet, self._statutes):
            try:
                rule(p, query, aq, idf or {})
            except NotImplementedError as err:
                p.notes.append(f"skipped {rule.__name__.strip('_')}: {err}")
        return p

    # ------------------------------------------------------------------ rules
    def _cross_code(self, p: Plan, query: Query, aq: AnalyzedQuery, idf: dict[str, float]) -> None:
        own = set(aq.sections)
        others = [k[len("sec:"):] for k in aq.expanded_terms if k.startswith("sec:") and k[len("sec:"):] not in own]
        others = [r for r in dict.fromkeys(others) if r.split(":")[0] in ("ipc", "bns")]
        if not others:
            return
        text = query.text
        for m in sorted(extract_sections(text), key=lambda m: -m.start):     # drop the original mentions
            text = text[: m.start] + " " + text[m.end:]
        text = " ".join((text + " " + " ".join(_section_text(r) for r in others)).split())
        mine = ", ".join(_section_text(r) for r in own if r.split(":")[0] in ("ipc", "bns")) or "the query's sections"
        p.add("cross_code", _variant(query, text),
              f"{mine} -> {', '.join(_section_text(r) for r in others)}: precedents decided under the other code")

    def _boolean(self, p: Plan, query: Query, aq: AnalyzedQuery, idf: dict[str, float]) -> None:
        clauses: list[list[str]] = []
        if aq.offence_ids:
            offence = list(dict.fromkeys(aq.offence_ids))
            secs = [t for t in aq.tokens if t.startswith("sec:")] + [k for k in aq.expanded_terms if k.startswith("sec:")]
            clauses.append(list(dict.fromkeys(offence + secs)))
        counts = Counter(t for t in aq.tokens if _is_content(t))
        for t in aq.expanded_terms:
            if _is_content(t):
                counts[t] += 1
        pool = [t for t in counts if idf.get(t, 0.0) > 0 and (self.df is None or self.df(t) >= self.min_df)]
        pool.sort(key=lambda t: (counts[t] * idf[t], t), reverse=True)      # salient: tf x idf
        n_terms = 2 if len(pool) <= 4 else self.max_boolean_terms
        n_terms -= 1 if clauses else 0
        clauses += [[t] for t in pool[:max(0, n_terms)]]
        if len(clauses) < 2:
            p.notes.append("boolean: fewer than two usable clauses - skipped")
            return
        shown = " AND ".join("(" + " OR ".join(c) + ")" if len(c) > 1 else c[0] for c in clauses)
        p.add("boolean", _variant(query, query.text), f"documents matching {shown}, ranked",
              options={"require_terms": clauses})

    def _facet(self, p: Plan, query: Query, aq: AnalyzedQuery, idf: dict[str, float]) -> None:
        if not query.state or query.filters.get("court"):
            return
        p.add("facet", _variant(query, query.text),
              f"only precedents binding in {query.state} (Supreme Court + its High Court)",
              weight=self.facet_weight, options={"restrict_states": [query.state]})

    def _statutes(self, p: Plan, query: Query, aq: AnalyzedQuery, idf: dict[str, float]) -> None:
        labels = []
        if self.normalizer is not None:
            labels = [self.normalizer.label(o) for o in dict.fromkeys(aq.offence_ids)]
            labels = [l for l in labels if l and not l.startswith("off:")]
        words = [t for t in aq.expanded_terms if _is_content(t) and t not in aq.tokens]
        if not labels and not words:
            return
        text = " ".join([query.text] + labels + words)
        p.add("statutes", _variant(query, text),
              "statute search with " + ", ".join((["offence names: " + "; ".join(labels)] if labels else [])
                                                 + (["expansions: " + " ".join(words)] if words else [])),
              target="statute", weight=self.statute_weight)


# ---------------------------------------------------------------------- optional LLM planners

LLM_PROMPT = """You help search a database of Indian criminal statutes (IPC and BNS) and court judgments.
Rewrite the user's question into 2 to 4 short keyword search queries (each under 12 words) that
together cover it: legal terms for the facts, the likely offence names, and section numbers if
you are sure of them. The incident date decides the code: BNS on or after 1 July 2024, IPC before.

Question: {question}
Incident date: {date}   User's state: {state}
Query analysis: {analysis}

Reply with ONLY a JSON list of strings."""


def parse_llm_queries(text: str, limit: int = 4) -> list[str]:
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        items = []
    out = [" ".join(str(s).split()) for s in items if isinstance(s, str)]
    return [s for s in dict.fromkeys(out) if 0 < len(s) <= 200][:limit]


@dataclass
class LLMPlanner:
    """Original query + 2-4 sub-queries proposed by Claude (validated like any user query).
    Needs ANTHROPIC_API_KEY (see rag/generate.py). Falls back to the original only, noting why."""

    model: str = ""
    weight: float = 0.7

    def propose(self, query: Query, aq: AnalyzedQuery) -> list[str]:
        from kanoon_bridge.rag.generate import call_claude

        analysis = "; ".join(f"{k}: {v}" for k, v in aq.trace if k in ("language", "normalised", "offences", "code_in_force"))
        prompt = LLM_PROMPT.format(question=query.text, date=query.incident_date or "unknown",
                                   state=query.state or "unknown", analysis=analysis)
        return parse_llm_queries(call_claude(prompt, self.model or None, max_tokens=300,
                                             system="You write search queries. Reply with JSON only."))

    def add_to(self, p: Plan, query: Query, aq: AnalyzedQuery) -> None:
        try:
            for text in self.propose(query, aq):
                p.add("llm", _variant(query, text), "proposed by the LLM planner", weight=self.weight)
        except (RuntimeError, ImportError) as err:
            p.notes.append(f"llm planner skipped: {err}")

    def plan(self, query: Query, aq: AnalyzedQuery, idf: dict[str, float] | None = None) -> Plan:
        p = Plan()
        p.add("original", query, "the question as asked")
        self.add_to(p, query, aq)
        return p


@dataclass
class HybridPlanner:
    """RulePlanner's sub-queries plus the LLM's."""

    rules: RulePlanner
    llm: LLMPlanner

    def plan(self, query: Query, aq: AnalyzedQuery, idf: dict[str, float] | None = None) -> Plan:
        p = self.rules.plan(query, aq, idf)
        self.llm.add_to(p, query, aq)
        return p


def make_planner(cfg, normalizer=None, df: Callable[[str], int] | None = None):
    """Planner named by configs/default.yaml agent.planner: rules | llm | hybrid."""
    acfg = cfg.agent
    rules = RulePlanner(max_boolean_terms=acfg.get("boolean_terms", 3), min_df=acfg.get("boolean_min_df", 2),
                        facet_weight=acfg.get("facet_weight", 0.8), normalizer=normalizer, df=df)
    kind = acfg.get("planner", "rules")
    if kind == "rules":
        return rules
    llm = LLMPlanner(model=acfg.get("llm_model", "") or "")
    if kind == "llm":
        return llm
    if kind == "hybrid":
        return HybridPlanner(rules, llm)
    raise ValueError(f"unknown agent.planner {kind!r} (rules | llm | hybrid)")
