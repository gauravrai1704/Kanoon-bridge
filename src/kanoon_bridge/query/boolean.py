"""Boolean retrieval over the positional index.  [owner: A — working]

Evaluates a QueryNode tree (from query/parser.py) to a set of doc ids.

Lecture concepts shown here (point to this file in the video):
  * linear-merge postings intersection (`intersect`, two pointers over sorted doc ids)
  * query optimisation: intersect terms in order of increasing document frequency
  * NOT as set difference against the candidate universe
  * phrase and proximity operators from positions
"""

from __future__ import annotations

from typing import Callable

from kanoon_bridge.index.positional import PositionalIndex
from kanoon_bridge.query.parser import Op, QueryNode
from kanoon_bridge.query.proximity import within_positions


def intersect(p1: list[str], p2: list[str]) -> list[str]:
    """Linear merge of two sorted doc-id lists (textbook two-pointer algorithm)."""
    out: list[str] = []
    i = j = 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            out.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1
    return out


def intersect_many(postings: list[list[str]]) -> list[str]:
    """Intersect many sorted lists, shortest first (lecture query optimisation); stops early
    as soon as the running result is empty."""
    if not postings:
        return []
    ordered = sorted(postings, key=len)
    result = ordered[0]
    for plist in ordered[1:]:
        if not result:
            break
        result = intersect(result, plist)
    return result


def _positions(node: QueryNode, index: PositionalIndex, analyze) -> dict[str, list[int]] | None:
    """doc -> sorted start positions for a TERM or PHRASE (a term that analyses to several
    tokens is treated as a phrase). None for an empty term (only stop words)."""
    tokens = analyze(node.value)
    if not tokens:
        return None
    if len(tokens) == 1:
        return index.postings.get(tokens[0], {})
    out: dict[str, list[int]] = {}
    for d in index.phrase(tokens):
        starts = set(index.postings[tokens[0]][d])
        for i, t in enumerate(tokens[1:], start=1):
            starts &= {p - i for p in index.postings[t][d]}
        out[d] = sorted(starts)
    return out


def evaluate(node: QueryNode, index: PositionalIndex, analyze: Callable[[str], list[str]],
             universe: set[str] | None = None) -> set[str]:
    """Evaluate the tree; `analyze` turns a term/phrase into index terms (same pipeline as docs).

    TERM   docs containing its analysed token; a section reference ("BNS 103") matches any doc
           with its offence id, so Boolean search is version-aware like ranking
           (several plain tokens, e.g. "breach-of-trust", act as a phrase)
    PHRASE index.phrase over its analysed tokens
    AND    intersect_many over sorted postings (shortest first); NOT children subtract
    OR     union
    NOT    universe minus the child
    PROX   both sides within k positions (TERM/PHRASE sides); other sides fall back to AND
    A term made only of stop words places no constraint.
    """
    universe = set(index.doc_len) if universe is None else universe
    op = node.op
    if op in (Op.TERM, Op.PHRASE):
        tokens = analyze(node.value)
        if not tokens:
            return set(universe)
        offences = [t for t in tokens if t.startswith("off:")]
        if offences and op == Op.TERM:
            # a section reference ("302 IPC", "BNS 103") matches its offence in either code
            out: set[str] = set()
            for t in offences:
                out |= set(index.postings.get(t, {}))
            return out & universe
        if op == Op.PHRASE or len(tokens) > 1:
            return index.phrase(tokens) & universe
        return set(index.postings.get(tokens[0], {})) & universe
    if op == Op.OR:
        out: set[str] = set()
        for child in node.children:
            out |= evaluate(child, index, analyze, universe)
        return out
    if op == Op.NOT:
        return universe - evaluate(node.children[0], index, analyze, universe)
    if op == Op.AND:
        positive = [c for c in node.children if c.op != Op.NOT]
        negative = [c for c in node.children if c.op == Op.NOT]
        lists = [sorted(evaluate(c, index, analyze, universe)) for c in positive]
        result = set(intersect_many(lists)) if lists else set(universe)
        for c in negative:                                   # a AND NOT b  =  a - b
            if not result:
                break
            result -= evaluate(c.children[0], index, analyze, universe)
        return result
    if op == Op.PROX:
        left, right = node.children
        if all(c.op in (Op.TERM, Op.PHRASE) for c in (left, right)):
            p1, p2 = _positions(left, index, analyze), _positions(right, index, analyze)
            if p1 is None or p2 is None:                     # a stop-word side: just the other side
                return evaluate(right if p1 is None else left, index, analyze, universe)
            return within_positions(p1, p2, node.k) & universe
        return evaluate(QueryNode(Op.AND, children=[left, right]), index, analyze, universe)
    raise ValueError(f"unknown operator {op}")
