"""Boolean retrieval over the positional index.  [owner: A]

Evaluates a QueryNode tree (from query/parser.py) to a set of doc ids.

Lecture concepts shown here (point to this file in the video):
  * linear-merge postings intersection
  * query optimisation: intersect terms in order of increasing document frequency
  * NOT as set difference against the candidate universe
"""

from __future__ import annotations

from typing import Callable

from kanoon_bridge.index.positional import PositionalIndex
from kanoon_bridge.query.parser import Op, QueryNode


def intersect(p1: list[str], p2: list[str]) -> list[str]:
    """Linear merge of two sorted doc-id lists.

    Uses the textbook two-pointer intersection algorithm.
    """
    i = 0
    j = 0
    result: list[str] = []

    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            result.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1

    return result


def intersect_many(postings: list[list[str]]) -> list[str]:
    """Intersect many lists, shortest first.

    Lists are sorted by length so that the smallest candidate set is
    processed first. The intersection stops immediately if it becomes empty.
    """
    if not postings:
        return []

    ordered = sorted(postings, key=len)

    result = ordered[0]

    for posting in ordered[1:]:
        if not result:
            break

        result = intersect(result, posting)

    return result


def evaluate(
    node: QueryNode,
    index: PositionalIndex,
    analyze: Callable[[str], list[str]],
    universe: set[str] | None = None,
) -> set[str]:
    """Evaluate a QueryNode tree against a positional index.

    `analyze` converts query terms into the same normalized tokens used
    when indexing documents.

    TERM:
        Returns documents containing all analyzed tokens.

    PHRASE:
        Uses the positional index's phrase query.

    AND:
        Intersects the results of both children.

    OR:
        Takes the union of both children.

    NOT:
        Returns the universe minus the child's results.

    PROX:
        Evaluates a proximity query using query.proximity.within.
    """
    if node.op == Op.TERM:
        tokens = analyze(node.value)

        if not tokens:
            return set()

        postings: list[list[str]] = []

        for token in tokens:
            docs = sorted(index.docs_with(token))
            postings.append(docs)

        return set(intersect_many(postings))

    if node.op == Op.PHRASE:
        tokens = analyze(node.value)

        if not tokens:
            return set()

        return index.phrase(tokens)

    if node.op == Op.AND:
        if len(node.children) < 2:
            return set()

        child_results = [
            evaluate(child, index, analyze, universe)
            for child in node.children
        ]

        child_results.sort(key=len)

        result = child_results[0]

        for child_result in child_results[1:]:
            result = result.intersection(child_result)

            if not result:
                break

        return result

    if node.op == Op.OR:
        result: set[str] = set()

        for child in node.children:
            result.update(
                evaluate(child, index, analyze, universe)
            )

        return result

    if node.op == Op.NOT:
        if universe is None:
            universe = set(index.doc_len)

        if not node.children:
            return set(universe)

        child_result = evaluate(
            node.children[0],
            index,
            analyze,
            universe,
        )

        return set(universe) - child_result

    if node.op == Op.PROX:
        if len(node.children) != 2:
            return set()

        from kanoon_bridge.query.proximity import within

        left = node.children[0]
        right = node.children[1]

        left_terms = _node_terms(left, analyze)
        right_terms = _node_terms(right, analyze)

        if not left_terms or not right_terms:
            return set()

        return within(
            index,
            left_terms,
            right_terms,
            node.k,
        )

    raise ValueError(f"Unsupported query operator: {node.op}")


def _node_terms(
    node: QueryNode,
    analyze: Callable[[str], list[str]],
) -> list[str]:
    """Convert a TERM or PHRASE node into normalized index terms."""
    if node.op == Op.TERM:
        return analyze(node.value)

    if node.op == Op.PHRASE:
        return analyze(node.value)

    return []