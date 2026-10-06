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
from kanoon_bridge.query.parser import QueryNode


def intersect(p1: list[str], p2: list[str]) -> list[str]:
    """Linear merge of two sorted doc-id lists.

    TODO(A): the textbook two-pointer algorithm (do not just use set &, the point is to
    show the merge). Optional: skip pointers.
    """
    raise NotImplementedError("TODO(A): two-pointer postings intersection")


def intersect_many(postings: list[list[str]]) -> list[str]:
    """Intersect many lists, shortest first (lecture query optimisation).

    TODO(A): sort by length, fold with `intersect`, stop early when empty.
    """
    raise NotImplementedError("TODO(A): multi-way intersection by increasing df")


def evaluate(node: QueryNode, index: PositionalIndex, analyze: Callable[[str], list[str]],
             universe: set[str] | None = None) -> set[str]:
    """Evaluate the tree; `analyze` turns a term/phrase into index terms (same pipeline as docs).

    TODO(A): TERM -> docs containing all analysed tokens; PHRASE -> index.phrase;
    AND/OR -> intersect/union; NOT -> universe minus child; PROX -> query.proximity.within.
    """
    raise NotImplementedError("TODO(A): evaluate Boolean query tree")
