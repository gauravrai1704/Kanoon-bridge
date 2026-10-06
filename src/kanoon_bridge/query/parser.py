"""Query syntax parser.  [owner: A]

Supported syntax (Westlaw-style, plus facets):

    murder knife                      free text (ranked)
    "criminal breach of trust"        phrase
    bail AND parity NOT dowry         Boolean (AND, OR, NOT; parentheses)
    bail /5 parity                    proximity: within 5 tokens
    state:delhi  date:2025-03-01  code:bns  court:supreme_court   facet filters

parse() splits filters off, builds a tree for the Boolean/phrase/proximity part, and keeps
the free text for ranking. Terms in the tree must go through text.pipeline.analyze_text
before they hit the index (do that in query/boolean.py, not here).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

FILTER_KEYS = ("state", "date", "code", "court", "type")
_FILTER_RE = re.compile(rf"\b({'|'.join(FILTER_KEYS)}):(\S+)", re.IGNORECASE)


class Op(str, Enum):
    TERM = "term"
    PHRASE = "phrase"
    AND = "and"
    OR = "or"
    NOT = "not"
    PROX = "prox"           # children = [left, right], k = window


@dataclass
class QueryNode:
    op: Op
    value: str = ""                         # TERM / PHRASE text
    children: list["QueryNode"] = field(default_factory=list)
    k: int = 0                              # PROX window


@dataclass
class ParsedQuery:
    raw: str
    free_text: str                          # text with filters removed, for ranking
    filters: dict[str, str] = field(default_factory=dict)
    tree: QueryNode | None = None           # None if the query has no Boolean/phrase/prox syntax

    @property
    def is_boolean(self) -> bool:
        return self.tree is not None


def extract_filters(text: str) -> tuple[str, dict[str, str]]:
    """Pull `key:value` facet filters out of the query (working)."""
    filters = {m.group(1).lower(): m.group(2).lower() for m in _FILTER_RE.finditer(text)}
    return _FILTER_RE.sub("", text).strip(), filters


def has_operators(text: str) -> bool:
    """True if the text uses Boolean, phrase or proximity syntax (working)."""
    # "(" after a digit is a sub-section like 103(1), not grouping; "/" in "u/s" is not proximity.
    return bool(re.search(r'"|\bAND\b|\bOR\b|\bNOT\b|(?<![a-z])/\d+|(?<![\da-z])\(', text))


def parse(text: str) -> ParsedQuery:
    """Parse a query string.

    Working: filter extraction. TODO(A): build `tree` with a small recursive-descent parser
    (precedence NOT > PROX > AND > OR; adjacent terms default to AND inside a Boolean query).
    Until then, Boolean syntax is ignored with a warning and the query is ranked as free text.
    """
    free, filters = extract_filters(text)
    parsed = ParsedQuery(raw=text, free_text=free, filters=filters)
    if has_operators(free):
        try:
            parsed.tree = _parse_tree(free)
        except NotImplementedError:
            import logging

            logging.getLogger(__name__).warning("Boolean parser not implemented; ranking as free text")
    return parsed


def _parse_tree(text: str) -> QueryNode:
    raise NotImplementedError("TODO(A): recursive-descent Boolean/phrase/proximity parser")
