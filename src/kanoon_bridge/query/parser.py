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
_FILTER_RE = re.compile(
    rf"\b({'|'.join(FILTER_KEYS)}):(\S+)",
    re.IGNORECASE,
)


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
    """Pull `key:value` facet filters out of the query."""
    filters = {
        m.group(1).lower(): m.group(2).lower()
        for m in _FILTER_RE.finditer(text)
    }
    return _FILTER_RE.sub("", text).strip(), filters


def has_operators(text: str) -> bool:
    """True if the text uses Boolean, phrase or proximity syntax."""
    # "(" after a digit is a sub-section like 103(1), not grouping;
    # "/" in "u/s" is not proximity.
    return bool(
        re.search(
            r'"|\bAND\b|\bOR\b|\bNOT\b|(?<![a-z])/\d+|(?<![\da-z])\(',
            text,
        )
    )


def parse(text: str) -> ParsedQuery:
    """Parse a query string.

    Filters are removed first. If Boolean, phrase, proximity, or grouping
    syntax is present, the remaining query is parsed into a QueryNode tree.

    Operator precedence:
        NOT > PROX > AND > OR

    Adjacent terms default to AND inside a Boolean query.
    """
    free, filters = extract_filters(text)
    parsed = ParsedQuery(raw=text, free_text=free, filters=filters)

    if has_operators(free):
        parsed.tree = _parse_tree(free)

    return parsed


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<SPACE>\s+)
    |(?P<PHRASE>"(?:[^"\\]|\\.)*")
    |(?P<LPAREN>\()
    |(?P<RPAREN>\))
    |(?P<PROX>/\d+)
    |(?P<WORD>[^\s()]+)
    """,
    re.VERBOSE,
)


@dataclass
class _Token:
    kind: str
    value: str


def _tokenize_query(text: str) -> list[_Token]:
    """Convert the query string into parser tokens."""
    tokens: list[_Token] = []
    position = 0

    for match in _TOKEN_RE.finditer(text):
        if match.start() != position:
            raise ValueError(
                f"Unexpected character at position {position}: "
                f"{text[position:match.start()]!r}"
            )

        position = match.end()
        kind = match.lastgroup
        value = match.group()

        if kind == "SPACE":
            continue

        tokens.append(_Token(kind, value))

    if position != len(text):
        raise ValueError(
            f"Unexpected character at position {position}: {text[position:]!r}"
        )

    tokens.append(_Token("EOF", ""))
    return tokens


class _Parser:
    """Recursive-descent parser for Boolean/phrase/proximity queries."""

    def __init__(self, text: str):
        self.tokens = _tokenize_query(text)
        self.position = 0

    def current(self) -> _Token:
        return self.tokens[self.position]

    def advance(self) -> _Token:
        token = self.current()
        self.position += 1
        return token

    def match(self, kind: str, value: str | None = None) -> bool:
        token = self.current()

        if token.kind != kind:
            return False

        if value is not None and token.value.upper() != value.upper():
            return False

        return True

    def consume(self, kind: str, value: str | None = None) -> _Token:
        if not self.match(kind, value):
            expected = value if value is not None else kind
            actual = self.current().value or "end of query"
            raise ValueError(
                f"Expected {expected!r}, got {actual!r}"
            )

        return self.advance()

    def parse(self) -> QueryNode:
        node = self.parse_or()

        if not self.match("EOF"):
            raise ValueError(
                f"Unexpected token {self.current().value!r}"
            )

        return node

    # OR has the lowest precedence.
    def parse_or(self) -> QueryNode:
        node = self.parse_and()

        while self.match("WORD", "OR"):
            self.advance()
            right = self.parse_and()
            node = QueryNode(
                op=Op.OR,
                children=[node, right],
            )

        return node

    # AND has higher precedence than OR.
    def parse_and(self) -> QueryNode:
        node = self.parse_prox()

        while True:
            if self.match("WORD", "AND"):
                self.advance()
                right = self.parse_prox()
                node = QueryNode(
                    op=Op.AND,
                    children=[node, right],
                )
                continue

            # Adjacent expressions imply AND.
            if self._starts_expression():
                right = self.parse_prox()
                node = QueryNode(
                    op=Op.AND,
                    children=[node, right],
                )
                continue

            break

        return node

    # Proximity has higher precedence than AND.
    def parse_prox(self) -> QueryNode:
        node = self.parse_not()

        while self.match("PROX"):
            token = self.advance()

            try:
                k = int(token.value[1:])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid proximity window: {token.value!r}"
                ) from exc

            if k <= 0:
                raise ValueError(
                    "Proximity window must be greater than zero"
                )

            right = self.parse_not()

            node = QueryNode(
                op=Op.PROX,
                children=[node, right],
                k=k,
            )

        return node

    # NOT has the highest precedence.
    def parse_not(self) -> QueryNode:
        if self.match("WORD", "NOT"):
            self.advance()

            child = self.parse_not()

            return QueryNode(
                op=Op.NOT,
                children=[child],
            )

        return self.parse_primary()

    def parse_primary(self) -> QueryNode:
        # Parenthesised expression.
        if self.match("LPAREN"):
            self.advance()
            node = self.parse_or()
            self.consume("RPAREN")
            return node

        # Phrase.
        if self.match("PHRASE"):
            value = self.advance().value

            # Remove surrounding quotes.
            value = value[1:-1]

            return QueryNode(
                op=Op.PHRASE,
                value=value,
            )

        # Normal term.
        if self.match("WORD"):
            token = self.advance()

            # AND / OR / NOT cannot be standalone terms.
            if token.value.upper() in {"AND", "OR", "NOT"}:
                raise ValueError(
                    f"Unexpected operator {token.value!r}"
                )

            return QueryNode(
                op=Op.TERM,
                value=token.value,
            )

        actual = self.current().value or "end of query"
        raise ValueError(
            f"Expected term, phrase, or '(', got {actual!r}"
        )

    def _starts_expression(self) -> bool:
        """Return True when the current token can start an expression."""
        token = self.current()

        if token.kind in {"PHRASE", "LPAREN"}:
            return True

        if token.kind == "WORD":
            return token.value.upper() not in {"OR", "AND"}

        return False


def _parse_tree(text: str) -> QueryNode:
    """Build a Boolean/phrase/proximity query tree."""
    parser = _Parser(text)
    return parser.parse()