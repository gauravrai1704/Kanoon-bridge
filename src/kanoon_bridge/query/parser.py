"""Query syntax parser.  [owner: A — working]

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

    @property
    def text_for_ranking(self) -> str:
        """Free text without operators: the tree's positive words, else the free text."""
        return ranking_text(self.tree) if self.tree is not None else self.free_text


def extract_filters(text: str) -> tuple[str, dict[str, str]]:
    """Pull `key:value` facet filters out of the query (working)."""
    filters = {m.group(1).lower(): m.group(2).lower() for m in _FILTER_RE.finditer(text)}
    return _FILTER_RE.sub("", text).strip(), filters


def has_operators(text: str) -> bool:
    """True if the text uses Boolean, phrase or proximity syntax (working)."""
    # "(" after a digit is a sub-section like 103(1), not grouping; "/" in "u/s" is not proximity.
    return bool(re.search(r'"|\bAND\b|\bOR\b|\bNOT\b|(?<![a-z])/\d+|(?<![\da-z])\(', text))


def parse(text: str) -> ParsedQuery:
    """Parse a query string: filters off, then a Boolean/phrase/proximity tree if the text
    uses that syntax. A malformed Boolean query (e.g. unbalanced parentheses) is ranked as
    free text, with a warning, rather than failing the search."""
    free, filters = extract_filters(text)
    parsed = ParsedQuery(raw=text, free_text=free, filters=filters)
    if has_operators(free):
        try:
            parsed.tree = _parse_tree(free)
        except ValueError as err:
            import logging

            logging.getLogger(__name__).warning("could not parse Boolean query (%s); ranking as free text", err)
    return parsed


# --------------------------------------------------------------------------- recursive descent
#
#   query   := or_expr
#   or_expr := and_expr ("OR" and_expr)*
#   and_expr:= not_expr (["AND"] not_expr)*          adjacent terms = AND
#   not_expr:= "NOT" not_expr | prox
#   prox    := atom ("/k" atom)*                     left-associative
#   atom    := WORD | "PHRASE" | "(" or_expr ")"
#
# Binding, tightest first: proximity, NOT, AND, OR. (NOT applies to the proximity group that
# follows it, so "NOT bail /5 parity" excludes documents where the two are close.)

_TOKEN_RE = re.compile(r'\s*(?:(?P<phrase>"[^"]*")|(?P<lpar>\()|(?P<rpar>\))|(?P<prox>/(?P<k>\d+))'
                       r'|(?P<sec>\d+[a-z]?(?:\(\w+\))+)|(?P<word>[^\s()"]+))', re.IGNORECASE)


def _lex(text: str) -> list[tuple[str, str]]:
    """Tokens for the parser. A section mention ("BNS 103", "Section 302 IPC") stays ONE word,
    so it reaches the index as a section/offence token rather than two loose words."""
    from kanoon_bridge.text.tokenize import extract_sections

    out, pos = [], 0
    text = text.strip()
    spans = {}
    for m in extract_sections(text):
        if '"' not in text[m.start:m.end] and not any(op in text[m.start:m.end].split() for op in ("AND", "OR", "NOT")):
            spans[m.start] = max(spans.get(m.start, 0), m.end)
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        if pos in spans and not _inside_phrase(text, pos):
            out.append(("WORD", text[pos:spans[pos]]))
            pos = spans[pos]
            continue
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(f"unexpected character at {pos}: {text[pos:pos + 10]!r}")
        pos = m.end()
        if m.group("phrase") is not None:
            out.append(("PHRASE", m.group("phrase")[1:-1].strip()))
        elif m.group("lpar"):
            out.append(("(", "("))
        elif m.group("rpar"):
            out.append((")", ")"))
        elif m.group("prox"):
            out.append(("PROX", m.group("k")))
        else:
            word = m.group("sec") or m.group("word")
            out.append((word, word) if word in ("AND", "OR", "NOT") else ("WORD", word))
    return out


def _inside_phrase(text: str, pos: int) -> bool:
    return text.count('"', 0, pos) % 2 == 1


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.toks, self.i = tokens, 0

    def peek(self) -> str | None:
        return self.toks[self.i][0] if self.i < len(self.toks) else None

    def take(self) -> tuple[str, str]:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def or_expr(self) -> QueryNode:
        left = self.and_expr()
        children = [left]
        while self.peek() == "OR":
            self.take()
            children.append(self.and_expr())
        return children[0] if len(children) == 1 else QueryNode(Op.OR, children=children)

    def and_expr(self) -> QueryNode:
        children = [self.not_expr()]
        while self.peek() not in (None, ")", "OR"):
            if self.peek() == "AND":
                self.take()
            children.append(self.not_expr())
        return children[0] if len(children) == 1 else QueryNode(Op.AND, children=children)

    def not_expr(self) -> QueryNode:
        if self.peek() == "NOT":
            self.take()
            return QueryNode(Op.NOT, children=[self.not_expr()])
        return self.prox()

    def prox(self) -> QueryNode:
        left = self.atom()
        while self.peek() == "PROX":
            k = int(self.take()[1])
            left = QueryNode(Op.PROX, children=[left, self.atom()], k=k)
        return left

    def atom(self) -> QueryNode:
        kind = self.peek()
        if kind is None:
            raise ValueError("query ends where a term was expected")
        if kind == "(":
            self.take()
            node = self.or_expr()
            if self.peek() != ")":
                raise ValueError("missing ')'")
            self.take()
            return node
        if kind == "PHRASE":
            return QueryNode(Op.PHRASE, value=self.take()[1])
        if kind == "WORD":
            return QueryNode(Op.TERM, value=self.take()[1])
        raise ValueError(f"unexpected {self.take()[1]!r}")


def _parse_tree(text: str) -> QueryNode:
    tokens = _lex(text)
    if not tokens:
        raise ValueError("empty query")
    p = _Parser(tokens)
    node = p.or_expr()
    if p.peek() is not None:
        raise ValueError(f"unexpected {p.take()[1]!r}")
    return node


def ranking_text(node: QueryNode | None, negated: bool = False) -> str:
    """The positive words of a tree (everything not under NOT), used to rank the matches."""
    if node is None:
        return ""
    if node.op in (Op.TERM, Op.PHRASE):
        return "" if negated else node.value
    if node.op == Op.NOT:
        return ranking_text(node.children[0], not negated)
    return " ".join(t for t in (ranking_text(c, negated) for c in node.children) if t)


def show(node: QueryNode) -> str:
    """Readable form for traces: AND(bail, PROX5(dowry, death))."""
    if node.op == Op.TERM:
        return node.value
    if node.op == Op.PHRASE:
        return f'"{node.value}"'
    if node.op == Op.PROX:
        return f"{show(node.children[0])} /{node.k} {show(node.children[1])}"
    if node.op == Op.NOT:
        return f"NOT {show(node.children[0])}"
    return "(" + f" {node.op.value.upper()} ".join(show(c) for c in node.children) + ")"
