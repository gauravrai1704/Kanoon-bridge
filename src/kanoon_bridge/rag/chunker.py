"""Top-ranked statutes and zone paragraphs -> numbered chunks for the prompt.

Chunk = one statute section, or one zone paragraph of a top precedent (ratio and decision
zones first). Numbers are what the LLM cites: [1], [2], ...
"""

from __future__ import annotations

from dataclasses import dataclass

from kanoon_bridge.schema import SearchResult


@dataclass
class Chunk:
    n: int                   # citation number shown to the LLM
    doc_id: str
    zone: str
    text: str
    rank: int                # rank of its document in the search result
    score: float


def make_chunks(result: SearchResult, docs: dict, max_chunks: int = 8, max_chars: int = 1200) -> list[Chunk]:
    """TODO: top statutes first, then the ratio/decision paragraphs of top precedents;
    truncate each to max_chars at a sentence boundary. `docs`: doc_id -> Document."""
    raise NotImplementedError("TODO: build RAG chunks")
