"""Check each generated sentence against the chunk it cites, with tf-idf cosine.

A sentence whose cosine with its cited chunk is below `threshold` (or that cites nothing)
is flagged "unsupported". Pure IR: same tokenizer, idf from the precedent index.

Metric for the report: supported-sentence rate vs the same LLM with no retrieval.
"""

from __future__ import annotations

from kanoon_bridge.rag.chunker import Chunk
from kanoon_bridge.rag.generate import Answer


def check(answer: Answer, chunks: list[Chunk], idf: dict[str, float], threshold: float = 0.2) -> Answer:
    """TODO: tf-idf vectors via text.pipeline.analyze_text, cosine per (sentence, cited chunk),
    append 'unsupported: <sentence>' to answer.flags."""
    raise NotImplementedError("TODO: citation support check")
