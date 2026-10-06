"""Check each generated sentence against the chunk it cites, with tf-idf cosine.  [owner: Gaurav — working]

A sentence whose cosine with its cited chunk is below `threshold`, or that cites nothing / a
number that is not a chunk, is flagged "unsupported". Pure IR: same text pipeline as the index,
idf from the precedent index (log-tf x idf, cosine).

`any_chunk=True` scores a sentence against its best-matching chunk instead: used for the
closed-book baseline, whose sentences cite nothing.
Metric for the report: supported-sentence rate, RAG vs the same LLM with no retrieval.
"""

from __future__ import annotations

from kanoon_bridge.rag._util import cosine, idf_lookup, strip_citations, terms, tfidf
from kanoon_bridge.rag.chunker import Chunk
from kanoon_bridge.rag.generate import Answer

NO_ANSWER = ("do not answer", "does not answer", "not enough", "no sources")


def check(answer: Answer, chunks: list[Chunk], idf=None, threshold: float = 0.2, any_chunk: bool = False) -> Answer:
    look = idf_lookup(idf)
    vecs = {c.n: tfidf(terms(c.text), look) for c in chunks}
    answer.support = []
    for sentence, cited in answer.sentences:
        body = strip_citations(sentence)
        if any(p in body.lower() for p in NO_ANSWER) and cited is None:
            answer.support.append(1.0)                 # an honest "the sources don't say" is fine
            continue
        v = tfidf(terms(body), look)
        if any_chunk:
            score = max((cosine(v, cv) for cv in vecs.values()), default=0.0)
        elif cited is None or cited not in vecs:
            score = 0.0
        else:
            score = cosine(v, vecs[cited])
        answer.support.append(score)
        if score < threshold:
            why = "no citation" if (cited is None and not any_chunk) else f"cosine {score:.2f}"
            answer.flags.append(f"unsupported ({why}): {body}")
    return answer
