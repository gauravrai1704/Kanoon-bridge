"""Top-ranked statutes and zone paragraphs -> numbered chunks for the prompt.  [owner: Gaurav — working]

Chunk = one statute section, or one zone paragraph of a top precedent. Statutes come first
(at most half the budget), then for each top precedent its best paragraph from the ratio or
decision zone (best = most query-term overlap; falls back to any zone). Numbers are what the
LLM cites: [1], [2], ...
"""

from __future__ import annotations

from dataclasses import dataclass

from kanoon_bridge.rag._util import split_sentences, terms

PREFERRED_ZONES = ("ratio", "decision")


@dataclass
class Chunk:
    n: int                   # citation number shown to the LLM
    doc_id: str
    zone: str
    text: str
    rank: int                # rank of its document in the search result
    score: float
    title: str = ""

    def header(self) -> str:
        return f"[{self.n}] {self.title or self.doc_id} ({self.zone})"


def truncate(text: str, max_chars: int) -> str:
    """Cut at a sentence boundary when possible."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    out = ""
    for s in split_sentences(text):
        if len(out) + len(s) + 1 > max_chars:
            break
        out = f"{out} {s}".strip()
    return out or text[:max_chars].rsplit(" ", 1)[0] + " ..."


def _best_paragraph(doc, qterms: set[str]):
    paras = [p for p in doc.paragraphs if p.zone in PREFERRED_ZONES] or list(doc.paragraphs)
    if not paras:
        return None
    return max(paras, key=lambda p: (len(qterms & set(terms(p.text))), p.zone in PREFERRED_ZONES, -p.para_id))


def statute_label(doc) -> str:
    """'BNS Section 103 - Punishment for murder' when the section ref is known, else the title."""
    ref = (getattr(doc, "meta", None) or {}).get("ref") or ""
    if ":" in ref and ref.split(":", 1)[0] in ("ipc", "bns"):
        code, sec = ref.split(":", 1)
        title = doc.title if doc.title and "section" not in doc.title.lower() else ""
        return f"{code.upper()} Section {sec}" + (f" - {title}" if title else "")
    return doc.title or doc.doc_id


def make_chunks(result, docs, max_chunks: int = 8, max_chars: int = 1200,
                query_terms: list[str] | None = None) -> list[Chunk]:
    """`result`: SearchResult or AgentResult. `docs`: doc_id -> Document (index/docstore.py)."""
    if query_terms is None:
        q = getattr(result, "query", None)
        query_terms = list(q.weighted_terms()) if hasattr(q, "weighted_terms") else terms(getattr(q, "text", ""))
    qterms = set(query_terms)
    chunks: list[Chunk] = []

    for i, hit in enumerate(result.statutes[: max(1, max_chunks // 2)], start=1):
        doc = docs.get(hit.doc_id) if hasattr(docs, "get") else None
        if doc is None:
            continue
        body = "\n".join(p.text for p in doc.paragraphs if p.para_id >= 0) or doc.text   # drop the heading line
        chunks.append(Chunk(len(chunks) + 1, hit.doc_id, "statute", truncate(body, max_chars), i, hit.score,
                            statute_label(doc)))

    for i, hit in enumerate(result.precedents, start=1):
        if len(chunks) >= max_chunks:
            break
        doc = docs.get(hit.doc_id) if hasattr(docs, "get") else None
        para = _best_paragraph(doc, qterms) if doc is not None else None
        if para is None:
            continue
        chunks.append(Chunk(len(chunks) + 1, hit.doc_id, para.zone, truncate(para.text, max_chars), i, hit.score, doc.title))
    return chunks
