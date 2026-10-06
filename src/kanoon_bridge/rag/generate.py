"""Generate a short, cited answer from chunks.  [owner: Gaurav — working]

Two generators, same output (an Answer whose sentences each carry one [n] citation):

    claude      Claude via the Anthropic API, temperature 0. Needs  pip install -e ".[rag]"  and
                ANTHROPIC_API_KEY in .env (never in git). Model: configs/default.yaml rag.model,
                or env KB_RAG_MODEL, default claude-haiku-4-5-20251001.
    extractive  offline, no LLM: for each top chunk, the sentence with the most query-term
                overlap, quoted with its citation. Used in tests, in the sample run, and when no
                key is set. (Its sentences come from the chunks, so its support rate is ~1 by
                construction - report the Claude numbers, not these.)

`closed_book=True` asks Claude WITHOUT sources: the "no retrieval" baseline for the
supported-sentence comparison in eval/agent_eval.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from kanoon_bridge.rag._util import citations, split_sentences, strip_citations, terms
from kanoon_bridge.rag.chunker import Chunk

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

SYSTEM = ("You help users understand Indian criminal law. You answer only from the numbered sources "
          "you are given and you never invent section numbers. This is not legal advice.")

PROMPT = """Answer in 3-5 short sentences.
Use ONLY the numbered sources. End every sentence with exactly one citation like [2].
If the sources do not answer the question, say so in one sentence with no citation.
Always name the code (IPC or BNS) with a section number. The Bharatiya Nyaya Sanhita (BNS)
replaced the Indian Penal Code (IPC) for offences on or after 1 July 2024: use the code in force
on the incident date, and mention the other code's equivalent section if a source gives it.

Question: {question}
Incident date: {incident_date}   User's state: {state}

Sources:
{sources}
"""

CLOSED_BOOK_PROMPT = """Answer in 3-5 short sentences about Indian criminal law. Always name the code
(IPC or BNS) with a section number. Incident date: {incident_date}. User's state: {state}.

Question: {question}
"""


@dataclass
class Answer:
    text: str
    sentences: list[tuple[str, int | None]] = field(default_factory=list)   # (sentence, cited chunk n)
    flags: list[str] = field(default_factory=list)                         # filled by the checks
    abstained: bool = False
    generator: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    support: list[float] = field(default_factory=list)                     # cosine per sentence

    @property
    def supported_rate(self) -> float:
        if not self.sentences:
            return 0.0
        bad = sum(1 for f in self.flags if f.startswith("unsupported"))
        return 1 - bad / len(self.sentences)

    @property
    def version_flags(self) -> list[str]:
        return [f for f in self.flags if f.startswith(("wrong code", "bare number"))]


def load_env(path: str | Path | None = None) -> None:
    """Read KEY=VALUE lines from .env into os.environ (python-dotenv if installed)."""
    from kanoon_bridge.config import project_path

    path = Path(path) if path else project_path(".env")
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def has_api_key() -> bool:
    load_env()
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def pick_generator(name: str | None) -> str:
    name = (name or "auto").lower()
    if name == "auto":
        return "claude" if has_api_key() else "extractive"
    return name


def format_sources(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"{c.header()}\n{c.text}" for c in chunks)


def parse_answer(text: str, generator: str = "") -> Answer:
    sentences = []
    for s in split_sentences(text):
        cites = citations(s)
        sentences.append((s, cites[0] if cites else None))
    return Answer(text=text.strip(), sentences=sentences, generator=generator)


def call_claude(prompt: str, model: str | None = None, max_tokens: int = 600, system: str = SYSTEM) -> str:
    load_env()
    try:
        from anthropic import Anthropic
    except ImportError as err:
        raise RuntimeError('Claude generator needs:  pip install -e ".[rag]"') from err
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set (put it in .env)")
    client = Anthropic()
    msg = client.messages.create(model=model or os.environ.get("KB_RAG_MODEL") or DEFAULT_MODEL,
                                 max_tokens=max_tokens, temperature=0, system=system,
                                 messages=[{"role": "user", "content": prompt}])
    return "".join(getattr(b, "text", "") for b in msg.content)


def extractive(question: str, chunks: list[Chunk], max_sentences: int = 4) -> Answer:
    """No-LLM answer: the top-ranked statute's best sentence, then the sentences with most
    query-term overlap across the other chunks (at most one per chunk, ties broken by chunk
    order), each quoted with its citation."""
    qterms = set(terms(question))
    cands = []
    for c in chunks:
        sents = [s for s in (split_sentences(c.text) or [c.text]) if len(s.split()) >= 6] or [c.text]
        best = max(sents, key=lambda s: len(qterms & set(terms(s))))
        cands.append((len(qterms & set(terms(best))), -c.n, c, best))
    # the top-ranked statute always leads (retrieval already judged it best); the rest by overlap
    lead = [x for x in cands if x[2].n == 1 and x[2].zone == "statute"]
    rest = [x for x in sorted(cands, key=lambda x: (x[0], x[1]), reverse=True) if x[0] > 0 and x not in lead]
    cands = (lead + rest)[:max_sentences]
    if not cands:
        return Answer(text="The retrieved sources do not answer this question.", generator="extractive")
    out = []
    for _, _, c, best in sorted(cands, key=lambda x: x[2].n):
        s = best.strip().rstrip(".")
        label = c.title or c.doc_id
        out.append(f"{label}: {s}. [{c.n}]" if c.zone == "statute" else f"In {label}, the court said: {s}. [{c.n}]")
    return parse_answer(" ".join(out), "extractive")


def generate(question: str, chunks: list[Chunk], incident_date: str = "unknown", state: str = "unknown",
             generator: str | None = "auto", model: str | None = None, closed_book: bool = False) -> Answer:
    gen = pick_generator(generator)
    if closed_book:
        text = call_claude(CLOSED_BOOK_PROMPT.format(question=question, incident_date=incident_date, state=state), model)
        ans = parse_answer(text, "claude-closed-book")
    elif gen == "extractive":
        ans = extractive(question, chunks)
    elif gen == "claude":
        if not chunks:
            return Answer(text="No sources were retrieved for this question.", generator="claude")
        prompt = PROMPT.format(question=question, incident_date=incident_date, state=state,
                               sources=format_sources(chunks))
        ans = parse_answer(call_claude(prompt, model), "claude")
    else:
        raise ValueError(f"unknown generator {gen!r} (auto | claude | extractive)")
    ans.chunks = chunks
    return ans


def plain(sentence: str) -> str:
    return strip_citations(sentence)
