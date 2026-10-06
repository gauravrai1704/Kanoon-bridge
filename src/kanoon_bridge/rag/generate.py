"""Generate a short, cited answer from chunks.

Every sentence must end with exactly one citation like [2]. Keep the API key in .env
(ANTHROPIC_API_KEY or whichever provider you use; declare it in the AI-use section).
Needs: pip install -e ".[rag]"

Order of checks after generation:
    citation_check.check -> version_check.check -> (abstain.decide runs BEFORE generation)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kanoon_bridge.rag.chunker import Chunk

PROMPT = """You are helping a user understand Indian criminal law. Answer in 3-5 short sentences.
Use ONLY the numbered sources. End every sentence with one citation like [2].
If the sources do not answer the question, say so. Always name the code (IPC or BNS) with a section.
This is not legal advice.

Question: {question}
Incident date: {incident_date}   User's state: {state}

Sources:
{sources}
"""


@dataclass
class Answer:
    text: str
    sentences: list[tuple[str, int | None]] = field(default_factory=list)   # (sentence, cited chunk n)
    flags: list[str] = field(default_factory=list)                         # filled by the checks
    abstained: bool = False


def generate(question: str, chunks: list[Chunk], incident_date: str = "unknown", state: str = "unknown") -> Answer:
    """TODO: format PROMPT, call the LLM (temperature 0), split sentences, parse [n] citations."""
    raise NotImplementedError("TODO: LLM call")
