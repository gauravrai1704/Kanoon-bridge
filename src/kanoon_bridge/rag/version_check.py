"""Flag answers that cite the wrong code for the incident date, or bare colliding numbers.

Novel check (no generic RAG checker does this):
  * IPC section cited for an incident on/after 1 July 2024 (or BNS before it) -> flag
  * a bare number like "302" with no code -> flag, and show both readings

Reuses text/tokenize.extract_sections, text/collision.CollisionResolver and
text/version_norm.VersionNormalizer — no new logic, just wiring.
Metric: version errors per answer, with vs without the normaliser.
"""

from __future__ import annotations

from datetime import date

from kanoon_bridge.rag.generate import Answer


def check(answer: Answer, incident_date: date | None, resolver, normalizer) -> Answer:
    """TODO: extract sections from answer.text; compare each code with resolver.code_for_date;
    for bare numbers add both readings; suggest the equivalent via normalizer.equivalents."""
    raise NotImplementedError("TODO: version-validity check")
