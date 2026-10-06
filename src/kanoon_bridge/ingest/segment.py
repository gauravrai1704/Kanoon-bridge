"""Split judgments into zones and statutes into sub-sections.  [owner: A]

This is the "what is a document" decision from the lectures, and it feeds BM25F zones,
dense paragraphs and RAG chunks.

Zones (schema.ZONES): facts, arguments, ratio, decision (+ statute, other).

Strategy:
  1. if IL-PCSR gives per-paragraph rhetorical-role labels, map them to our zones with
     ROLE_TO_ZONE (check the label names in hour 0)
  2. otherwise use CUE_PHRASES on each paragraph, carrying the previous zone forward
  3. anything unclear -> "other" (low weight in configs: zones.other)

Indian judgments do not mark sections reliably (the IL-PCSR authors note this), so treat
zones as approximate and say so in the report.
"""

from __future__ import annotations

import re

from kanoon_bridge.schema import Document, Paragraph

# TODO(A): replace keys with the real IL-PCSR / IndianKanoon role labels.
ROLE_TO_ZONE: dict[str, str] = {
    "facts": "facts",
    "issue": "facts",
    "arguments_petitioner": "arguments",
    "arguments_respondent": "arguments",
    "analysis": "ratio",
    "precedent_relied": "ratio",
    "ratio": "ratio",
    "ruling_lower_court": "facts",
    "ruling_present_court": "decision",
    "statute": "ratio",
    "none": "other",
}

# First-match cue phrases for paragraphs without labels. TODO(A): tune on 20 judgments.
CUE_PHRASES: list[tuple[str, re.Pattern]] = [
    ("decision", re.compile(r"\b(appeal is (allowed|dismissed)|we (allow|dismiss)|accordingly,? (the )?(appeal|petition)|conviction is (upheld|set aside))\b", re.I)),
    ("ratio", re.compile(r"\b(we (are of the (considered )?(view|opinion)|hold that)|in our (view|opinion)|the question (that|which) arises)\b", re.I)),
    ("arguments", re.compile(r"\b(learned counsel|it (is|was) (contended|submitted|argued)|on behalf of the (appellant|respondent|state))\b", re.I)),
    ("facts", re.compile(r"\b(brief(ly)? (stated|facts)|the facts|prosecution case|fir was (lodged|registered))\b", re.I)),
]


def split_paragraphs(text: str) -> list[str]:
    """Split raw judgment text on blank lines or numbered paragraph markers (working baseline)."""
    parts = re.split(r"\n\s*\n|\n(?=\s*\d{1,3}\.\s)", text)
    return [p.strip() for p in parts if p.strip()]


def zone_from_cues(paragraph: str, previous: str = "facts") -> str:
    """Zone for an unlabelled paragraph: first matching cue, else carry the previous zone (working baseline)."""
    for zone, pattern in CUE_PHRASES:
        if pattern.search(paragraph):
            return zone
    return previous


def segment_judgment(text: str, role_labels: list[str] | None = None) -> list[Paragraph]:
    """Paragraphs with zones.

    TODO(A): when role_labels are given (one per paragraph), use ROLE_TO_ZONE; otherwise
    run zone_from_cues paragraph by paragraph. Number para_id from 0.
    """
    raise NotImplementedError("TODO(A): segment judgments into zones")


def segment_statute(doc: Document) -> Document:
    """TODO(A): split a statute's text into sub-section paragraphs, all with zone='statute'."""
    raise NotImplementedError("TODO(A): segment statutes")
