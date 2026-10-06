"""Split judgments into zones and statutes into sub-sections.  [owner: Gaurav — working]

This is the "what is a document" decision from the lectures; it feeds BM25F zones, dense
paragraphs and RAG chunks. Zones (schema.ZONES): facts, arguments, ratio, decision (+ statute, other).

1. IL-PCSR gives one rhetorical-role label per paragraph (the same scheme its authors' Para-GNN
   uses). ROLE_TO_ZONE maps those labels to our zones.
2. Paragraphs without a label use CUE_PHRASES, carrying the previous zone forward.

Indian judgments do not mark sections reliably (the IL-PCSR authors note this), so zones are
approximate — say so in the report.
"""

from __future__ import annotations

import re

from kanoon_bridge.schema import Document, Paragraph

# IL-PCSR rhetorical-role labels (utils/dataset.py: RR_CONSTANTS) -> our zones.
ROLE_TO_ZONE: dict[str, str] = {
    "facts": "facts",
    "issue": "facts",
    "petarg": "arguments",
    "resparg": "arguments",
    "argument by petitioner": "arguments",
    "argument by respondent": "arguments",
    "courtres": "ratio",
    "court reasoning": "ratio",
    "precedent": "ratio",
    "statute": "ratio",
    "statue": "ratio",
    "section": "ratio",
    "cdiscource": "ratio",
    "court disclosure": "ratio",
    "conclusion": "decision",
    "none": "other",
    "": "other",
}

CUE_PHRASES: list[tuple[str, re.Pattern]] = [
    ("decision", re.compile(r"\b(appeal is (allowed|dismissed)|we (allow|dismiss)|accordingly,? (the )?(appeal|petition)|conviction is (upheld|set aside)|petition (is|stands) (allowed|dismissed))\b", re.I)),
    ("ratio", re.compile(r"\b(we (are of the (considered )?(view|opinion)|hold that)|in our (view|opinion)|the question (that|which) arises|it is (well )?settled)\b", re.I)),
    ("arguments", re.compile(r"\b(learned counsel|it (is|was) (contended|submitted|argued)|on behalf of the (appellant|respondent|state|petitioner))\b", re.I)),
    ("facts", re.compile(r"\b(brief(ly)? (stated|facts)|the facts|prosecution case|fir was (lodged|registered))\b", re.I)),
]


def split_paragraphs(text: str) -> list[str]:
    """Split raw judgment text on blank lines or numbered paragraph markers."""
    parts = re.split(r"\n\s*\n|\n(?=\s*\d{1,3}\.\s)", text)
    return [p.strip() for p in parts if p.strip()]


def zone_from_cues(paragraph: str, previous: str = "facts") -> str:
    """Zone for an unlabelled paragraph: first matching cue, else carry the previous zone."""
    for zone, pattern in CUE_PHRASES:
        if pattern.search(paragraph):
            return zone
    return previous


def zone_from_role(role: str | None) -> str | None:
    """Our zone for an IL-PCSR role label, or None if the label is unknown/missing."""
    if role is None:
        return None
    return ROLE_TO_ZONE.get(str(role).strip().lower())


def segment_judgment(text: str | list[str], role_labels: list[str] | None = None) -> list[Paragraph]:
    """Paragraphs with zones.

    `text` is either raw text (split here) or IL-PCSR's list of paragraphs. With role labels
    (one per paragraph) each label decides its zone; 'None' labels and unlabelled paragraphs fall
    back to cue phrases, carrying the previous zone forward.
    """
    paras = split_paragraphs(text) if isinstance(text, str) else [p for p in text if p and p.strip()]
    labels = list(role_labels or [])
    out: list[Paragraph] = []
    previous = "facts"
    for i, para in enumerate(paras):
        zone = zone_from_role(labels[i]) if i < len(labels) else None
        if zone in (None, "other"):
            cue = zone_from_cues(para, previous)
            zone = cue if zone is None or cue != previous else zone
        out.append(Paragraph(text=para.strip(), zone=zone, para_id=i))
        if zone != "other":
            previous = zone
    return out


_SUBSECTION_RE = re.compile(r"(?=\(\d{1,2}\)\s)")


def segment_statute(doc: Document) -> Document:
    """Split each statute paragraph on '(1) ', '(2) ' markers; every piece gets zone 'statute'."""
    pieces: list[str] = []
    for p in doc.paragraphs:
        pieces += [s.strip() for s in _SUBSECTION_RE.split(p.text) if s.strip()]
    doc.paragraphs = [Paragraph(text=t, zone="statute", para_id=i) for i, t in enumerate(pieces or [doc.title])]
    return doc
