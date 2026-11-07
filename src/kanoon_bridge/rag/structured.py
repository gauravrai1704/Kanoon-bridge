"""Structured legal answers: compute what can be computed, quote the rest.  [layer 3 v2 — working]

A generated answer can get a punishment or a section number wrong; a parser cannot hallucinate.
The structured answer fills the facts of the law from the retrieved statute text and the drift
tables, and quotes the precedents, so every sentence is either computed from a cited source or
copied from it:

    law         the top retrieved statute of the code in force on the incident date
                (the first statute chunk whose code is in force; else the top one, flagged)
    punishment  drift/punishment.extract on that statute's text            -> cites the statute
    precedent   the best-matching sentence of the top precedent chunk,
                quoted                                                     -> cites the precedent
    notes       not citations but system facts, shown apart from the answer:
                the old-code counterpart (crosswalk), what changed (drift table), and the
                transfer warning for an old-code precedent (drift/transfer.py)

Used as the offline generator ("structured") of RagPipeline v2, and as the reference an LLM
answer is checked against (rag/claim_check.py compares the LLM's quantities with the
statute's).
"""

from __future__ import annotations

import re

from kanoon_bridge.rag._util import split_sentences, terms
from kanoon_bridge.rag.chunker import Chunk
from kanoon_bridge.rag.generate import Answer, parse_answer

_NEW = {"bns", "bnss", "bsa"}


def _ref_of(chunk: Chunk, docs) -> str:
    doc = docs.get(chunk.doc_id) if docs is not None and hasattr(docs, "get") else None
    return ((getattr(doc, "meta", None) or {}).get("ref") or (chunk.doc_id if ":" in chunk.doc_id else "")).lower()


def _show(ref: str) -> str:
    from kanoon_bridge.drift.transfer import show

    return show(ref)


from kanoon_bridge.ingest.segment import _SUBSECTION_RE as _SUBSEC  # noqa: E402  (not at "sub-section (2)")


def case_name(title: str) -> str:
    """A case title safe inside one sentence: abbreviation periods dropped ("STATE OF U.P. THRU.
    SECY. HOME DEPTT." -> "STATE OF U.P THRU SECY HOME DEPTT"), so the sentence splitter cannot cut it."""
    return re.sub(r"\.(?=\s|$)", "", title or "").strip()


def punishment_by_subsection(text: str, max_parts: int = 8) -> list[str]:
    """'It is punishable with ...' per sub-section that prescribes a punishment: one sentence when
    the section has one punishment, '(2) ...; (3) ...' otherwise (never a merge of different
    sub-sections' terms into one impossible range)."""
    from kanoon_bridge.drift.punishment import extract

    parts = [p.strip() for p in _SUBSEC.split(text or "") if p.strip()]
    found = []
    for p in parts:
        pun = extract(p)
        if not pun.empty and pun.describe() not in [d for _, d in found]:
            label = re.match(r"\((\d{1,2})\)", p)
            found.append((label.group(1) if label else "", pun.describe()))
    if not found:
        return []
    if len(found) == 1:
        return [f"It is punishable with {found[0][1]}"]
    return ["It is punishable, " + "; ".join(f"under sub-section ({n}), with {d}" if n else f"with {d}"
                                             for n, d in found[:max_parts])]


def _era(chunk: Chunk, docs) -> tuple[str | None, object]:
    from kanoon_bridge.query.era import of_judgment

    doc = docs.get(chunk.doc_id) if docs is not None and hasattr(docs, "get") else None
    return (of_judgment(getattr(doc, "statutes_cited", None)) if doc is not None else None), doc


def _pick_precedent(precedents: list[Chunk], docs, code_in_force: str | None, prefer_new_era: bool, window: int = 3):
    """The precedent to quote: for a question under the new codes, the best-ranked judgment decided
    under them if one is in the top `window`; else the top precedent."""
    if prefer_new_era and code_in_force == "bns":
        for c in precedents[:window]:
            if _era(c, docs)[0] == "new":
                return c
    return precedents[0]


def era_note(chunk: Chunk, docs, code_in_force: str | None, drift, law_ref: str) -> str:
    """For a question under the new codes, a note when the quoted judgment was decided under the
    old ones: which old section it applied and whether the new code changed it (drift/transfer.py)."""
    if code_in_force != "bns":
        return ""
    era, doc = _era(chunk, docs)
    if era != "old":
        return ""
    year = f" ({doc.decision_date.year})" if getattr(doc, "decision_date", None) else ""
    head = f"the quoted judgment{year} was decided under the old codes"
    if drift is None or not law_ref:
        return head
    t = drift.transfer(doc.statutes_cited, [law_ref])
    if not t.old_ref:
        return head
    warn = "; check that its reasoning still applies" if t.score < 0.85 else ""
    return f"{head}: it {t.reason()}{warn}"


def structured(question: str, chunks: list[Chunk], docs, code_in_force: str | None = None,
               drift=None, normalizer=None, statute_text=None, prefer_new_era: bool = True) -> Answer:
    """Build the structured answer (see module docstring). `code_in_force`: "ipc" | "bns" | None;
    `drift`: drift.transfer.TransferScorer; `statute_text(doc_id)`: full statute text for the
    punishment parser (default: the chunk text)."""
    statutes = [c for c in chunks if c.zone == "statute"]
    precedents = [c for c in chunks if c.zone != "statute"]
    if not statutes and not precedents:
        return Answer(text="The retrieved sources do not answer this question.", generator="structured")
    sentences, notes = [], []
    law = None
    for c in statutes:
        ref = _ref_of(c, docs)
        code = ref.split(":")[0]
        if code_in_force is None or (code in _NEW) == (code_in_force == "bns"):
            law = c
            break
    if law is None and statutes:
        law = statutes[0]
        notes.append("none of the retrieved statutes is of the code in force on the incident date")
    if law is not None:
        ref = _ref_of(law, docs)
        label = law.title or _show(ref)
        when = {"bns": "on or after 1 July 2024", "ipc": "before 1 July 2024"}.get(code_in_force or "", "")
        name = _show(ref) if ref else ""
        heading = (law.title.split(" - ", 1)[1] if " - " in (law.title or "") else "") or label
        doc = docs.get(law.doc_id) if docs is not None and hasattr(docs, "get") else None
        if doc is not None and heading == law.title:
            from kanoon_bridge.agent.research import statute_heading

            heading = statute_heading(doc)          # IL-PCSR's bare "Section 302 of The Indian Penal Code"
        sentences.append(f"The provision that applies{(' to an incident ' + when) if when else ''} is "
                         f"{name + ', ' if name else ''}\"{heading}\". [{law.n}]")
        text = statute_text(law.doc_id) if statute_text else law.text
        for part in punishment_by_subsection(text or law.text):
            sentences.append(f"{part}. [{law.n}]")
        if normalizer is not None and ref:
            eq = normalizer.equivalents(ref)
            if eq:
                other = "old" if ref.split(":")[0] in _NEW else "new"
                notes.append(f"{_show(ref)} corresponds to {', '.join(_show(e) for e in eq)} in the {other} code")
        if drift is not None and ref:
            table = drift.table_for(ref.split(":")[0])
            rep = table.report(ref) if table is not None else None
            if rep is not None and rep.drift_class.value not in ("identical", "new") and rep.reasons:
                notes.append("what changed: " + "; ".join(rep.reasons[:3]))
    if precedents:
        qterms = set(terms(question))
        c = _pick_precedent(precedents, docs, code_in_force, prefer_new_era)
        sents = [s for s in split_sentences(c.text) if len(s.split()) >= 6] or [c.text]
        best = max(sents, key=lambda s: len(qterms & set(terms(s))))
        sentences.append(f"In {case_name(c.title or c.doc_id)}, the court said: \"{best.strip().rstrip('.')}\". [{c.n}]")
        note = era_note(c, docs, code_in_force, drift, _ref_of(law, docs) if law is not None else "")
        if note:
            notes.append(note)
    ans = parse_answer(" ".join(sentences), "structured")
    ans.notes = notes
    ans.chunks = chunks
    return ans


__all__ = ["structured"]
