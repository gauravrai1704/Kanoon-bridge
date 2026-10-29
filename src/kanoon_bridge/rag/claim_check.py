"""Claim-level verification of an answer against the sources it cites.  [layer 3 v2 — working]

v1 checked each sentence with one number: the tf-idf cosine between the sentence and its cited
chunk. A sentence that copies the statute's words but changes "seven years" to "three years",
or "shall" to "shall not", or "BNS 103" to "BNS 101", keeps a high cosine and passes. Legal
answers fail exactly there, so v2 checks each cited sentence on five axes:

    support     tf-idf cosine with the cited chunk >= threshold (v1's check; skipped
                when the claim's own words have no content term left, e.g.
                "It is punishable with fine", which the quantity check covers)    -> unsupported
    quantities  every term, fine amount, age, period or count the sentence states
                must also be stated by the cited chunk                            -> contradicted
    sections    every section it names must be the cited statute itself, be named
                in the cited text, or be the cited statute's counterpart in the
                other code (crosswalk)                                            -> unsupported
    negation    a modal negation ("shall not", "is not", "not punishable", "no person
                shall") the source sentence lacks; or a dropped one, when the claim
                restates that source sentence (>= 60% of its terms)              -> contradicted
    code        a section of a code not in force on the incident date              -> contradicted
                (rag/version_check.py's rule, per claim)

Quotations are treated apart: a quoted passage ('the court said: "..."') must appear verbatim
(whitespace and case aside) in the cited source, and is then exempt from the other checks: a
2015 judgment rightly says "Section 302 IPC", and a quote is never "corrected". The checks run on
the words outside the quotes.

Statute claims are checked against the cited statute's FULL text and heading (`source_text`),
not the truncated chunk.

Verdicts: supported | contradicted | unsupported, each with reasons.

Sentences that say the sources do not answer, and the answer's `notes` (system facts from the
crosswalk and drift tables), are not claims and are not checked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kanoon_bridge.drift.punishment import _NUM, words_to_number
from kanoon_bridge.rag._util import cosine, idf_lookup, split_sentences, strip_citations, terms, tfidf
from kanoon_bridge.rag.chunker import Chunk
from kanoon_bridge.text.tokenize import extract_sections

NO_ANSWER = ("do not answer", "does not answer", "not enough", "no sources")
# modal negation only: an incidental "not" deep in a long statutory sentence is not a polarity
_NEG = re.compile(r"\b(?:shall|is|are|was|will|may|be)\s+not\b|\bcannot\b|\bnot\s+(?:be\s+)?punish\w*|"
                  r"\bno\s+(?:person|court|police officer|offence|one)\b|\bnever\b", re.I)
_QTY = re.compile(rf"(?:₹|rs\.?|inr)\s*(?P<money>\d[\d,]*)|(?P<num>\d[\d,]*(?:\.\d+)?|{_NUM})\s+"
                  rf"(?P<unit>years?|months?|days?|hours?|rupees|lakh|lakhs|crore|crores|persons|times)\b", re.I)
_UNIT = {"year": "years", "years": "years", "month": "months", "months": "months", "day": "days", "days": "days",
         "hour": "hours", "hours": "hours", "rupees": "rupees", "lakh": "rupees", "lakhs": "rupees", "crore": "rupees",
         "crores": "rupees", "persons": "persons", "times": "times"}
_SCALE = {"lakh": 100_000, "lakhs": 100_000, "crore": 10_000_000, "crores": 10_000_000}
_NEW = {"bns", "bnss", "bsa"}
_QUOTE = re.compile(r"[\"“”]([^\"“”]{8,})[\"“”]")


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def quantities(text: str) -> set[tuple[str, float]]:
    """{('years', 7.0), ('rupees', 10000.0), ...} stated in a text (words and digits alike)."""
    out: set[tuple[str, float]] = set()
    for m in _QTY.finditer(text or ""):
        if m.group("money"):
            n = words_to_number(m.group("money"))
            if n is not None:
                out.add(("rupees", n))
            continue
        n = words_to_number(m.group("num"))
        unit = m.group("unit").lower()
        if n is None:
            continue
        out.add((_UNIT[unit], n * _SCALE.get(unit, 1)))
    return out


def _negated(text: str) -> bool:
    return bool(_NEG.search(text or ""))


@dataclass
class ClaimCheck:
    sentence: str
    cited: int | None
    verdict: str = "supported"              # supported | contradicted | unsupported
    reasons: list[str] = field(default_factory=list)
    support: float = 0.0

    def fail(self, verdict: str, reason: str) -> None:
        if self.verdict != "contradicted":   # contradicted outranks unsupported
            self.verdict = verdict
        self.reasons.append(reason)


@dataclass
class ClaimVerifier:
    idf: object = None                      # dict / callable term -> idf (precedent index)
    threshold: float = 0.2
    normalizer: object = None               # VersionNormalizer, for counterpart sections
    docs: object = None                     # doc_id -> Document, for the cited statute's own ref and text

    def source_text(self, c: Chunk) -> str:
        """The cited source as the checks see it: heading + full text for a statute, the chunk otherwise."""
        doc = self.docs.get(c.doc_id) if self.docs is not None and hasattr(self.docs, "get") else None
        if c.zone == "statute" and doc is not None:
            body = " ".join(p.text for p in doc.paragraphs if p.para_id >= 0)
            return f"{c.title}. {body}"
        return f"{c.title}. {c.text}" if c.zone == "statute" else c.text

    def _chunk_ref(self, c: Chunk) -> str:
        doc = self.docs.get(c.doc_id) if self.docs is not None and hasattr(self.docs, "get") else None
        return ((getattr(doc, "meta", None) or {}).get("ref") or "").lower()

    def check_sentence(self, sentence: str, cited: int | None, chunks: dict[int, Chunk],
                       code_in_force: str | None = None) -> ClaimCheck | None:
        body = strip_citations(sentence)
        if any(p in body.lower() for p in NO_ANSWER) and cited is None:
            return None
        cc = ClaimCheck(body, cited)
        c = chunks.get(cited) if cited is not None else None
        if c is None:
            cc.fail("unsupported", "no citation" if cited is None else f"cites [{cited}], which is not a source")
            return cc
        src = self.source_text(c)
        quotes = [m.group(1) for m in _QUOTE.finditer(body)]
        # an unquoted verbatim excerpt ("BNS Section 59 - ...: Whoever, being ... …") counts as a quote
        if not quotes and ": " in body:
            excerpt = body.split(": ", 1)[1].rstrip(" .…").rstrip("…").strip()
            if len(excerpt.split()) >= 6 and _norm(excerpt) in _norm(src):
                quotes = [excerpt]
                body = body.replace(excerpt, f"\"{excerpt}\"", 1)
        for qt in quotes:
            if _norm(qt) not in _norm(src):
                cc.fail("contradicted", f"the quotation is not in [{cited}]")
        own_words = _QUOTE.sub(" ", body) if quotes else body
        look = idf_lookup(self.idf)
        own_vec = tfidf(terms(own_words), look)
        no_content = not any(w > 0 for w in own_vec.values())   # only stop words / zero-idf terms
        cc.support = 1.0 if (quotes and cc.verdict == "supported") or no_content else \
            cosine(tfidf(terms(body), look), tfidf(terms(src), look))
        if cc.support < self.threshold:
            cc.fail("unsupported", f"little overlap with [{cited}] (cosine {cc.support:.2f})")
        body = own_words                     # the remaining checks look at the answer's own words
        # quantities: every one the claim states must be in the source
        src_q = quantities(src)
        for unit, n in sorted(quantities(body)):
            if (unit, n) not in src_q:
                same_unit = sorted(v for u, v in src_q if u == unit)
                shown = f"{n:g} {unit}"
                cc.fail("contradicted", f"says {shown}; [{cited}] states " +
                        (", ".join(f"{v:g} {unit}" for v in same_unit) if same_unit else f"no {unit}"))
        # sections: the cited statute itself, a section its text names, or its counterpart
        own = self._chunk_ref(c)
        allowed = {own} if own else set()
        allowed |= {f"{m.code.value}:{m.section}".lower() for m in extract_sections(src)}
        if own and self.normalizer is not None:
            allowed |= set(self.normalizer.equivalents(own))
        allowed_nums = {a.split(":")[1].split("(")[0] for a in allowed if ":" in a}
        for m in extract_sections(body):
            ref = f"{m.code.value}:{m.section}".lower()
            base = f"{m.code.value}:{m.section.split('(')[0]}".lower()
            if ref in allowed or base in allowed or (m.code.value == "?" and m.section.split("(")[0] in allowed_nums):
                pass
            else:
                cc.fail("unsupported", f"names {m.code.value.upper()} {m.section}, which [{cited}] does not")
            if code_in_force and m.code.value in ("ipc", "bns", "crpc", "bnss", "iea", "bsa"):
                if (m.code.value in _NEW) != (code_in_force == "bns"):
                    cc.fail("contradicted", f"{m.code.value.upper()} {m.section} is not in force on the incident date")
        # negation: polarity of the claim vs the most similar sentence of the source
        sents = split_sentences(src) or [src]
        bt = set(terms(body))
        best = max(sents, key=lambda s: len(bt & set(terms(s))))
        shared = bt & set(terms(best))
        if _negated(body) and not _negated(best) and len(shared) >= 3:
            cc.fail("contradicted", f"negates [{cited}]: \"{best[:90]}\"")
        elif not _negated(body) and _negated(best) and len(shared) >= 0.6 * max(1, len(set(terms(best)))):
            # dropping a negation only counts when the claim restates that very sentence
            cc.fail("contradicted", f"drops the negation of [{cited}]: \"{best[:90]}\"")
        return cc

    def check(self, answer, chunks: list[Chunk], code_in_force: str | None = None) -> list[ClaimCheck]:
        """ClaimCheck per checkable sentence; also writes v1-compatible flags and support onto `answer`."""
        by_n = {c.n: c for c in chunks}
        out: list[ClaimCheck] = []
        answer.support = []
        for sentence, cited in answer.sentences:
            cc = self.check_sentence(sentence, cited, by_n, code_in_force)
            if cc is None:
                answer.support.append(1.0)
                continue
            answer.support.append(cc.support)
            out.append(cc)
            if cc.verdict != "supported":
                answer.flags.append(f"{'unsupported' if cc.verdict == 'unsupported' else 'contradicted'} "
                                    f"({'; '.join(cc.reasons)}): {cc.sentence}")
        answer.claims = out
        return out


__all__ = ["ClaimCheck", "ClaimVerifier", "quantities"]
