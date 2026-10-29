"""Statute verification: the agent inspects its top statute candidates before answering.  [layer 2 — working]

Retrieval ranks statutes by text match, which makes two mistakes a lawyer would not:

    wrong family     "dowry death" -> the Evidence Act's PRESUMPTION as to dowry death (IEA 113B /
                     BSA 118), not the offence (IPC 304B / BNS 80)
    too specific     "cheating" -> "Cheating by personation" (BNS 319), not "Cheating" (BNS 318)

The verifier reads each of the top-k candidates' heading, code and offence id (cheap, from the
facet index and the statute terms) and re-scores them:

    v(d) = 1 / rank(d)                                    the retrieval order, as a prior
                                                          (if none of the top k is of the routed
                                                          family, the best one from the top 20 joins)
         + w_family  * [d's code family == the routed family]
         + w_force   * [d's code is in force on the incident date]
         + w_heading * specificity(query, heading(d))     stemmed word overlap / union: an exact
                                                          heading beats a longer, more specific one
         - w_general * [d is a definitions / general-explanations section and the question is
                        about facts]                     (such sections define terms, they do not
                                                          answer "which offence")

Every change of the top candidate is written to the trace with the reason. Weights in
configs/default.yaml agent.verify_*. A statute ranked below k is never promoted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kanoon_bridge.agent.router import FAMILY_OF_CODE
from kanoon_bridge.schema import Code, ScoredDoc

STOP = {"of", "the", "and", "or", "by", "to", "in", "for", "a", "an", "with", "on", "any", "who", "is", "under",
        "section", "punishment", "offence", "offences", "what", "which", "when", "cases", "case", "law", "act", "code",
        "certain", "etc", "being", "not", "as", "at", "it", "its", "is", "be", "are"}


def _words(text: str) -> set[str]:
    import re

    from kanoon_bridge.drift.elements import _stem

    return {_stem(w) for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 2 and w not in STOP}


GENERAL = re.compile(r"^(?:definitions?|general explanations?|short title|punishments|interpretation|"
                     r"construction of references|saving|repeal and savings?)\b", re.I)


def _ref(engine, doc_id: str) -> str:
    from kanoon_bridge import present

    return present.statute_ref(engine, doc_id) or doc_id


def specificity(query: str, heading: str) -> float:
    q, h = _words(query), _words(heading)
    return len(q & h) / len(q | h) if (q and h) else 0.0


@dataclass
class StatuteVerifier:
    k: int = 5
    w_family: float = 0.5
    w_force: float = 0.5
    w_heading: float = 1.0
    w_general: float = 1.0           # penalty: a definitions / general-explanations section for a fact question

    @classmethod
    def from_config(cls, cfg) -> "StatuteVerifier":
        a = cfg.agent
        return cls(k=a.get("verify_k", 5), w_family=a.get("verify_family", 0.5), w_force=a.get("verify_force", 0.5),
                   w_heading=a.get("verify_heading", 1.0), w_general=a.get("verify_general", 1.0))

    def rerank(self, statutes: list[ScoredDoc], query, aq, family: str, engine, title_of,
               fact_question: bool = True) -> tuple[list[ScoredDoc], list[str]]:
        """Re-score the top k statutes; returns (new list, trace notes)."""
        if len(statutes) < 2:
            return statutes, []
        head, tail = statutes[: self.k], statutes[self.k:]
        # no candidate of the routed family in the head: bring in the best one from further down
        if not any(FAMILY_OF_CODE.get(_ref(engine, h.doc_id).split(":")[0]) == family for h in head):
            extra = next((h for h in tail if FAMILY_OF_CODE.get(_ref(engine, h.doc_id).split(":")[0]) == family), None)
            if extra is not None:
                tail = [h for h in tail if h is not extra]
                head = head + [extra]
        in_force = aq.code_in_force.value if aq.code_in_force in (Code.IPC, Code.BNS) else None
        new_codes = {"bns", "bnss", "bsa"}
        scored = []
        for i, h in enumerate(head, start=1):
            ref = _ref(engine, h.doc_id)
            code = ref.split(":")[0]
            fam_ok = FAMILY_OF_CODE.get(code) == family
            force_ok = in_force is None or (code in new_codes) == (in_force == "bns")
            spec = specificity(query.text, title_of(h.doc_id))
            general = fact_question and bool(GENERAL.match(title_of(h.doc_id) or ""))
            v = (1.0 / i + self.w_family * fam_ok + self.w_force * force_ok + self.w_heading * spec
                 - self.w_general * general)
            h.components = {**(h.components or {}), "verify": round(v, 4), "heading_match": round(spec, 3)}
            scored.append((v, -i, h, ref, fam_ok, force_ok, spec))
        scored.sort(key=lambda x: (round(x[0], 9), x[6], x[1]), reverse=True)   # ties: the closer heading
        notes = []
        if scored[0][2].doc_id != head[0].doc_id:
            _, _, h, ref, fam_ok, force_ok, spec = scored[0]
            old = _ref(engine, head[0].doc_id)
            why = []
            old_code = old.split(":")[0]
            if FAMILY_OF_CODE.get(old_code) != family and fam_ok:
                why.append(f"{old.upper()} is {FAMILY_OF_CODE.get(old_code, 'other')} law, the question is {family} law")
            if spec > specificity(query.text, title_of(head[0].doc_id)):
                why.append(f"\"{title_of(h.doc_id)}\" matches the question's words more exactly")
            notes.append(f"verified statutes: {ref.upper()} moved to the top over {old.upper()}"
                         + (f" ({'; '.join(why)})" if why else ""))
        out = [x[2] for x in scored] + tail
        for r, h in enumerate(out, start=1):
            h.rank = r
        return out, notes


__all__ = ["StatuteVerifier", "specificity"]
