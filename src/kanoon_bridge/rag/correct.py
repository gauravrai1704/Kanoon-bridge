"""Corrective step: repair or remove the claims verification rejected.  [layer 3 v2 — working]

After rag/claim_check.py, each failing sentence is handled in the cheapest way that makes the
answer true to its sources (cf. corrective RAG, Yan et al. 2024), and every action is recorded
in `answer.corrections`, so the user sees what was changed:

    wrong code       "IPC 302" for an incident after 1 July 2024 -> its counterpart in the code
                     in force ("BNS 103"), from the crosswalk
    contradicted     a wrong quantity or a flipped negation -> the sentence is replaced by the
                     cited source's own best-matching sentence, quoted
    unsupported      no usable citation or little overlap -> the sentence is removed

With an LLM generator, `feedback_prompt` builds one regeneration request listing the problems;
the regenerated answer is verified again and only then repaired deterministically.

The corrected answer is verified again by the caller; a second failure is left flagged, never
silently dropped.
"""

from __future__ import annotations

import re

from kanoon_bridge.rag._util import split_sentences, terms
from kanoon_bridge.rag.chunker import Chunk
from kanoon_bridge.rag.generate import parse_answer
from kanoon_bridge.text.tokenize import extract_sections

_NEW = {"bns", "bnss", "bsa"}
_LABEL = {"ipc": "IPC", "bns": "BNS", "crpc": "CrPC", "bnss": "BNSS", "iea": "Evidence Act", "bsa": "BSA"}


def _swap_codes(sentence: str, code_in_force: str | None, normalizer) -> tuple[str, list[str]]:
    """Replace sections of the code not in force by their counterparts in force (never inside a
    quotation: what a court said stays as it said it)."""
    if not code_in_force or normalizer is None:
        return sentence, []
    done = []
    quoted = [(m.start(), m.end()) for m in re.finditer(r"[\"“”][^\"“”]{8,}[\"“”]", sentence)]
    for m in sorted(extract_sections(sentence), key=lambda m: -m.start):
        if any(a <= m.start < b for a, b in quoted):
            continue
        code = m.code.value
        if code not in _LABEL or (code in _NEW) == (code_in_force == "bns"):
            continue
        eq = normalizer.equivalents(f"{code}:{m.section}")
        if not eq:
            continue
        new_code, _, new_sec = eq[0].partition(":")
        repl = f"{_LABEL[new_code]} {new_sec.upper()}"
        done.append(f"{_LABEL[code]} {m.section.upper()} -> {repl} (code in force)")
        sentence = sentence[: m.start] + repl + sentence[m.end:]
    return sentence, done


def correct(answer, chunks: list[Chunk], code_in_force: str | None = None, normalizer=None, verifier=None):
    """A new Answer with failing claims repaired or removed (see module docstring). With a
    `verifier` (rag/claim_check.ClaimVerifier), a sentence whose sections were swapped into the
    code in force is kept if it now passes every check."""
    by_n = {c.n: c for c in chunks}
    failing = {cc.sentence: cc for cc in getattr(answer, "claims", []) if cc.verdict != "supported"}
    out, actions = [], []
    for sentence, cited in answer.sentences:
        body = re.sub(r"\s*\[\d+\]\s*$", "", sentence).strip()
        cc = failing.get(re.sub(r"\[\d+\]", "", sentence).strip())
        if cc is None:
            out.append(sentence)
            continue
        fixed, swaps = _swap_codes(body, code_in_force, normalizer)
        only_code = swaps and all("not in force" in r for r in cc.reasons)
        if swaps and not only_code and verifier is not None:     # did the swap cure every reason?
            again = verifier.check_sentence(f"{fixed} [{cited}]", cited, by_n, code_in_force)
            only_code = again is not None and again.verdict == "supported"
        if only_code:
            out.append(f"{fixed} [{cited}]")
            actions += [f"changed {s}" for s in swaps]
            continue
        c = by_n.get(cited)
        if cc.verdict == "contradicted" and c is not None:
            bt = set(terms(body))
            best = max(split_sentences(c.text) or [c.text], key=lambda s: len(bt & set(terms(s))))
            out.append(f"{(c.title or c.doc_id)} states: \"{best.strip().rstrip('.')}\". [{c.n}]")
            actions.append(f"replaced a contradicted sentence ({'; '.join(cc.reasons)}) by the source's own words")
        else:
            actions.append(f"removed an unsupported sentence ({'; '.join(cc.reasons)}): {body}")
    if not out:
        new = parse_answer("The retrieved sources do not support an answer to this question.", answer.generator)
    else:
        new = parse_answer(" ".join(out), answer.generator)
    new.chunks, new.notes, new.flags = answer.chunks, list(getattr(answer, "notes", [])), [
        f for f in answer.flags if not f.startswith(("unsupported", "contradicted"))]
    new.corrections = actions
    return new


FEEDBACK = """Your previous answer had these problems:
{problems}

Rewrite the answer. Keep every sentence that was fine. Fix each problem using ONLY the numbered
sources: copy numbers (years, fines, ages, periods) and section numbers exactly as the cited
source states them, use the code in force on the incident date, and cite one source per sentence.

Previous answer:
{answer}
"""


def feedback_prompt(answer) -> str:
    probs = "\n".join(f"- \"{cc.sentence}\": {'; '.join(cc.reasons)}" for cc in answer.claims if cc.verdict != "supported")
    return FEEDBACK.format(problems=probs, answer=answer.text)


__all__ = ["correct", "feedback_prompt"]
