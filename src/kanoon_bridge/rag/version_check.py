"""Flag answers that cite the wrong code for the incident date, or bare colliding numbers.  [owner: Gaurav — working]

    * IPC section cited for an incident on/after 1 July 2024 (or BNS before it)
        -> "wrong code: IPC 302 cited, BNS in force on 2025-01-03; equivalent BNS 103"
    * a bare number like "302" with no code
        -> "bare number: 302 -> ipc:302 0.84 (...), bns:302 0.16 (...)"

Wiring only: text/tokenize.extract_sections, text/collision.CollisionResolver,
text/version_norm.VersionNormalizer. With normalizer=None no equivalents are suggested
(the "without the normaliser" condition of the metric).
"""

from __future__ import annotations

from datetime import date

from kanoon_bridge.rag.generate import Answer
from kanoon_bridge.schema import Code
from kanoon_bridge.text.tokenize import extract_sections, tokenize


# code -> (the code in force before 1 July 2024, the one in force after) for its family
_OLD_NEW = {Code.IPC: ("ipc", "bns"), Code.BNS: ("ipc", "bns"), Code.CRPC: ("crpc", "bnss"),
            Code.BNSS: ("crpc", "bnss"), Code.IEA: ("iea", "bsa"), Code.BSA: ("iea", "bsa")}


def check(answer: Answer, incident_date: date | None, resolver, normalizer=None) -> Answer:
    in_force = resolver.code_for_date(incident_date) if (resolver is not None and incident_date) else None
    seen: set[str] = set()
    for sentence, _ in answer.sentences or [(answer.text, None)]:
        context = tokenize(sentence, keep_sections=False)
        for m in extract_sections(sentence):
            key = f"{m.code.value}:{m.section}"
            if key in seen:
                continue
            seen.add(key)
            if m.code in _OLD_NEW:
                # the replaced/replacing code that applies on the date ("bns" -> bnss / bsa)
                want = in_force and _OLD_NEW[m.code][0 if in_force == "ipc" else 1]
                if want and m.code.value != want:
                    in_force_code = want
                    msg = (f"wrong code: {m.code.value.upper()} {m.section} cited, "
                           f"{in_force_code.upper()} in force on {incident_date}")
                    if normalizer is not None:
                        eq = [e for e in normalizer.equivalents(key) if e.startswith(in_force_code + ":")]
                        if eq:
                            msg += "; equivalent " + ", ".join(e.replace(":", " ").upper() for e in eq)
                    answer.flags.append(msg)
            elif m.code == Code.UNKNOWN and resolver is not None:
                readings = resolver.resolve(m.section, context, incident_date)
                answer.flags.append(f"bare number: {m.section} -> " + ", ".join(
                    f"{r.section_ref} {r.confidence:.2f} ({r.reason})" for r in readings))
    return answer
