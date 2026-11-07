"""Which era of judgments a question asks for: decided under the new codes or the old ones.  [layer 1 — working]

Offence identifiers make IPC 503 and BNS 351 one offence, which is right for "all cases on this
offence" but wrong for "cases decided under the new code" (E10: literal crosswalk expansion beat
the core 0.355 to 0.151 MAP). The era is a separate, explicit intent:

    new   judgments that cite a BNS / BNSS / BSA section
    old   judgments that cite IPC / CrPC / Evidence Act sections and no new-code section

Set by the query filter `era:new` / `era:old`, by `Query.filters["era"]`, or detected from explicit
wording. Naming a new-code section alone is NOT an era request: "cases under section 103 BNS"
asks for the offence, and its IPC-era precedents are the point (E3).
"""

from __future__ import annotations

import re

_CODE_NEW = r"(?:bns|bnss|bsa|bharatiya\s+(?:nyaya|nagarik|sakshya)\w*(?:\s+\w+)?|new\s+(?:criminal\s+)?(?:codes?|laws?|sanhitas?))"
_CODE_OLD = r"(?:ipc|crpc|cr\.?p\.?c|evidence\s+act|indian\s+penal\s+code|old\s+(?:criminal\s+)?(?:codes?|laws?))"
# words about DECISIONS, not about charges: "a case registered under IPC and IT Act" is not an era request
_DECIDED = r"(?:decided|delivered|judgments?|rulings?|decisions?|precedents?)"

NEW = [re.compile(p, re.I) for p in (
    rf"\b{_DECIDED}\s+(?:\w+\s+){{0,3}}under\s+(?:the\s+)?{_CODE_NEW}\b(?!\s*\d)(?!\s+section)",
    rf"\b(?:post|after)[-\s]+(?:the\s+)?{_CODE_NEW}\b",
    rf"\b{_CODE_NEW}[-\s]+(?:era|regime)\b",
    r"\b(?:since|after)\s+(?:1(?:st)?\s+)?july,?\s+2024\b",
    r"\bnew[-\s]+code\s+(?:era|cases|judgments)\b",
)]
OLD = [re.compile(p, re.I) for p in (
    rf"\b{_DECIDED}\s+(?:\w+\s+){{0,3}}under\s+(?:the\s+)?{_CODE_OLD}\b(?!\s*\d)(?!\s+section)",
    rf"\b(?:pre|before)[-\s]+(?:the\s+)?(?:bns|bnss|bsa|new\s+(?:criminal\s+)?(?:codes?|laws?))\b",
    rf"\b{_CODE_OLD}[-\s]+(?:era|regime)\b",
    r"\bbefore\s+(?:1(?:st)?\s+)?july,?\s+2024\b",
)]
NEW_CODES = ("bns", "bnss", "bsa")
OLD_CODES = ("ipc", "crpc", "iea")


def detect(text: str) -> str | None:
    """'new' | 'old' | None from explicit era wording (never from a bare section reference)."""
    new = any(p.search(text or "") for p in NEW)
    old = any(p.search(text or "") for p in OLD)
    return "new" if new and not old else "old" if old and not new else None


def of_judgment(statutes_cited) -> str | None:
    """The era a judgment belongs to by what it cites: 'new', 'old', or None (cites neither)."""
    codes = {r.split(":", 1)[0] for r in statutes_cited or []}
    if codes & set(NEW_CODES):
        return "new"
    if codes & set(OLD_CODES):
        return "old"
    return None


__all__ = ["detect", "of_judgment", "NEW_CODES", "OLD_CODES"]
