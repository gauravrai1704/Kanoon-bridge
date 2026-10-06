"""Legal-aware tokeniser.  [owner: A — working baseline, extend as needed]

What it does
------------
1. Finds section mentions ("Section 302 IPC", "u/s 498A IPC", "s. 103(1) BNS", "302 IPC",
   "sections 302, 307 and 34 of the Indian Penal Code") and turns each into ONE token
   "sec:<code>:<section>", e.g. "sec:ipc:302". A number with no code becomes "sec:?:302"
   (resolved later by text/collision.py).
2. Lower-cases (case folding) and splits the rest into word tokens. Latin letters/digits and
   Devanagari words are both kept.

Section tokens are what version normalisation (text/version_norm.py) works on, so keeping
them intact here is essential: "302" alone would collide with every other 302.

    >>> tokenize("Convicted u/s 302 IPC and Section 34")
    ['convicted', 'sec:ipc:302', 'and', 'sec:?:34']

TODO(A): handle "r/w" (read with), section ranges ("302-304"), Act names beyond the main
codes, and roman-numeral clauses. Add each new case to tests/test_tokenize.py first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kanoon_bridge.schema import Code

SECTION_PREFIX = "sec:"

# Words that name a code, mapped to the Code they mean.
_CODE_WORDS: list[tuple[str, Code]] = [
    (r"indian\s+penal\s+code", Code.IPC),
    (r"i\.?\s?p\.?\s?c\.?", Code.IPC),
    (r"bharatiya\s+nyaya\s+sanhita", Code.BNS),
    (r"b\.?\s?n\.?\s?s\.?(?!s)", Code.BNS),
    (r"code\s+of\s+criminal\s+procedure", Code.CRPC),
    (r"cr\.?\s?p\.?\s?c\.?", Code.CRPC),
    (r"bharatiya\s+nagarik\s+suraksha\s+sanhita", Code.BNSS),
    (r"b\.?\s?n\.?\s?s\.?\s?s\.?", Code.BNSS),
]
_CODE_RE = "|".join(f"(?:{p})" for p, _ in _CODE_WORDS)

_NUM = r"\d{1,3}[a-z]?(?:\s?\(\s?\d{1,2}\s?\))?"          # 302, 498a, 103(1)
_NUM_LIST = rf"{_NUM}(?:\s*(?:,|and|&|/|or)\s*{_NUM})*"     # 302, 307 and 34

# "section 302 ipc", "u/s 302", "sections 302, 307 and 34 of the ipc", "s. 103(1) bns"
_LEAD_RE = re.compile(
    rf"\b(?:u/ss?\.?|under\s+sections?|sections?|secs?\.?|ss?\.)\s*"
    rf"(?P<nums>{_NUM_LIST})"
    rf"(?:\s*(?:of\s+(?:the\s+)?)?(?P<code>{_CODE_RE}))?",
)
# "302 ipc", "498a i.p.c."
_TRAIL_RE = re.compile(rf"\b(?P<nums>{_NUM_LIST})\s+(?:of\s+(?:the\s+)?)?(?P<code>{_CODE_RE})")

_WORD_RE = re.compile(r"sec:[a-z?]+:[0-9a-z()]+|[a-z0-9]+(?:'[a-z]+)?|[ऀ-ॿ]+")


@dataclass
class SectionMention:
    """A section number found in text, with its code if the text said which."""

    section: str                 # "302", "498a", "103(1)"
    code: Code                   # Code.UNKNOWN when the text gave no code
    start: int                   # character span in the lower-cased text
    end: int

    @property
    def token(self) -> str:
        return f"{SECTION_PREFIX}{self.code.value}:{self.section}"


def _code_from_words(words: str | None) -> Code:
    if not words:
        return Code.UNKNOWN
    for pattern, code in _CODE_WORDS:
        if re.fullmatch(pattern, words.strip()):
            return code
    return Code.UNKNOWN


def _split_nums(nums: str) -> list[str]:
    parts = re.split(r"\s*(?:,|and|&|/|or)\s*", nums)
    return [re.sub(r"\s+", "", p) for p in parts if p.strip()]


def extract_sections(text: str) -> list[SectionMention]:
    """Find every section mention in `text` (any case). Spans refer to text.lower()."""
    low = text.lower()
    found: list[SectionMention] = []
    taken: list[tuple[int, int]] = []
    for regex in (_LEAD_RE, _TRAIL_RE):
        for m in regex.finditer(low):
            if any(s < m.end() and m.start() < e for s, e in taken):
                continue
            code = _code_from_words(m.group("code"))
            for num in _split_nums(m.group("nums")):
                found.append(SectionMention(num, code, m.start(), m.end()))
            taken.append((m.start(), m.end()))
    found.sort(key=lambda s: s.start)
    return found


def tokenize(text: str, keep_sections: bool = True) -> list[str]:
    """Lower-case `text` and split into tokens; section mentions become single tokens."""
    low = text.lower()
    if not keep_sections:
        return _WORD_RE.findall(low)
    out: list[str] = []
    cursor = 0
    mentions = extract_sections(text)
    # Group mentions by span (a list like "302, 307 and 34 IPC" has several per span).
    spans: dict[tuple[int, int], list[SectionMention]] = {}
    for m in mentions:
        spans.setdefault((m.start, m.end), []).append(m)
    for (start, end), group in sorted(spans.items()):
        out.extend(_WORD_RE.findall(low[cursor:start]))
        out.extend(m.token for m in group)
        cursor = end
    out.extend(_WORD_RE.findall(low[cursor:]))
    return out


def is_section_token(token: str) -> bool:
    return token.startswith(SECTION_PREFIX)


def has_devanagari(text: str) -> bool:
    return bool(re.search(r"[ऀ-ॿ]", text))
