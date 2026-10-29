"""Legal issue decomposition: what must be shown, and does the evidence cover it?  [layer 2 — working]

A criminal provision is a list of ingredients ("whoever ... voluntarily causes hurt ... by means
of any instrument for stabbing ..."). A good research trail finds authority on EACH ingredient,
not ten judgments on the easiest one. For a fact question the agent:

    1. takes the statute it trusts most (after verification, agent/verify.py)
    2. splits it into ingredient clauses (drift/provisions.split_clauses; punishment-only
       clauses and illustrations dropped) and gives each its 4 most specific terms (highest
       idf in the precedent index, after the shared text pipeline)
    3. looks for evidence of each issue in the retrieved precedents: a paragraph that contains
       at least half of the issue's terms (ratio and decision zones first)
    4. for each issue with no evidence, issues one targeted sub-query (the question plus the
       issue's terms, kind "issue") and fuses the result in

Coverage = issues with evidence / issues. The memo (agent/memo.py) shows each issue, its best
evidence and what is still open. Evidence finding is lexical and deliberately simple: it is a
coverage signal for the agent's stopping rule, not a legal judgement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PUNISH_ONLY = re.compile(r"^\(?\d*\)?\s*whoever .{0,60}shall be punished|^\s*shall be punished", re.I)
_SKIP = re.compile(r"^\s*(?:illustrations?|explanation)\b", re.I)


@dataclass
class Issue:
    label: str                         # the clause, shortened, as shown to the user
    terms: list[str]                   # analysed terms that signal it
    evidence: list[tuple[str, str]] = field(default_factory=list)   # (doc_id, quote)

    @property
    def covered(self) -> bool:
        return bool(self.evidence)


def _short(text: str, n: int = 14) -> str:
    words = text.split()
    return " ".join(words[:n]) + (" …" if len(words) > n else "")


def decompose(statute_text: str, analyze, idf, max_issues: int = 5, terms_per_issue: int = 4) -> list[Issue]:
    """Ingredient issues of one provision (see module docstring)."""
    from kanoon_bridge.drift.provisions import split_clauses

    issues: list[Issue] = []
    seen: set[tuple[str, ...]] = set()
    for clause in split_clauses(statute_text):
        if _SKIP.match(clause) or (_PUNISH_ONLY.search(clause) and len(clause.split()) < 30):
            continue
        cand = [t for t in dict.fromkeys(analyze(clause))
                if not t.startswith(("sec:", "off:")) and not t.isdigit() and len(t) > 2 and idf(t) > 0]
        cand.sort(key=idf, reverse=True)
        terms = cand[:terms_per_issue]
        key = tuple(sorted(terms))
        if len(terms) < 2 or key in seen:
            continue
        seen.add(key)
        issues.append(Issue(_short(re.sub(r"^\(\w{1,4}\)\s*", "", clause)), terms))
        if len(issues) >= max_issues:
            break
    return issues


def find_evidence(issues: list[Issue], doc_ids: list[str], docs, analyze, per_issue: int = 2,
                  min_share: float = 0.5) -> float:
    """Fill each issue's evidence from the given precedents; returns coverage in [0, 1]."""
    if not issues:
        return 1.0
    from kanoon_bridge.rag._util import split_sentences

    for issue in issues:
        issue.evidence = []
        need = max(1, int(round(min_share * len(issue.terms))))
        for d in doc_ids:
            doc = docs.get(d) if docs is not None else None
            if doc is None:
                continue
            paras = sorted(doc.paragraphs, key=lambda p: p.zone not in ("ratio", "decision"))
            for p in paras:
                if len(set(issue.terms) & set(analyze(p.text))) >= need:
                    best = max(split_sentences(p.text) or [p.text],
                               key=lambda s: len(set(issue.terms) & set(analyze(s))))
                    issue.evidence.append((d, _short(best, 40)))
                    break
            if len(issue.evidence) >= per_issue:
                break
    return sum(i.covered for i in issues) / len(issues)


__all__ = ["Issue", "decompose", "find_evidence"]
