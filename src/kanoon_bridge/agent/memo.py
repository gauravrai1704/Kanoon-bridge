"""The research memo: what the agent found, how, and what is still open.  [layer 2 — working]

A list of ranked documents is not what a lawyer hands over; a short memo is. The agent v2
fills one per question:

    question / route        what was asked and how the agent read it (agent/router.py)
    law in force            the verified statute(s), with the code in force on the incident
                            date, what changed from the old code (drift/), and the punishment
                            the text prescribes (drift/punishment.py, computed, not generated)
    issues                  the provision's ingredients, each with its best evidence from the
                            precedents, or marked open (agent/issues.py)
    coverage, searches      how much of the provision the evidence covers, and at what cost

`render()` gives plain text for the CLI; `to_dict()` feeds the web app and the RAG layer, whose
structured answer (rag/structured.py) is built from the same fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from kanoon_bridge.agent.issues import Issue


@dataclass
class LawEntry:
    ref: str                       # "bns:103"
    title: str
    in_force: bool
    change_note: str = ""          # from the drift table
    punishment: str = ""           # parsed from the statute text


@dataclass
class ResearchMemo:
    question: str
    route: str = ""
    code_in_force: str = ""
    law: list[LawEntry] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    coverage: float | None = None
    searches: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def open_issues(self) -> list[Issue]:
        return [i for i in self.issues if not i.covered]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["open_issues"] = [i.label for i in self.open_issues]
        return d

    def render(self, title_of=None) -> str:
        title_of = title_of or (lambda d: d)
        lines = [f"Question: {self.question}", f"Read as: {self.route}"]
        if self.code_in_force:
            lines.append(f"Code in force: {self.code_in_force.upper()}")
        if self.law:
            lines.append("Law:")
            for e in self.law:
                code, _, num = e.ref.partition(":")
                lines.append(f"  - {code.upper()} {num.upper()} {e.title}" + ("" if e.in_force else " (not in force on the date)"))
                if e.punishment and "no punishment" not in e.punishment:
                    lines.append(f"      punishment: {e.punishment}")
                if e.change_note:
                    lines.append(f"      {e.change_note}")
        if self.issues:
            cov = f" (coverage {self.coverage:.0%})" if self.coverage is not None else ""
            lines.append(f"Issues to establish{cov}:")
            for i in self.issues:
                mark = "covered" if i.covered else "OPEN"
                lines.append(f"  [{mark}] {i.label}")
                for d, quote in i.evidence[:1]:
                    lines.append(f"      {title_of(d)}: \"{quote}\"")
        lines.append(f"Searches run: {self.searches}")
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


__all__ = ["ResearchMemo", "LawEntry"]
