""""What changed?" for one section: the view the CLI and the web app show.  [owner: Gaurav — working]

    ex = DriftExplorer.load()
    ex.view("ipc:124a")     # JSON-friendly dict: class, carry-over, reasons, punishments,
                            # per-predecessor carry-over, and (when the bare-act text is on disk)
                            # both texts with a word-level diff
    ex.view("CrPC 438")     # any of the three families: IPC/BNS, CrPC/BNSS, Evidence Act/BSA
    print(ex.render("bns:106"))

The tables come from the committed CSVs, so the summary works without the raw data; the
side-by-side text and diff need make fetch (the bare-act text is not committed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kanoon_bridge.drift.model import DriftAnalyzer, DriftTable
from kanoon_bridge.drift.provisions import CODE_FAMILY, FAMILIES

CLASS_TEXT = {
    "identical": "Same text",
    "cosmetic": "Same law, reworded",
    "restructured": "Same law, sections merged or split",
    "punishment": "Same offence, different punishment",
    "substantive": "The provision itself changed",
    "new": "New in the {new}",
    "repealed": "Not carried into the {new}",
    "unverified": "Not verified (old text incomplete in our source)",
}
# how people write the multi-word codes -> our code ids
_ALIASES = [(r"indian\s+evidence\s+act", "iea"), (r"evidence\s+act", "iea"), (r"cr\.?\s?p\.?\s?c\.?", "crpc")]
_CODES = "|".join(sorted(CODE_FAMILY, key=len, reverse=True))


@dataclass
class DriftExplorer:
    tables: dict[str, DriftTable]                                   # old code -> table
    analyzers: dict[str, DriftAnalyzer | None] = field(default_factory=dict)

    @classmethod
    def load(cls, table: DriftTable | None = None) -> "DriftExplorer":
        if table is not None:
            return cls({table.family.old: table})
        tables = {}
        for fam in FAMILIES:
            try:
                tables[fam] = DriftTable.load(family=fam)
            except FileNotFoundError:
                continue
        if not tables:
            raise FileNotFoundError("no drift table: run scripts/13_build_drift.py")
        return cls(tables)

    @property
    def table(self) -> DriftTable | None:
        return self.tables.get("ipc")

    def _an(self, family: str = "ipc") -> DriftAnalyzer | None:
        if family not in self.analyzers:
            try:
                self.analyzers[family] = DriftAnalyzer.from_sources(family=family)
            except (FileNotFoundError, OSError):
                self.analyzers[family] = None
        return self.analyzers[family]

    @staticmethod
    def parse_ref(ref: str) -> str:
        """'IPC 124A', 'ipc:124a', 'BNS-106', '124a ipc', 'CrPC 438', 'section 65B Evidence Act'
        -> 'ipc:124a', 'bns:106', 'crpc:438', 'iea:65b'."""
        r = ref.strip().lower().replace("section", " ")
        for pattern, code in _ALIASES:
            r = re.sub(pattern, f" {code} ", r)
        m = re.search(rf"\b({_CODES})\b[\s:._-]*(\d{{1,3}}[a-z]{{0,2}})", r) or \
            re.search(rf"(\d{{1,3}}[a-z]{{0,2}})\s*({_CODES})\b", r)
        if not m:
            raise ValueError(f"not a section of the IPC, BNS, CrPC, BNSS, Evidence Act or BSA: {ref!r} "
                             "(try ipc:124a, bns:106, crpc:438)")
        a, b = m.groups()
        return f"{a}:{b}" if a in CODE_FAMILY else f"{b}:{a}"

    def rows(self) -> list[dict]:
        """Every section of every built family, for the web list."""
        out = []
        for fam, t in self.tables.items():
            F = t.family
            for r in [*t.sections.values(), *t.repealed.values()]:
                out.append({"family": fam, "old_label": F.old_label, "new_label": F.new_label,
                            "new": r.new_section, "old": ";".join(r.old_sections), "class": r.drift_class.value,
                            "carry_over": r.carry_over, "title": r.title,
                            "ref": f"{F.new}:{r.new_section}" if r.new_section else f"{F.old}:{r.old_sections[0]}"})
        return out

    def view(self, ref: str) -> dict:
        ref = self.parse_ref(ref)
        code, _, sec = ref.partition(":")
        fam = CODE_FAMILY[code]
        table = self.tables.get(fam.old)
        rep = table.report(ref) if table is not None else None
        out: dict = {"ref": ref, "found": rep is not None, "family": fam.old,
                     "old_label": fam.old_label, "new_label": fam.new_label}
        if rep is None:
            if table is None:
                out["message"] = f"the {fam.old_label} -> {fam.new_label} drift table is not built"
                return out
            hint = table.inferred.get(sec) if code == fam.old else None
            label = fam.old_label if code == fam.old else fam.new_label
            out["message"] = (f"{label} {sec.upper()} is not in the crosswalk; its text is closest to "
                              f"{fam.new_label} {hint[0].upper()}" if hint
                              else f"{label} {sec.upper()} is not in the drift table")
            return out
        out.update({
            "new_section": rep.new_section, "old_sections": rep.old_sections, "class": rep.drift_class.value,
            "class_text": CLASS_TEXT.get(rep.drift_class.value, rep.drift_class.value).format(new=fam.new_label),
            "carry_over": rep.carry_over, "reasons": rep.reasons, "flags": rep.flags,
            "punishment_old": rep.punishment_old, "punishment_new": rep.punishment_new, "title": rep.title,
            "features": {k: round(v, 3) for k, v in rep.features.items()},
            "pairs": [{"old": i, "new": rep.new_section, "carry_over": table.pairs[(i, rep.new_section)].carry_over,
                       "class": table.pairs[(i, rep.new_section)].drift_class.value}
                      for i in rep.old_sections if (i, rep.new_section) in table.pairs],
        })
        if code == fam.old and rep.new_section:
            out["successors"] = [{"new": b, "carry_over": table.pairs[(sec, b)].carry_over,
                                  "class": table.pairs[(sec, b)].drift_class.value} for b in table.successors(ref)]
        an = self._an(fam.old)
        if an is not None:
            from kanoon_bridge.drift.diff import token_diff

            old = " ".join(an.old_prov[s].text for s in rep.old_sections if s in an.old_prov)
            new = an.new_prov[rep.new_section].text if rep.new_section in an.new_prov else ""
            out["old_titles"] = {s: an.old_prov[s].title for s in rep.old_sections if s in an.old_prov}
            out["new_title"] = an.new_prov[rep.new_section].title if rep.new_section in an.new_prov else ""
            out["old_text"], out["new_text"] = old, new
            if old and new:
                out["diff"] = [{"op": op, "old": a, "new": b} for op, a, b in token_diff(old, new, an.renumber)]
        return out

    def render(self, ref: str) -> str:
        v = self.view(ref)
        if not v["found"]:
            return v["message"]
        O, N = v["old_label"], v["new_label"]
        olds = ", ".join(f"{O} {s.upper()}" for s in v["old_sections"]) or "—"
        new = f"{N} {v['new_section'].upper()}" if v["new_section"] else f"(no {N} successor)"
        lines = [f"{olds}  ->  {new}", f"  {v['class_text']}   (carry-over of {O}-era precedent: {v['carry_over']:.2f})"]
        lines += [f"  - {r}" for r in v["reasons"][1:]] if v["new_section"] else [f"  - {r}" for r in v["reasons"]]
        pun = (v.get("punishment_old") or "") + (v.get("punishment_new") or "")
        if pun and "no punishment" not in pun:
            lines.append(f"  punishment  {O}: {v.get('punishment_old') or '—'}")
            lines.append(f"  punishment  {N}: {v.get('punishment_new') or '—'}")
        for s in v.get("successors", []):
            if s["new"] != v["new_section"]:
                lines.append(f"  also -> {N} {s['new'].upper()} ({s['class']}, carry-over {s['carry_over']:.2f})")
        if "diff" in v:
            changed = [d for d in v["diff"] if d["op"] not in ("equal", "renumbered")][:8]
            renum = sum(d["op"] == "renumbered" for d in v["diff"])
            if renum:
                lines.append(f"  {renum} cross-reference(s) only renumbered (not counted as change)")
            if changed:
                lines.append("  wording changes (first few):")
                for d in changed:
                    a, b = _cut(d["old"]), _cut(d["new"])
                    lines.append(f"    {d['op']:7s} {a or '·'}  =>  {b or '·'}")
        return "\n".join(lines)


def _cut(s: str, n: int = 70) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


__all__ = ["DriftExplorer", "CLASS_TEXT"]
