""""What changed?" for one section: the view the CLI and the web app show.  [owner: Gaurav — working]

    ex = DriftExplorer.load()
    ex.view("ipc:124a")     # JSON-friendly dict: class, carry-over, reasons, punishments,
                            # per-predecessor carry-over, and (when the bare-act text is on disk)
                            # both texts with a word-level diff
    print(ex.render("bns:106"))

The table comes from the committed CSVs, so the summary works without the raw data; the
side-by-side text and diff need make fetch (the bare-act text is not committed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kanoon_bridge.drift.model import DriftAnalyzer, DriftClass, DriftTable

CLASS_TEXT = {
    "identical": "Same text",
    "cosmetic": "Same law, reworded",
    "restructured": "Same law, sections merged or split",
    "punishment": "Same offence, different punishment",
    "substantive": "The provision itself changed",
    "new": "New in the BNS",
    "repealed": "Not carried into the BNS",
    "unverified": "Not verified (IPC text incomplete in our source)",
}


@dataclass
class DriftExplorer:
    table: DriftTable
    analyzer: DriftAnalyzer | None = None
    _tried: bool = field(default=False, repr=False)

    @classmethod
    def load(cls, table: DriftTable | None = None) -> "DriftExplorer":
        return cls(table or DriftTable.load())

    def _an(self) -> DriftAnalyzer | None:
        if self.analyzer is None and not self._tried:
            self._tried = True
            try:
                self.analyzer = DriftAnalyzer.from_sources()
            except (FileNotFoundError, OSError):
                self.analyzer = None
        return self.analyzer

    @staticmethod
    def parse_ref(ref: str) -> str:
        """'IPC 124A', 'ipc:124a', 'BNS-106', '124a ipc' -> 'ipc:124a' / 'bns:106'."""
        r = ref.strip().lower().replace("section", " ")
        m = re.search(r"\b(ipc|bns)\b[\s:._-]*(\d{1,3}[a-z]{0,2})", r) or re.search(r"(\d{1,3}[a-z]{0,2})\s*(ipc|bns)\b", r)
        if not m:
            raise ValueError(f"not an IPC or BNS section: {ref!r} (try ipc:124a or bns:106)")
        a, b = m.groups()
        return f"{a}:{b}" if a in ("ipc", "bns") else f"{b}:{a}"

    def view(self, ref: str) -> dict:
        ref = self.parse_ref(ref)
        code, _, sec = ref.partition(":")
        rep = self.table.report(ref)
        out: dict = {"ref": ref, "found": rep is not None}
        if rep is None:
            hint = self.table.inferred.get(sec) if code == "ipc" else None
            out["message"] = (f"IPC {sec.upper()} is not in the crosswalk; its text is closest to BNS {hint[0].upper()}"
                              if hint else f"{code.upper()} {sec.upper()} is not in the drift table")
            return out
        out.update({
            "bns_section": rep.bns_section, "ipc_sections": rep.ipc_sections, "class": rep.drift_class.value,
            "class_text": CLASS_TEXT.get(rep.drift_class.value, rep.drift_class.value),
            "carry_over": rep.carry_over, "reasons": rep.reasons, "flags": rep.flags,
            "punishment_old": rep.punishment_old, "punishment_new": rep.punishment_new,
            "features": {k: round(v, 3) for k, v in rep.features.items()},
            "pairs": [{"ipc": i, "bns": rep.bns_section, "carry_over": self.table.pairs[(i, rep.bns_section)].carry_over,
                       "class": self.table.pairs[(i, rep.bns_section)].drift_class.value}
                      for i in rep.ipc_sections if (i, rep.bns_section) in self.table.pairs],
        })
        if code == "ipc" and rep.bns_section:
            succ = self.table.successors(ref)
            out["successors"] = [{"bns": b, "carry_over": self.table.pairs[(sec, b)].carry_over,
                                  "class": self.table.pairs[(sec, b)].drift_class.value} for b in succ]
        an = self._an()
        if an is not None:
            from kanoon_bridge.drift.diff import token_diff

            old = " ".join(an.ipc[s].text for s in rep.ipc_sections if s in an.ipc)
            new = an.bns[rep.bns_section].text if rep.bns_section in an.bns else ""
            out["ipc_titles"] = {s: an.ipc[s].title for s in rep.ipc_sections if s in an.ipc}
            out["bns_title"] = an.bns[rep.bns_section].title if rep.bns_section in an.bns else ""
            out["old_text"], out["new_text"] = old, new
            if old and new:
                out["diff"] = [{"op": op, "old": a, "new": b} for op, a, b in token_diff(old, new, an.renumber)]
        return out

    def render(self, ref: str) -> str:
        v = self.view(ref)
        if not v["found"]:
            return v["message"]
        olds = ", ".join(f"IPC {s.upper()}" for s in v["ipc_sections"]) or "—"
        new = f"BNS {v['bns_section'].upper()}" if v["bns_section"] else "(no BNS successor)"
        lines = [f"{olds}  ->  {new}", f"  {v['class_text']}   (carry-over of IPC-era precedent: {v['carry_over']:.2f})"]
        lines += [f"  - {r}" for r in v["reasons"][1:]] if v["bns_section"] else [f"  - {r}" for r in v["reasons"]]
        if v.get("punishment_old") or v.get("punishment_new"):
            lines.append(f"  punishment  IPC: {v.get('punishment_old') or '—'}")
            lines.append(f"              BNS: {v.get('punishment_new') or '—'}")
        for s in v.get("successors", []):
            if s["bns"] != v["bns_section"]:
                lines.append(f"  also -> BNS {s['bns'].upper()} ({s['class']}, carry-over {s['carry_over']:.2f})")
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
