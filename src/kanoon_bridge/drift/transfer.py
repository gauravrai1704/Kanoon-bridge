"""Precedent transferability: does an old-code judgment still speak to this new-code question?  [owner: Gaurav — working]

Layer 1 matches an IPC-era precedent to a BNS-era question through shared offence ids (and a
CrPC / Evidence Act precedent to a BNSS / BSA question through shared provision ids). That is
the right recall decision and the wrong confidence decision: a 2015 judgment on IPC 124A
(sedition) matches a 2025 question on BNS 152 at full strength, though s.152 is a different
offence. This module scores each such match with the drift tables of all three families
(IPC -> BNS, CrPC -> BNSS, Evidence Act -> BSA):

    transfer(p | q) = max over (old-code section s cited by p, new-code section b the question
                      is about, s and b in the same family) of carry_over(s -> b)
                      (1.0 when p cites no old-code section that maps to the question;
                      0.0 when every old-code section p applies was dropped)

The question is "about" the new-code sections it names or, when it names none, the new-code
sections of the offence / provision ids it was normalised to. A question is new-code-era when
its incident date is on/after 1 July 2024, or when it names a BNS / BNSS / BSA section and gives
no date (all three codes changed on the same day). The most favourable reading (max) is used,
so a precedent that also cites an unchanged section is not penalised for a changed one.

Ranking (SearchOptions.drift): final(p) -= mu * (1 - transfer) * spread, where spread is the
larger of the top score's magnitude and the score range over the candidates (sign-safe for
learned scores, and not vanishing when only a few candidates are close); mu in configs.
Nothing changes for an old-code-era question, a precedent without old-code citations, or when
no table is built.

The same scores drive the "why this result" line in present.py and the drift check of
layer-3 answers (`answer_flags`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kanoon_bridge.drift.model import DriftClass, DriftReport, DriftTable
from kanoon_bridge.drift.provisions import CODE_FAMILY, FAMILIES
from kanoon_bridge.schema import Code

NEW_CODES = tuple(f.new for f in FAMILIES.values())        # ("bns", "bnss", "bsa")


def show(ref: str) -> str:
    """'crpc:438' -> 'CrPC 438', 'iea:65b' -> 'Evidence Act 65B', 'bns:103' -> 'BNS 103'."""
    code, _, num = ref.partition(":")
    fam = CODE_FAMILY.get(code)
    label = (fam.old_label if code == fam.old else fam.new_label) if fam else code.upper()
    return f"{label} {num.upper()}"


@dataclass
class Transfer:
    score: float                       # 1 = the precedent's law is unchanged in the new code
    old_ref: str = ""                  # the cited old-code section the score comes from ("ipc:124a")
    new_ref: str = ""                  # the new-code section it was read onto ("bns:152")
    drift_class: DriftClass | None = None

    def reason(self) -> str:
        if not self.old_ref:
            return ""
        i, b = show(self.old_ref), show(self.new_ref) if self.new_ref else ""
        if self.drift_class == DriftClass.REPEALED:
            fam = CODE_FAMILY[self.old_ref.partition(":")[0]]
            return (f"decided under {i}, which the {fam.new_label} did not re-enact: does not apply on or "
                    "after 1 July 2024")
        if self.drift_class == DriftClass.SUBSTANTIVE and self.score >= 0.8:
            return f"interprets {i}; {b} keeps that text and adds new provisions — carry-over {self.score:.2f}"
        what = {DriftClass.SUBSTANTIVE: "changed the provision itself",
                DriftClass.PUNISHMENT: "changed the punishment (reasoning on guilt carries over, on sentence it may not)",
                DriftClass.RESTRUCTURED: "merged or split the section",
                DriftClass.COSMETIC: "reworded it",
                DriftClass.IDENTICAL: "kept the text",
                DriftClass.UNVERIFIED: "may have changed it (not verified)"}.get(self.drift_class, "changed it")
        return f"interprets {i}; {b} {what} — carry-over {self.score:.2f}"


def _base(sec: str) -> str:
    return re.sub(r"\(.*", "", sec).lower()


@dataclass
class TransferScorer:
    tables: dict[str, DriftTable]      # old code ("ipc" | "crpc" | "iea") -> its drift table
    normalizer: object                 # text.version_norm.VersionNormalizer
    mu: float = 0.5
    min_penalised: float = 0.85        # transfers at or above this are left alone
    _by_new: dict[str, DriftTable] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.tables, DriftTable):                 # one table: the penal family
            self.tables = {self.tables.family.old: self.tables}
        self._by_new = {t.family.new: t for t in self.tables.values()}

    @classmethod
    def load(cls, normalizer, cfg=None) -> "TransferScorer":
        """Every family whose drift table is built; FileNotFoundError when none is."""
        from kanoon_bridge.config import load_config

        cfg = cfg or load_config()
        d = cfg.get("drift") or {}
        tables = {}
        for fam in FAMILIES:
            try:
                tables[fam] = DriftTable.load(family=fam)
            except FileNotFoundError:
                continue
        if not tables:
            raise FileNotFoundError("no drift table: run scripts/13_build_drift.py")
        return cls(tables, normalizer, mu=d.get("mu", 0.5), min_penalised=d.get("min_penalised", 0.85))

    @property
    def table(self) -> DriftTable | None:
        """The penal (IPC -> BNS) table."""
        return self.tables.get("ipc")

    def table_for(self, code: str) -> DriftTable | None:
        """The table of the family a code belongs to (old or new code)."""
        return self.tables.get(code) or self._by_new.get(code)

    def report(self, ref: str) -> DriftReport | None:
        t = self.table_for(ref.partition(":")[0].lower())
        return t.report(ref) if t is not None else None

    # ------------------------------------------------------------------ the question
    def new_era(self, aq) -> bool:
        if aq.code_in_force == Code.BNS:
            return True
        if aq.code_in_force == Code.IPC:
            return False
        return any(s.partition(":")[0] in NEW_CODES for s in aq.sections)

    bns_era = new_era                  # the name used before the procedure families were added

    def targets(self, aq) -> list[str]:
        """New-code section refs the question is about ("bns:152", "bnss:482"): the ones it
        names, else those of its offence / provision ids."""
        refs = [s for s in aq.sections if s.partition(":")[0] in self._by_new]
        if not refs:
            offs = list(aq.offence_ids) + [t for t in aq.expanded_terms if t.startswith("off:")]
            refs = [s for o in dict.fromkeys(offs) for code in self._by_new
                    for s in self.normalizer.sections_of(o, code)]
        out: list[str] = []
        for r in refs:
            code, _, sec = r.partition(":")
            ref = f"{code}:{_base(sec)}"
            if _base(sec) and ref not in out:
                out.append(ref)
        return out

    # ------------------------------------------------------------------ one precedent
    def transfer(self, statutes_cited: list[str], targets: list[str]) -> Transfer:
        cited = list(dict.fromkeys(f"{c}:{_base(s)}" for c, _, s in (r.partition(":") for r in statutes_cited)
                                   if c in self.tables))
        if not cited or not targets:
            return Transfer(1.0)
        best: Transfer | None = None
        for ref in cited:
            code, _, i = ref.partition(":")
            table = self.tables[code]
            for tgt in targets:
                new_code, _, b = tgt.partition(":")
                if new_code != table.family.new:
                    continue
                p = table.pairs.get((i, b))
                if p is not None and (best is None or p.carry_over > best.score):
                    best = Transfer(p.carry_over, ref, tgt, p.drift_class)
        if best is not None:
            return best
        # every old-code section the precedent applies is one the new code dropped: its law is gone
        if all(ref.partition(":")[2] in self.tables[ref.partition(":")[0]].repealed for ref in cited):
            return Transfer(0.0, cited[0], "", DriftClass.REPEALED)
        return Transfer(1.0)

    def rerank(self, final: dict[str, float], aq, metas, components: dict, depth: int = 100) -> tuple[dict[str, float], list[str]]:
        """Apply the transfer penalty to the top `depth` precedents (see module docstring)."""
        if not final or not self.new_era(aq):
            return final, []
        targets = self.targets(aq)
        if not targets:
            return final, []
        head = sorted(final, key=final.get, reverse=True)[:depth]
        spread = max(final[head[0]] - final[head[-1]], abs(final[head[0]])) or 1.0
        out = dict(final)
        hit: list[str] = []
        for d in head:
            m = metas.get(d)
            if m is None:
                continue
            t = self.transfer(m.statutes_cited, targets)
            if t.old_ref:
                components.setdefault(d, {})["transfer"] = t.score
            if t.score < self.min_penalised:
                out[d] = final[d] - self.mu * (1 - t.score) * spread
                hit.append(d)
        return out, hit

    # ------------------------------------------------------------------ layer 3
    def answer_flags(self, answer, aq, metas=None) -> list[str]:
        """Drift checks on a generated answer for a new-code-era question:
        * a sentence that cites an old-code section whose successor changed (or that was dropped)
        * a sentence grounded in an old-code-era precedent whose provision changed (transfer < 0.5)"""
        flags: list[str] = []
        if not self.new_era(aq):
            return flags
        from kanoon_bridge.text.tokenize import extract_sections

        seen: set[str] = set()
        targets = self.targets(aq)
        by_n = {c.n: c for c in getattr(answer, "chunks", [])}
        for sentence, n in answer.sentences or [(answer.text, None)]:
            for m in extract_sections(sentence):
                table = self.tables.get(m.code.value)
                if table is None:
                    continue
                key = f"{m.code.value}:{_base(m.section)}"
                if key in seen:
                    continue
                seen.add(key)
                fam = table.family
                if _base(m.section) in table.repealed:
                    flags.append(f"law changed: {show(key)} was not re-enacted in the {fam.new_label}")
                    continue
                for b in table.successors(key):
                    rep = table.sections.get(b)
                    if rep and rep.drift_class in (DriftClass.SUBSTANTIVE, DriftClass.PUNISHMENT):
                        detail = next((r for r in rep.reasons if r.startswith("punishment")), "") \
                            if rep.drift_class == DriftClass.PUNISHMENT else rep.reasons[0].split(": ", 1)[-1]
                        flags.append(f"law changed: {show(key)} -> {show(f'{fam.new}:{b}')} ({detail})")
            chunk = by_n.get(n)
            if chunk is not None and metas is not None and chunk.doc_id in metas and chunk.doc_id not in seen:
                seen.add(chunk.doc_id)
                t = self.transfer(metas[chunk.doc_id].statutes_cited, targets)
                if t.score < 0.5 and t.old_ref:
                    flags.append(f"stale precedent: [{n}] {t.reason()}")
        return flags


__all__ = ["Transfer", "TransferScorer", "show"]
