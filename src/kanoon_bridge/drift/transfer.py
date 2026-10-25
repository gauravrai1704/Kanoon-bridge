"""Precedent transferability: does an IPC-era judgment still speak to this BNS-era question?  [owner: Gaurav — working]

Layer 1 matches an IPC-era precedent to a BNS-era question through shared offence ids. That
is the right recall decision and the wrong confidence decision: a 2015 judgment on IPC 124A
(sedition) matches a 2025 question on BNS 152 at full strength, though s.152 is a different
offence. This module scores each such match with the drift table:

    transfer(p | q) = max over (IPC section s cited by p, BNS section b the question is about)
                      of carry_over(s -> b)           (1.0 when p does not lean on the IPC at all;
                      0.0 when every IPC section p applies was dropped by the BNS)

The question is "about" BNS sections it names, plus the BNS sections of the offence ids it
was normalised to. A question is BNS-era when its incident date is on/after 1 July 2024, or
when it names a BNS section and gives no date. The most favourable reading (max) is used, so
a precedent that also cites an unchanged section is not penalised for a changed one.

Ranking (SearchOptions.drift): final(p) -= mu * (1 - transfer) * spread, where spread is the
larger of the top score's magnitude and the score range over the candidates (sign-safe for
learned scores, and not vanishing when only a few candidates are close); mu in configs.
Nothing changes for an IPC-era question, a precedent without IPC citations, or when the
table is missing.

The same scores drive the "why this result" line in present.py and the drift check of
layer-3 answers (`answer_flags`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kanoon_bridge.drift.model import DriftClass, DriftTable
from kanoon_bridge.schema import Code


@dataclass
class Transfer:
    score: float                       # 1 = the precedent's law is unchanged in the BNS
    ipc_ref: str = ""                  # the cited IPC section the score comes from
    bns_ref: str = ""                  # the BNS section it was read onto
    drift_class: DriftClass | None = None

    def reason(self, table: DriftTable | None = None) -> str:
        if not self.ipc_ref:
            return ""
        i, b = _show(self.ipc_ref), _show(self.bns_ref) if self.bns_ref else ""
        if self.drift_class == DriftClass.REPEALED:
            return f"decided under {i}, which the BNS did not re-enact: does not apply to a BNS-era offence"
        what = {DriftClass.SUBSTANTIVE: "changed the offence itself",
                DriftClass.PUNISHMENT: "changed the punishment (reasoning on guilt carries over, on sentence it may not)",
                DriftClass.RESTRUCTURED: "merged or split the section",
                DriftClass.COSMETIC: "reworded it",
                DriftClass.IDENTICAL: "kept the text",
                DriftClass.UNVERIFIED: "may have changed it (not verified)"}.get(self.drift_class, "changed it")
        return f"interprets {i}; {b} {what} — carry-over {self.score:.2f}"


def _show(ref: str) -> str:
    code, _, num = ref.partition(":")
    return f"{code.upper()} {num.upper()}"


def _base(sec: str) -> str:
    return re.sub(r"\(.*", "", sec).lower()


@dataclass
class TransferScorer:
    table: DriftTable
    normalizer: object                 # text.version_norm.VersionNormalizer
    mu: float = 0.5
    min_penalised: float = 0.95        # transfers at or above this are left alone

    @classmethod
    def load(cls, normalizer, cfg=None) -> "TransferScorer":
        from kanoon_bridge.config import load_config

        cfg = cfg or load_config()
        d = cfg.get("drift") or {}
        return cls(DriftTable.load(), normalizer, mu=d.get("mu", 0.5), min_penalised=d.get("min_penalised", 0.95))

    # ------------------------------------------------------------------ the question
    def bns_era(self, aq) -> bool:
        if aq.code_in_force == Code.BNS:
            return True
        if aq.code_in_force == Code.IPC:
            return False
        return any(s.startswith("bns:") for s in aq.sections)

    def targets(self, aq) -> list[str]:
        """BNS base sections the question is about."""
        out: list[str] = []
        offs = list(aq.offence_ids) + [t for t in aq.expanded_terms if t.startswith("off:")]
        refs = [s for s in aq.sections if s.startswith("bns:")]
        refs += [s for o in dict.fromkeys(offs) for s in self.normalizer.sections_of(o, "bns")]
        for r in refs:
            b = _base(r.partition(":")[2])
            if b and b not in out:
                out.append(b)
        return out

    # ------------------------------------------------------------------ one precedent
    def transfer(self, statutes_cited: list[str], targets: list[str]) -> Transfer:
        ipc = list(dict.fromkeys(_base(s.partition(":")[2]) for s in statutes_cited if s.startswith("ipc:")))
        if not ipc or not targets:
            return Transfer(1.0)
        best: Transfer | None = None
        for i in ipc:
            for b in targets:
                p = self.table.pairs.get((i, b))
                if p is not None and (best is None or p.carry_over > best.score):
                    best = Transfer(p.carry_over, f"ipc:{i}", f"bns:{b}", p.drift_class)
        if best is not None:
            return best
        # every IPC section the precedent applies is one the BNS dropped: its law is gone
        if all(i in self.table.repealed for i in ipc):
            return Transfer(0.0, f"ipc:{ipc[0]}", "", DriftClass.REPEALED)
        return Transfer(1.0)

    def rerank(self, final: dict[str, float], aq, metas, components: dict, depth: int = 100) -> tuple[dict[str, float], list[str]]:
        """Apply the transfer penalty to the top `depth` precedents (see module docstring)."""
        if not final or not self.bns_era(aq):
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
            if t.ipc_ref:
                components.setdefault(d, {})["transfer"] = t.score
            if t.score < self.min_penalised:
                out[d] = final[d] - self.mu * (1 - t.score) * spread
                hit.append(d)
        return out, hit

    # ------------------------------------------------------------------ layer 3
    def answer_flags(self, answer, aq, metas=None) -> list[str]:
        """Drift checks on a generated answer for a BNS-era question:
        * a sentence that cites an IPC section whose BNS successor changed
        * a sentence grounded in an IPC-era precedent whose provision changed (transfer < 0.5)"""
        flags: list[str] = []
        if not self.bns_era(aq):
            return flags
        from kanoon_bridge.text.tokenize import extract_sections

        seen: set[str] = set()
        targets = self.targets(aq)
        by_n = {c.n: c for c in getattr(answer, "chunks", [])}
        for sentence, n in answer.sentences or [(answer.text, None)]:
            for m in extract_sections(sentence):
                if m.code != Code.IPC:
                    continue
                key = _base(m.section)
                if key in seen:
                    continue
                seen.add(key)
                if key in self.table.repealed:
                    flags.append(f"law changed: IPC {key.upper()} was not re-enacted in the BNS")
                    continue
                for b in self.table.successors(f"ipc:{key}"):
                    rep = self.table.sections.get(b)
                    if rep and rep.drift_class in (DriftClass.SUBSTANTIVE, DriftClass.PUNISHMENT):
                        detail = next((r for r in rep.reasons if r.startswith("punishment")), "") \
                            if rep.drift_class == DriftClass.PUNISHMENT else rep.reasons[0].split(": ", 1)[-1]
                        flags.append(f"law changed: IPC {key.upper()} -> BNS {b.upper()} ({detail})")
            chunk = by_n.get(n)
            if chunk is not None and metas is not None and chunk.doc_id in metas and chunk.doc_id not in seen:
                seen.add(chunk.doc_id)
                t = self.transfer(metas[chunk.doc_id].statutes_cited, targets)
                if t.score < 0.5 and t.ipc_ref:
                    flags.append(f"stale precedent: [{n}] {t.reason()}")
        return flags


__all__ = ["Transfer", "TransferScorer"]
