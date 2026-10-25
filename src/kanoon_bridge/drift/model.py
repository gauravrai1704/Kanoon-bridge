"""Drift reports: how far each BNS section moved from its IPC predecessor(s).  [owner: Gaurav — working]

For every BNS section (and every IPC section left without a successor) the analyser builds
a drift vector, a class and a carry-over score:

    drift vector   retention, novelty, edit similarity, clauses added / lost, mental-state
                   words added / removed, thresholds changed, punishment fields changed (and
                   whether harsher), content-word Jaccard, merge / split structure, title
                   similarity, length ratio, whether the old text looks incomplete
    class          NEW | IDENTICAL | COSMETIC | RESTRUCTURED | PUNISHMENT | SUBSTANTIVE
                   (and REPEALED for an IPC section with no successor, UNVERIFIED when the old
                   text in our source is too incomplete to judge). Priority: SUBSTANTIVE >
                   PUNISHMENT > RESTRUCTURED > COSMETIC > IDENTICAL.
    carry-over     c(ipc s -> bns b) in [0, 1]: how safely a precedent that interpreted s can be
                   read as interpreting b. It is computed per (IPC, BNS) pair from the share of
                   s that survives in b, discounted for new material, a changed mental state,
                   changed thresholds and (for the sentencing part) a changed punishment.
                   `reworded` is the novelty inside clauses that have an IPC counterpart (the old
                   law rewritten); `added` is the share of the BNS text in clauses that have none
                   (new law next to the old, e.g. BNS 103(2) next to murder). Rewording weighs
                   more: it changes what the precedent read; an addition only widens the section.

        c_guilt    = retention(s in b) * (1 - w_nov * reworded) * (1 - w_add * added) * m_mens_rea * m_threshold
        c_sentence = c_guilt * m_punishment
        c          = (1 - w_sent) * c_guilt + w_sent * c_sentence

                   Weights live in configs/default.yaml `drift:`. They were set by hand from
                   legal reasoning, NOT fitted to the gold labels (gold.py), so the evaluation
                   of the score against those labels is a fair test.

Group comparison (all predecessors of a BNS section together) gives the section's class;
pairwise comparison (one cited IPC section against its successor) gives the carry-over that
transfer.py uses, because a precedent cites one IPC section, not the merged group.

    table = DriftTable.build()                    # needs IPC + BNS text (make fetch)
    table.report("bns:152").drift_class           # DriftClass.SUBSTANTIVE
    table.carry_over("ipc:124a", "bns:152")       # ~0.3
    table.save_csv(); DriftTable.load()           # data/crosswalk/drift_*.csv (committed)
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from kanoon_bridge.config import Config, load_config, project_path
from kanoon_bridge.drift import diff as dfm
from kanoon_bridge.drift import elements as elm
from kanoon_bridge.drift import punishment as pun
from kanoon_bridge.drift.provisions import Provision, renumber_references, split_clauses


class DriftClass(str, Enum):
    NEW = "new"
    IDENTICAL = "identical"
    COSMETIC = "cosmetic"
    RESTRUCTURED = "restructured"
    PUNISHMENT = "punishment"
    SUBSTANTIVE = "substantive"
    REPEALED = "repealed"
    UNVERIFIED = "unverified"

    @property
    def changed(self) -> bool:
        """Did the law change in a way a lawyer must check (not just wording or numbering)?"""
        return self in (DriftClass.PUNISHMENT, DriftClass.SUBSTANTIVE, DriftClass.REPEALED, DriftClass.NEW)


DEFAULTS = {
    "match_min": 0.3,             # clause containment below this = clause added / lost
    "novelty_substantive": 0.25,  # novelty at or above this = substantive
    "retention_substantive": 0.7, # retention at or below this = substantive
    "identical_edit": 0.97,       # edit similarity at or above this (and nothing else changed) = identical
    "w_novelty": 0.5,             # c_guilt discount per unit of rewording inside matched clauses
    "w_added": 0.15,              # c_guilt discount per unit of text in added clauses
    "m_mens_rea": 0.6,            # c_guilt multiplier when the mental state changed
    "m_threshold": 0.7,           # c_guilt multiplier when an age / period / count threshold changed
    "m_punishment": 0.5,          # c_sentence multiplier when the punishment changed
    "w_sentence": 0.15,           # share of the sentencing part in the carry-over
    "inferred_min": 0.5,          # an unmapped IPC section this contained in a BNS section is not repealed
    "added_min": 1,               # this many added clauses = substantive
    "incomplete_words": 15,       # an old text shorter than this is treated as incomplete
    "prior": {"same": 0.95, "merge": 0.85, "split": 0.8, "modified": 0.6},   # carry-over when unverifiable
}

FEATURES = ["retention", "novelty", "novelty_matched", "added_share", "edit_similarity", "added_clauses", "lost_clauses", "mens_rea_changed",
            "threshold_changed", "punishment_changed", "punishment_harsher", "content_jaccard", "merge",
            "split", "title_similarity", "log_length_ratio", "old_incomplete"]


@dataclass
class DriftReport:
    bns_section: str                         # "" for a repealed IPC section
    ipc_sections: list[str]
    drift_class: DriftClass
    carry_over: float                        # group-level (all predecessors together)
    features: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    punishment_old: str = ""
    punishment_new: str = ""
    added_clauses: list[str] = field(default_factory=list)
    lost_clauses: list[str] = field(default_factory=list)
    title: str = ""

    @property
    def ref(self) -> str:
        return f"bns:{self.bns_section}" if self.bns_section else f"ipc:{self.ipc_sections[0]}"


@dataclass
class PairDrift:
    ipc_section: str
    bns_section: str
    retention: float
    carry_over: float
    drift_class: DriftClass


_PRE_REPEALED = re.compile(r"^\s*(?:rep\.|repealed|omitted)\b|\brepealed by\b", re.I)


def _base(section: str) -> str:
    return section.split("(", 1)[0].strip().lower()


def _no_illustrations(clauses: list[str]) -> list[str]:
    """Illustrations explain the law, they are not the law; the IPC source often omits them."""
    return [c for c in clauses if not re.match(r"illustrations?\b", c, re.I)] or clauses


def _added_share(al: dfm.Alignment) -> float:
    """Share of the new text (in words) that sits in clauses with no old counterpart."""
    added = sum(len(dfm.normalise(c)) for c in al.added_clauses)
    return min(1.0, added / al.new_words) if al.new_words else 0.0


def _title_similarity(a: str, b: str) -> float:
    ta, tb = set(dfm.normalise(a)) - elm.STOP, set(dfm.normalise(b)) - elm.STOP
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 1.0


@dataclass
class DriftAnalyzer:
    ipc: dict[str, Provision]
    bns: dict[str, Provision]
    predecessors: dict[str, list[str]]       # bns base section -> ipc base sections
    successors: dict[str, list[str]]         # ipc base section -> bns base sections
    relation: dict[tuple[str, str], str]     # (ipc, bns) -> crosswalk relation
    notes: dict[tuple[str, str], str] = field(default_factory=dict)   # (ipc, bns) -> crosswalk note
    params: dict = field(default_factory=lambda: dict(DEFAULTS))
    renumber: dict[str, str] = field(default_factory=dict)
    renumbering: bool = True                 # ablation switch: renumber IPC cross-references
    use_punishment: bool = True              # ablation switches for the evaluation
    use_elements: bool = True
    clause_alignment: bool = True            # False = each text is one clause (ablation)
    inferred: dict[str, tuple[str, float]] = field(default_factory=dict)   # filled by repealed()

    @classmethod
    def from_sources(cls, cfg: Config | None = None, ipc: dict | None = None, bns: dict | None = None,
                     normalizer=None) -> "DriftAnalyzer":
        from kanoon_bridge.drift.provisions import load_bns, load_ipc
        from kanoon_bridge.text.version_norm import VersionNormalizer

        cfg = cfg or load_config()
        ipc = ipc if ipc is not None else load_ipc(cfg)
        bns = bns if bns is not None else load_bns(cfg)
        norm = normalizer or VersionNormalizer.load(cfg)
        pred: dict[str, list[str]] = defaultdict(list)
        succ: dict[str, list[str]] = defaultdict(list)
        rel: dict[tuple[str, str], str] = {}
        notes: dict[tuple[str, str], str] = {}
        for r in norm.crosswalk:
            i, b = _base(r.ipc_section), _base(r.bns_section)
            if not i or not b:
                continue
            if i not in pred[b]:
                pred[b].append(i)
            if b not in succ[i]:
                succ[i].append(b)
            rel.setdefault((i, b), r.relation)
            if r.note:
                notes[(i, b)] = r.note.strip().lower()
        params = dict(DEFAULTS)
        params.update({k: v for k, v in (cfg.get("drift") or {}).items() if k in DEFAULTS})
        an = cls(ipc=ipc, bns=bns, predecessors=dict(pred), successors=dict(succ), relation=rel, params=params, notes=notes)
        an.renumber = {i: s[0] for i, s in succ.items() if len(s) == 1}
        return an

    # ------------------------------------------------------------------ helpers
    def _old_text(self, sections: list[str]) -> tuple[str, list[str], str]:
        texts, clauses, titles = [], [], []
        for s in sections:
            p = self.ipc.get(s)
            if p is None:
                continue
            t = renumber_references(p.text, self.renumber) if self.renumbering else p.text
            texts.append(t)
            clauses += split_clauses(t)
            titles.append(p.title)
        return " ".join(texts), clauses, " ".join(titles)

    def _incomplete(self, old_text: str, new_text: str) -> bool:
        n_old, n_new = len(old_text.split()), len(new_text.split())
        if n_old < self.params["incomplete_words"]:
            return True
        return not old_text.rstrip().endswith((".", ";", ":", "—", '"', "”", ")")) and n_new > 2 * n_old

    def compare(self, old_sections: list[str], new: Provision) -> tuple[dict, dict]:
        """Drift vector between the IPC sections (together) and one BNS provision.
        Returns (features, detail) where detail keeps the readable parts."""
        p = self.params
        old_text, old_clauses, old_title = self._old_text(old_sections)
        if self.clause_alignment:
            old_cl, new_cl = _no_illustrations(old_clauses), _no_illustrations(new.clauses or split_clauses(new.text))
        else:
            old_cl, new_cl = [old_text], [new.text]
        al = dfm.align(old_cl, new_cl, match_min=p["match_min"])
        po, pn = pun.extract(old_text), pun.extract(new.text)
        pd = pun.compare(po, pn) if self.use_punishment else pun.PunishmentDelta()
        ed = elm.compare(old_text, new.text) if self.use_elements else elm.ElementDelta()
        incomplete = self._incomplete(old_text, new.text)
        succ_counts = [len(self.successors.get(s, [])) for s in old_sections]
        f = {
            "retention": al.retention,
            "novelty": al.novelty,
            "novelty_matched": al.novelty_matched,
            "added_share": _added_share(al),
            "edit_similarity": al.edit_similarity,
            "added_clauses": float(len(al.added_clauses)),
            "lost_clauses": float(len(al.lost_clauses)),
            "mens_rea_changed": float(ed.mens_rea_changed),
            "threshold_changed": float(ed.threshold_changed),
            "punishment_changed": float(pd.any),
            "punishment_harsher": float(bool(pd.harsher)),
            "content_jaccard": ed.content_jaccard,
            "merge": float(len(old_sections) > 1),
            "split": float(any(c > 1 for c in succ_counts)),
            "title_similarity": _title_similarity(old_title, new.title),
            "log_length_ratio": math.log((al.new_words + 1) / (al.old_words + 1)),
            "old_incomplete": float(incomplete),
        }
        detail = {"alignment": al, "punishment_delta": pd, "elements": ed, "punishment_old": po,
                  "punishment_new": pn, "old_text": old_text}
        return f, detail

    def classify(self, f: dict) -> DriftClass:
        p = self.params
        if f["old_incomplete"]:
            if f["punishment_changed"]:
                return DriftClass.PUNISHMENT
            return DriftClass.UNVERIFIED
        substantive = (f["mens_rea_changed"] or f["threshold_changed"] or f["added_clauses"] >= p["added_min"]
                       or (f["lost_clauses"] >= 1 and not f["split"])
                       or f["novelty"] >= p["novelty_substantive"]
                       or (f["retention"] <= p["retention_substantive"] and not f["split"]))
        if substantive:
            return DriftClass.SUBSTANTIVE
        if f["punishment_changed"]:
            return DriftClass.PUNISHMENT
        if f["merge"] or f["split"]:
            return DriftClass.RESTRUCTURED
        if f["edit_similarity"] >= p["identical_edit"]:
            return DriftClass.IDENTICAL
        return DriftClass.COSMETIC

    def carry_over(self, f: dict, relation: str = "same") -> float:
        p = self.params
        if f["old_incomplete"]:
            base = p["prior"].get(relation, 0.8)
            return round(base * (p["m_punishment"] if f["punishment_changed"] else 1.0) ** p["w_sentence"], 4)
        c_guilt = f["retention"] * (1 - p["w_novelty"] * f["novelty_matched"]) * (1 - p["w_added"] * f["added_share"])
        if f["mens_rea_changed"]:
            c_guilt *= p["m_mens_rea"]
        if f["threshold_changed"]:
            c_guilt *= p["m_threshold"]
        c_sent = c_guilt * (p["m_punishment"] if f["punishment_changed"] else 1.0)
        return round(max(0.0, min(1.0, (1 - p["w_sentence"]) * c_guilt + p["w_sentence"] * c_sent)), 4)

    # ------------------------------------------------------------------ reports
    def report(self, bns_section: str) -> DriftReport:
        b = _base(bns_section)
        new = self.bns.get(b)
        if new is None:
            raise KeyError(f"BNS {b} not in the source")
        olds = [s for s in self.predecessors.get(b, []) if s in self.ipc]
        if not olds:
            return DriftReport(b, [], DriftClass.NEW, 0.0, title=new.title,
                               punishment_new=pun.extract(new.text).describe(),
                               reasons=["new in the BNS: no IPC provision corresponds, so no IPC-era precedent applies directly"])
        f, det = self.compare(olds, new)
        if all(self._explanation_missing(s, b) for s in olds):
            f["old_incomplete"] = 1.0
        cls = self.classify(f)
        rel = self.relation.get((olds[0], b), "same")
        rep = DriftReport(b, olds, cls, self.carry_over(f, rel), features=f, title=new.title,
                          punishment_old=det["punishment_old"].describe(), punishment_new=det["punishment_new"].describe(),
                          added_clauses=det["alignment"].added_clauses, lost_clauses=det["alignment"].lost_clauses)
        rep.flags = [k for k in ("mens_rea_changed", "threshold_changed", "punishment_changed", "merge", "split",
                                 "old_incomplete") if f[k]]
        rep.reasons = self.reasons(rep, det)
        return rep

    def pair(self, ipc_section: str, bns_section: str) -> PairDrift:
        """Carry-over for one cited IPC section into one BNS successor.

        The BNS section is aligned against ALL its predecessors at once, and each BNS clause is
        attributed to the predecessor it matches best; the pair is then judged on the IPC
        section's own clauses and the BNS clauses attributed to it. So BNS 318's clauses that
        came from IPC 415 do not count as "rewording" of IPC 420. A split IPC section (one IPC,
        several BNS) is judged for retention against the union of its successors.
        """
        i, b = _base(ipc_section), _base(bns_section)
        rel = self.relation.get((i, b), "same")
        if i not in self.ipc or b not in self.bns:
            return PairDrift(i, b, float("nan"), self.params["prior"].get(rel, 0.8), DriftClass.UNVERIFIED)
        group = [s for s in self.predecessors.get(b, [i]) if s in self.ipc] or [i]
        if len(group) == 1:
            f, _ = self.compare([i], self.bns[b])
        else:
            f = self._attributed(i, group, self.bns[b])
        # the BNS section comes from an Explanation of the IPC section; when our IPC text has no
        # Explanation, the pair cannot be judged from text
        if self._explanation_missing(i, b):
            f["old_incomplete"] = 1.0
        succ = [s for s in self.successors.get(i, []) if s in self.bns]
        if len(succ) > 1:
            union = Provision("bns", "+".join(succ), "", " ".join(self.bns[s].text for s in succ))
            f["retention"] = self.compare([i], union)[0]["retention"]
        return PairDrift(i, b, f["retention"], self.carry_over(f, rel), self.classify(f))

    def _explanation_missing(self, i: str, b: str) -> bool:
        return ("explanation" in self.notes.get((i, b), "")
                and not re.search(r"\bexplanation\b", self.ipc[i].text if i in self.ipc else "", re.I))

    def _attributed(self, i: str, group: list[str], new: Provision) -> dict:
        owners: list[str] = []
        clauses: list[str] = []
        for s in group:
            t = renumber_references(self.ipc[s].text, self.renumber) if self.renumbering else self.ipc[s].text
            cs = _no_illustrations(split_clauses(t))
            clauses += cs
            owners += [s] * len(cs)
        new_clauses = _no_illustrations(new.clauses or split_clauses(new.text))
        al = dfm.align(clauses, new_clauses, match_min=self.params["match_min"])
        kept_owners = [o for c, o in zip(clauses, owners) if dfm.normalise(c)]
        mine_old = [k for k, o in enumerate(kept_owners) if o == i]
        mine_new = [n for n, j, c in al.pairs if kept_owners[j] == i and c >= self.params["match_min"]]
        own_text = " ".join(al.old_clauses[k] for k in mine_old)
        new_text = " ".join(al.new_clauses[n] for n in mine_new)
        sub = Provision("bns", new.section, new.title, new_text or new.text, clauses=[al.new_clauses[n] for n in mine_new])
        f, _ = self.compare([i], sub)
        w = [len(dfm.normalise(al.old_clauses[k])) for k in mine_old]
        if w:
            f["retention"] = sum(al.old_scores[k] * x for k, x in zip(mine_old, w)) / sum(w)
        f["added_share"] = _added_share(al)
        f["merge"] = 1.0
        f["old_incomplete"] = float(self._incomplete(own_text, new_text or new.text) and not mine_new)
        return f

    def repealed(self) -> list[DriftReport]:
        """IPC sections the BNS did not carry over.

        Candidates are IPC sections with no successor in the crosswalk, minus those repealed long
        before 2023 ("Rep. by the Prevention of Corruption Act, 1988"). The crosswalk is built from
        the BNS side and misses some carried-over sections, so each candidate is searched for in
        the BNS text first: if most of its wording (bigram containment >= `inferred_min`) sits in
        one BNS section, it is recorded in `inferred` as an unmapped successor, not as repealed.
        """
        out = []
        self.inferred = {}
        bns_tokens = {b: dfm.normalise(p.text) for b, p in self.bns.items()}
        for s, p in self.ipc.items():
            if s in self.successors or _PRE_REPEALED.search(p.title + " " + p.text[:60]):
                continue
            old = dfm.normalise(renumber_references(p.text, self.renumber))
            if len(old) >= 5:
                b, score = max(((b, dfm.containment(old, t)) for b, t in bns_tokens.items()), key=lambda x: x[1])
                if score >= self.params["inferred_min"]:
                    self.inferred[s] = (b, round(score, 3))
                    continue
            out.append(DriftReport("", [s], DriftClass.REPEALED, 0.0, title=p.title,
                                   punishment_old=pun.extract(p.text).describe(),
                                   reasons=[f"IPC {s.upper()} has no successor in the BNS: an IPC-era precedent on it "
                                            "does not apply to an offence committed on or after 1 July 2024"]))
        return out

    def reasons(self, rep: DriftReport, det: dict) -> list[str]:
        f = rep.features
        out: list[str] = []
        olds = ", ".join(f"IPC {s.upper()}" for s in rep.ipc_sections)
        head = {DriftClass.IDENTICAL: "same text", DriftClass.COSMETIC: "same law, reworded",
                DriftClass.RESTRUCTURED: "same law, sections merged or split",
                DriftClass.PUNISHMENT: "same offence, different punishment",
                DriftClass.SUBSTANTIVE: "the provision itself changed",
                DriftClass.UNVERIFIED: "the IPC text in our source is incomplete; change not verified"}
        out.append(f"BNS {rep.bns_section.upper()} <- {olds}: {head.get(rep.drift_class, rep.drift_class.value)}")
        if not f.get("old_incomplete"):
            out.append(f"{f['retention']:.0%} of the IPC text survives; {f['novelty']:.0%} of the BNS text is new")
        ed: elm.ElementDelta = det["elements"]
        if ed.mens_rea_added:
            out.append("mental state added: " + ", ".join(sorted(m.replace("_", " ") for m in ed.mens_rea_added)))
        if ed.mens_rea_removed:
            out.append("mental state dropped: " + ", ".join(sorted(m.replace("_", " ") for m in ed.mens_rea_removed)))
        if ed.threshold_changed:
            old = ", ".join(f"{k.replace('_', ' ')} {v:g}" for k, v in sorted(ed.thresholds_removed))
            new = ", ".join(f"{k.replace('_', ' ')} {v:g}" for k, v in sorted(ed.thresholds_added))
            out.append(f"threshold changed: {old} -> {new}")
        pd: pun.PunishmentDelta = det["punishment_delta"]
        if pd.any:
            out.append(f"punishment {pd.direction}: {rep.punishment_old} -> {rep.punishment_new}")
        if det["alignment"].added_clauses:
            out.append(f"{len(det['alignment'].added_clauses)} clause(s) with no IPC counterpart")
        if f.get("merge"):
            out.append(f"merges {len(rep.ipc_sections)} IPC sections")
        if f.get("split"):
            out.append("part of an IPC section that the BNS split")
        return out


# --------------------------------------------------------------------------- the table


@dataclass
class DriftTable:
    """All reports and pair carry-overs; what the search engine and the app load."""

    sections: dict[str, DriftReport] = field(default_factory=dict)    # bns base section -> report
    repealed: dict[str, DriftReport] = field(default_factory=dict)    # ipc base section -> report
    pairs: dict[tuple[str, str], PairDrift] = field(default_factory=dict)
    inferred: dict[str, tuple[str, float]] = field(default_factory=dict)   # unmapped IPC -> (BNS, containment)

    @classmethod
    def build(cls, analyzer: DriftAnalyzer | None = None, cfg: Config | None = None) -> "DriftTable":
        an = analyzer or DriftAnalyzer.from_sources(cfg)
        t = cls()
        for b in an.bns:
            t.sections[b] = an.report(b)
        for r in an.repealed():
            t.repealed[r.ipc_sections[0]] = r
        for (i, b) in an.relation:
            if b in an.bns:
                t.pairs[(i, b)] = an.pair(i, b)
        # a section's carry-over is the mean over its predecessors of the pairwise carry-over: how
        # safely a precedent on one of its IPC sections reads onto it (merges are not penalised
        # for the parts that came from the other merged sections)
        for b, rep in t.sections.items():
            cs = [t.pairs[(i, b)].carry_over for i in rep.ipc_sections if (i, b) in t.pairs]
            if cs:
                rep.carry_over = round(sum(cs) / len(cs), 4)
        t.inferred = dict(an.inferred)
        return t

    # ------------------------------------------------------------------ lookups
    def report(self, ref: str) -> DriftReport | None:
        code, _, sec = ref.lower().partition(":")
        sec = _base(sec)
        if code == "bns":
            return self.sections.get(sec)
        if code == "ipc":
            if sec in self.repealed:
                return self.repealed[sec]
            succ = [b for (i, b) in self.pairs if i == sec]
            return self.sections.get(succ[0]) if succ else None
        return None

    def carry_over(self, ipc_ref: str, bns_ref: str) -> float | None:
        """c(ipc -> bns) for a crosswalk pair; 0 for a repealed IPC section; None if not a pair."""
        i = _base(ipc_ref.lower().removeprefix("ipc:"))
        b = _base(bns_ref.lower().removeprefix("bns:"))
        if i in self.repealed:
            return 0.0
        p = self.pairs.get((i, b))
        return p.carry_over if p else None

    def successors(self, ipc_ref: str) -> list[str]:
        i = _base(ipc_ref.lower().removeprefix("ipc:"))
        return [b for (x, b) in self.pairs if x == i]

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for r in self.sections.values():
            out[r.drift_class.value] += 1
        out["repealed"] = len(self.repealed)
        return dict(out)

    # ------------------------------------------------------------------ CSV (committed)
    SECTION_COLS = ["bns_section", "ipc_sections", "title", "drift_class", "carry_over", *FEATURES, "flags",
                    "punishment_old", "punishment_new", "reasons"]
    PAIR_COLS = ["ipc_section", "bns_section", "drift_class", "retention", "carry_over"]

    def save_csv(self, directory: str | Path | None = None) -> tuple[Path, Path]:
        d = Path(directory) if directory else project_path("data/crosswalk")
        d.mkdir(parents=True, exist_ok=True)
        sp, pp = d / "drift_sections.csv", d / "drift_pairs.csv"
        with open(sp, "w", newline="", encoding="utf-8") as fh:
            fh.write("# Generated by scripts/13_build_drift.py (kanoon_bridge.drift): one row per BNS section\n"
                     "# plus one per IPC section with no successor (bns_section empty, class 'repealed').\n")
            w = csv.writer(fh)
            w.writerow(self.SECTION_COLS)
            for r in [*self.sections.values(), *self.repealed.values()]:
                w.writerow([r.bns_section, ";".join(r.ipc_sections), r.title, r.drift_class.value, f"{r.carry_over:.4f}",
                            *[f"{r.features.get(k, 0.0):.4f}" if r.features else "" for k in FEATURES],
                            ";".join(r.flags), r.punishment_old, r.punishment_new, " | ".join(r.reasons)])
        with open(pp, "w", newline="", encoding="utf-8") as fh:
            fh.write("# Generated by scripts/13_build_drift.py: carry-over c(ipc -> bns) per crosswalk pair\n")
            w = csv.writer(fh)
            w.writerow(self.PAIR_COLS)
            for p in self.pairs.values():
                w.writerow([p.ipc_section, p.bns_section, p.drift_class.value,
                            "" if math.isnan(p.retention) else f"{p.retention:.4f}", f"{p.carry_over:.4f}"])
        return sp, pp

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "DriftTable":
        d = Path(directory) if directory else project_path("data/crosswalk")
        sp, pp = d / "drift_sections.csv", d / "drift_pairs.csv"
        if not sp.exists() or not pp.exists():
            raise FileNotFoundError(f"{sp} missing: run scripts/13_build_drift.py")
        t = cls()
        with open(sp, encoding="utf-8") as fh:
            for row in csv.DictReader(l for l in fh if not l.startswith("#")):
                feats = {k: float(row[k]) for k in FEATURES if row.get(k)}
                r = DriftReport(row["bns_section"], [s for s in row["ipc_sections"].split(";") if s],
                                DriftClass(row["drift_class"]), float(row["carry_over"]), features=feats,
                                flags=[x for x in row["flags"].split(";") if x], punishment_old=row["punishment_old"],
                                punishment_new=row["punishment_new"],
                                reasons=[x for x in row["reasons"].split(" | ") if x], title=row.get("title", ""))
                if r.bns_section:
                    t.sections[r.bns_section] = r
                else:
                    t.repealed[r.ipc_sections[0]] = r
        with open(pp, encoding="utf-8") as fh:
            for row in csv.DictReader(l for l in fh if not l.startswith("#")):
                t.pairs[(row["ipc_section"], row["bns_section"])] = PairDrift(
                    row["ipc_section"], row["bns_section"], float(row["retention"]) if row["retention"] else float("nan"),
                    float(row["carry_over"]), DriftClass(row["drift_class"]))
        return t


__all__ = ["DriftClass", "DriftReport", "PairDrift", "DriftAnalyzer", "DriftTable", "FEATURES", "DEFAULTS"]
