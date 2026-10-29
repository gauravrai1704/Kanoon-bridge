"""Drift reports: how far each new-code section moved from its old-code predecessor(s).  [owner: Gaurav — working]

Works for the three code families replaced on 1 July 2024 (provisions.FAMILIES): IPC -> BNS,
CrPC -> BNSS, Evidence Act -> BSA. The examples below use the penal family; the field names
are generic (old_sections / new_section).

For every new section (and every old section left without a successor) the analyser builds
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
    table.carry_over("ipc:124a", "bns:152")       # ~0.2
    table.save_csv(); DriftTable.load()           # data/crosswalk/drift_*.csv (committed)
    DriftTable.build(DriftAnalyzer.from_sources(family="crpc"))   # CrPC -> BNSS, same API
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
from kanoon_bridge.drift.provisions import FAMILIES, Family, Provision, renumber_references, split_clauses


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
    "inserted_min": 99,           # this many inserted content words = substantive (99 = off; see drift_eval)
    "threshold_added_substantive": 0,   # 1 = a newly added time / age / count limit is substantive
    "incomplete_words": 15,       # an old text shorter than this is treated as incomplete
    "prior": {"same": 0.95, "merge": 0.85, "split": 0.8, "modified": 0.6},   # carry-over when unverifiable
}

FEATURES = ["retention", "novelty", "novelty_matched", "added_share", "edit_similarity", "inserted_words",
            "deleted_words", "threshold_added", "added_clauses", "lost_clauses", "mens_rea_changed",
            "threshold_changed", "punishment_changed", "punishment_harsher", "content_jaccard", "merge",
            "split", "title_similarity", "log_length_ratio", "old_incomplete"]


@dataclass
class DriftReport:
    new_section: str                         # "" for a repealed old-code section
    old_sections: list[str]
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
    old_code: str = "ipc"
    new_code: str = "bns"

    @property
    def ref(self) -> str:
        return f"{self.new_code}:{self.new_section}" if self.new_section else f"{self.old_code}:{self.old_sections[0]}"


@dataclass
class PairDrift:
    old_section: str
    new_section: str
    retention: float
    carry_over: float
    drift_class: DriftClass


_PRE_REPEALED = re.compile(r"^\s*(?:rep\.|repealed|omitted)\b|\brepealed by\b|\brep\.\s+by\b", re.I)


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
    old_prov: dict[str, Provision]           # old-code sections (IPC / CrPC / Evidence Act)
    new_prov: dict[str, Provision]           # new-code sections (BNS / BNSS / BSA)
    predecessors: dict[str, list[str]]       # new base section -> old base sections
    successors: dict[str, list[str]]         # old base section -> new base sections
    relation: dict[tuple[str, str], str]     # (old, new) -> crosswalk relation
    notes: dict[tuple[str, str], str] = field(default_factory=dict)   # (old, new) -> crosswalk note
    family: Family = FAMILIES["ipc"]
    params: dict = field(default_factory=lambda: dict(DEFAULTS))
    renumber: dict[str, str] = field(default_factory=dict)
    renumber_other: dict[str, dict[str, str]] = field(default_factory=dict)   # other code -> its renumbering
    renumbering: bool = True                 # ablation switch: renumber old cross-references
    use_punishment: bool = True              # ablation switches for the evaluation (off for CrPC / IEA)
    use_elements: bool = True
    clause_alignment: bool = True            # False = each text is one clause (ablation)
    inferred: dict[str, tuple[str, float]] = field(default_factory=dict)   # filled by repealed()

    @classmethod
    def from_sources(cls, cfg: Config | None = None, old: dict | None = None, new: dict | None = None,
                     normalizer=None, family: str = "ipc", crosswalk: list | None = None) -> "DriftAnalyzer":
        """`family`: "ipc" | "crpc" | "iea". `old` / `new` / `crosswalk` override what is read from
        disk (tests); `crosswalk` rows are (old section, new section, relation, note)."""
        from kanoon_bridge.drift.provisions import crosswalk_rows, load_bns, load_ipc, load_procedure

        cfg = cfg or load_config()
        fam = FAMILIES[family]
        if old is None or new is None:
            if family == "ipc":
                o = old if old is not None else load_ipc(cfg)
                n = new if new is not None else load_bns(cfg)
            else:
                o, n = load_procedure(fam, cfg)
            old, new = (old if old is not None else o), (new if new is not None else n)
        rows = crosswalk if crosswalk is not None else crosswalk_rows(fam, cfg, normalizer)
        pred: dict[str, list[str]] = defaultdict(list)
        succ: dict[str, list[str]] = defaultdict(list)
        rel: dict[tuple[str, str], str] = {}
        notes: dict[tuple[str, str], str] = {}
        for old_sec, new_sec, relation, note in rows:
            i, b = _base(old_sec), _base(new_sec)
            if not i or not b:
                continue
            if i not in pred[b]:
                pred[b].append(i)
            if b not in succ[i]:
                succ[i].append(b)
            rel.setdefault((i, b), relation)
            if note and not note.startswith("sim="):         # align_codes.py notes are similarity scores
                notes[(i, b)] = note.strip().lower()
        params = dict(DEFAULTS)
        params.update({k: v for k, v in (cfg.get("drift") or {}).items() if k in DEFAULTS})
        fam_params = ((cfg.get("drift") or {}).get("family_params") or {}).get(family) or {}
        params.update({k: v for k, v in fam_params.items() if k in DEFAULTS})
        an = cls(old_prov=old, new_prov=new, predecessors=dict(pred), successors=dict(succ), relation=rel,
                 params=params, notes=notes, family=fam)
        # procedure and evidence law rarely prescribe punishment; their "offence punishable with
        # death ... detention up to 15 days" text would read as one. Time limits there are caught
        # by the threshold reader (elements.thresholds: "within N days").
        an.use_punishment = family == "ipc"
        an.renumber = {i: s[0] for i, s in succ.items() if len(s) == 1}
        # references to the OTHER replaced codes inside this one (CrPC text citing IPC sections)
        for other in FAMILIES.values():
            if other.old == fam.old:
                continue
            try:
                m: dict[str, list[str]] = defaultdict(list)
                for o_sec, n_sec, *_ in crosswalk_rows(other, cfg, normalizer):
                    if _base(o_sec) and _base(n_sec) and _base(n_sec) not in m[_base(o_sec)]:
                        m[_base(o_sec)].append(_base(n_sec))
                an.renumber_other[other.old] = {i: v[0] for i, v in m.items() if len(v) == 1}
            except (FileNotFoundError, KeyError):
                continue
        return an

    # ------------------------------------------------------------------ helpers
    def _renumbered(self, text: str) -> str:
        if not self.renumbering:
            return text
        return renumber_references(text, self.renumber, self.renumber_other)

    def _old_text(self, sections: list[str]) -> tuple[str, list[str], str]:
        texts, clauses, titles = [], [], []
        for s in sections:
            p = self.old_prov.get(s)
            if p is None:
                continue
            t = self._renumbered(p.text)
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
        ins, dele = dfm.edit_counts(old_text, new.text) if self.use_elements else (0, 0)
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
            "threshold_added": float(ed.threshold_added),
            "inserted_words": float(ins),
            "deleted_words": float(dele),
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
                       or f.get("threshold_added", 0) and p["threshold_added_substantive"]
                       or f.get("inserted_words", 0) >= p["inserted_min"]
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
        new = self.new_prov.get(b)
        if new is None:
            raise KeyError(f"{self.family.new_label} {b} not in the source")
        olds = [s for s in self.predecessors.get(b, []) if s in self.old_prov]
        if not olds:
            F = self.family
            return DriftReport(b, [], DriftClass.NEW, 0.0, title=new.title, old_code=F.old, new_code=F.new,
                               punishment_new=pun.extract(new.text).describe() if self.use_punishment else "",
                               reasons=[f"new in the {F.new_label}: no {F.old_label} provision corresponds, so no "
                                        f"{F.old_label}-era precedent applies directly"])
        f, det = self.compare(olds, new)
        if all(self._explanation_missing(s, b) for s in olds) or new.defective:
            f["old_incomplete"] = 1.0
        cls = self.classify(f)
        rel = self.relation.get((olds[0], b), "same")
        rep = DriftReport(b, olds, cls, self.carry_over(f, rel), features=f, title=new.title,
                          old_code=self.family.old, new_code=self.family.new,
                          punishment_old=det["punishment_old"].describe() if self.use_punishment else "",
                          punishment_new=det["punishment_new"].describe() if self.use_punishment else "",
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
        if i not in self.old_prov or b not in self.new_prov:
            return PairDrift(i, b, float("nan"), self.params["prior"].get(rel, 0.8), DriftClass.UNVERIFIED)
        group = [s for s in self.predecessors.get(b, [i]) if s in self.old_prov] or [i]
        if len(group) == 1:
            f, _ = self.compare([i], self.new_prov[b])
        else:
            f = self._attributed(i, group, self.new_prov[b])
        # the BNS section comes from an Explanation of the IPC section; when our IPC text has no
        # Explanation, the pair cannot be judged from text
        if self._explanation_missing(i, b) or self.new_prov[b].defective:
            f["old_incomplete"] = 1.0
        succ = [s for s in self.successors.get(i, []) if s in self.new_prov]
        if len(succ) > 1:
            union = Provision(self.family.new, "+".join(succ), "", " ".join(self.new_prov[s].text for s in succ))
            f["retention"] = self.compare([i], union)[0]["retention"]
        return PairDrift(i, b, f["retention"], self.carry_over(f, rel), self.classify(f))

    def _explanation_missing(self, i: str, b: str) -> bool:
        return ("explanation" in self.notes.get((i, b), "")
                and not re.search(r"\bexplanation\b", self.old_prov[i].text if i in self.old_prov else "", re.I))

    def _attributed(self, i: str, group: list[str], new: Provision) -> dict:
        owners: list[str] = []
        clauses: list[str] = []
        for s in group:
            t = self._renumbered(self.old_prov[s].text)
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
        sub = Provision(self.family.new, new.section, new.title, new_text or new.text, clauses=[al.new_clauses[n] for n in mine_new])
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
        bns_tokens = {b: dfm.normalise(p.text) for b, p in self.new_prov.items()}
        for s, p in self.old_prov.items():
            if s in self.successors or _PRE_REPEALED.search(p.title.lstrip("[ ") + " " + p.text[:60].lstrip("[ ")):
                continue
            old = dfm.normalise(self._renumbered(p.text))
            if len(old) >= 5:
                b, score = max(((b, dfm.containment(old, t)) for b, t in bns_tokens.items()), key=lambda x: x[1])
                if score >= self.params["inferred_min"]:
                    self.inferred[s] = (b, round(score, 3))
                    continue
            F = self.family
            out.append(DriftReport("", [s], DriftClass.REPEALED, 0.0, title=p.title, old_code=F.old, new_code=F.new,
                                   punishment_old=pun.extract(p.text).describe() if self.use_punishment else "",
                                   reasons=[f"{F.old_label} {s.upper()} has no successor in the {F.new_label}: an "
                                            f"{F.old_label}-era precedent on it does not apply on or after 1 July 2024"]))
        return out

    def reasons(self, rep: DriftReport, det: dict) -> list[str]:
        f = rep.features
        out: list[str] = []
        F = self.family
        olds = ", ".join(f"{F.old_label} {s.upper()}" for s in rep.old_sections)
        head = {DriftClass.IDENTICAL: "same text", DriftClass.COSMETIC: "same law, reworded",
                DriftClass.RESTRUCTURED: "same law, sections merged or split",
                DriftClass.PUNISHMENT: "same offence, different punishment",
                DriftClass.SUBSTANTIVE: "the provision itself changed",
                DriftClass.UNVERIFIED: f"the {F.old_label} text in our source is incomplete; change not verified"}
        defect = self.new_prov[rep.new_section].defective if rep.new_section in self.new_prov else ""
        if defect and rep.drift_class == DriftClass.UNVERIFIED:
            head[DriftClass.UNVERIFIED] = f"not verified: {defect}"
        out.append(f"{F.new_label} {rep.new_section.upper()} <- {olds}: {head.get(rep.drift_class, rep.drift_class.value)}")
        if not f.get("old_incomplete"):
            out.append(f"{f['retention']:.0%} of the {F.old_label} text survives; {f['novelty']:.0%} of the "
                       f"{F.new_label} text is new")
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
            out.append(f"{len(det['alignment'].added_clauses)} clause(s) with no {F.old_label} counterpart")
        if f.get("merge"):
            out.append(f"merges {len(rep.old_sections)} {F.old_label} sections")
        if f.get("split"):
            article = "an" if F.old_label[0] in "AEIOU" else "a"
            out.append(f"part of {article} {F.old_label} section that the {F.new_label} split")
        return out


# --------------------------------------------------------------------------- the table


@dataclass
class DriftTable:
    """All reports and pair carry-overs; what the search engine and the app load."""

    sections: dict[str, DriftReport] = field(default_factory=dict)    # new base section -> report
    repealed: dict[str, DriftReport] = field(default_factory=dict)    # old base section -> report
    pairs: dict[tuple[str, str], PairDrift] = field(default_factory=dict)
    inferred: dict[str, tuple[str, float]] = field(default_factory=dict)   # unmapped old -> (new, containment)
    family: Family = FAMILIES["ipc"]

    @classmethod
    def build(cls, analyzer: DriftAnalyzer | None = None, cfg: Config | None = None) -> "DriftTable":
        an = analyzer or DriftAnalyzer.from_sources(cfg)
        t = cls(family=an.family)
        for b in an.new_prov:
            t.sections[b] = an.report(b)
        for r in an.repealed():
            t.repealed[r.old_sections[0]] = r
        for (i, b) in an.relation:
            if b in an.new_prov:
                t.pairs[(i, b)] = an.pair(i, b)
        # a section's carry-over is the mean over its predecessors of the pairwise carry-over: how
        # safely a precedent on one of its IPC sections reads onto it (merges are not penalised
        # for the parts that came from the other merged sections)
        for b, rep in t.sections.items():
            cs = [t.pairs[(i, b)].carry_over for i in rep.old_sections if (i, b) in t.pairs]
            if cs:
                rep.carry_over = round(sum(cs) / len(cs), 4)
        t.inferred = dict(an.inferred)
        return t

    # ------------------------------------------------------------------ lookups
    def report(self, ref: str) -> DriftReport | None:
        code, _, sec = ref.lower().partition(":")
        sec = _base(sec)
        if code == self.family.new:
            return self.sections.get(sec)
        if code == self.family.old:
            if sec in self.repealed:
                return self.repealed[sec]
            succ = [b for (i, b) in self.pairs if i == sec]
            return self.sections.get(succ[0]) if succ else None
        return None

    def carry_over(self, old_ref: str, new_ref: str) -> float | None:
        """c(old -> new) for a crosswalk pair; 0 for a repealed old section; None if not a pair.
        Refs may carry their code ("ipc:302") or not ("302")."""
        i = _base(old_ref.lower().removeprefix(self.family.old + ":"))
        b = _base(new_ref.lower().removeprefix(self.family.new + ":"))
        if i in self.repealed:
            return 0.0
        p = self.pairs.get((i, b))
        return p.carry_over if p else None

    def successors(self, old_ref: str) -> list[str]:
        i = _base(old_ref.lower().removeprefix(self.family.old + ":"))
        return [b for (x, b) in self.pairs if x == i]

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for r in self.sections.values():
            out[r.drift_class.value] += 1
        out["repealed"] = len(self.repealed)
        return dict(out)

    # ------------------------------------------------------------------ CSV (committed)
    SECTION_COLS = ["new_section", "old_sections", "title", "drift_class", "carry_over", *FEATURES, "flags",
                    "punishment_old", "punishment_new", "reasons"]
    PAIR_COLS = ["old_section", "new_section", "drift_class", "retention", "carry_over"]

    @staticmethod
    def paths(directory: str | Path | None = None, family: str = "ipc") -> tuple[Path, Path]:
        """drift_sections.csv / drift_pairs.csv for the penal family, drift_<old>_<new>_*.csv otherwise."""
        d = Path(directory) if directory else project_path("data/crosswalk")
        fam = FAMILIES[family]
        stem = "drift" if family == "ipc" else f"drift_{fam.old}_{fam.new}"
        return d / f"{stem}_sections.csv", d / f"{stem}_pairs.csv"

    def save_csv(self, directory: str | Path | None = None) -> tuple[Path, Path]:
        F = self.family
        sp, pp = self.paths(directory, F.old)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w", newline="", encoding="utf-8") as fh:
            fh.write(f"# Generated by scripts/13_build_drift.py (kanoon_bridge.drift): one row per {F.new_label} section\n"
                     f"# plus one per {F.old_label} section with no successor (new_section empty, class 'repealed').\n")
            w = csv.writer(fh)
            w.writerow(self.SECTION_COLS)
            for r in [*self.sections.values(), *self.repealed.values()]:
                w.writerow([r.new_section, ";".join(r.old_sections), r.title, r.drift_class.value, f"{r.carry_over:.4f}",
                            *[f"{r.features.get(k, 0.0):.4f}" if r.features else "" for k in FEATURES],
                            ";".join(r.flags), r.punishment_old, r.punishment_new, " | ".join(r.reasons)])
        with open(pp, "w", newline="", encoding="utf-8") as fh:
            fh.write(f"# Generated by scripts/13_build_drift.py: carry-over c({F.old} -> {F.new}) per crosswalk pair\n")
            w = csv.writer(fh)
            w.writerow(self.PAIR_COLS)
            for p in self.pairs.values():
                w.writerow([p.old_section, p.new_section, p.drift_class.value,
                            "" if math.isnan(p.retention) else f"{p.retention:.4f}", f"{p.carry_over:.4f}"])
        return sp, pp

    @classmethod
    def load(cls, directory: str | Path | None = None, family: str = "ipc") -> "DriftTable":
        sp, pp = cls.paths(directory, family)
        if not sp.exists() or not pp.exists():
            raise FileNotFoundError(f"{sp} missing: run scripts/13_build_drift.py")
        t = cls(family=FAMILIES[family])
        with open(sp, encoding="utf-8") as fh:
            for row in csv.DictReader(l for l in fh if not l.startswith("#")):
                feats = {k: float(row[k]) for k in FEATURES if row.get(k)}
                r = DriftReport(row["new_section"], [s for s in row["old_sections"].split(";") if s],
                                DriftClass(row["drift_class"]), float(row["carry_over"]), features=feats,
                                flags=[x for x in row["flags"].split(";") if x], punishment_old=row["punishment_old"],
                                punishment_new=row["punishment_new"],
                                reasons=[x for x in row["reasons"].split(" | ") if x], title=row.get("title", ""),
                                old_code=t.family.old, new_code=t.family.new)
                if r.new_section:
                    t.sections[r.new_section] = r
                else:
                    t.repealed[r.old_sections[0]] = r
        with open(pp, encoding="utf-8") as fh:
            for row in csv.DictReader(l for l in fh if not l.startswith("#")):
                t.pairs[(row["old_section"], row["new_section"])] = PairDrift(
                    row["old_section"], row["new_section"], float(row["retention"]) if row["retention"] else float("nan"),
                    float(row["carry_over"]), DriftClass(row["drift_class"]))
        return t


__all__ = ["DriftClass", "DriftReport", "PairDrift", "DriftAnalyzer", "DriftTable", "FEATURES", "DEFAULTS"]
