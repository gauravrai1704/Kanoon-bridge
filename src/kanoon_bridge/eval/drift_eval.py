"""E11-E12: does the drift model see what changed between the IPC and the BNS?  [owner: Gaurav — working]

E11a  change detection per BNS section (sections with an IPC predecessor), against two
      independent gold sources (drift/gold.py). Systems:
          v1_crosswalk      what layer 1 assumes: changed iff the crosswalk relation is not "same"
          edit_similarity   difflib ratio of the two texts below a threshold (tuned on dev)
          tfidf_cosine      tf-idf cosine of the two texts below a threshold (tuned on dev)
          drift_rules       ours, no tuning: class PUNISHMENT or SUBSTANTIVE
          drift_rules_subst ours, no tuning: class SUBSTANTIVE only
          drift_rules_dev   ours: the rule thresholds grid-searched on the dev half (`tune_rules`)
          drift_lr          ours, learned: logistic regression on the drift vector (5-fold CV;
                            the test-split row is trained on dev only)
      Dev/test split: by section number (even/odd of a stable hash), half each. Thresholds and
      the dev-trained model never see the test half. Metrics: accuracy, F1 of "changed",
      macro-F1, Cohen's kappa; AUC for scored systems; McNemar's exact test against v1.
E11b  ablations of drift_rules (renumbering, punishment, elements, clause alignment), all rows.
E11c  punishment change detection against S2's "punishment" category.
E11d  punishment extraction: punishment.extract on the bare act vs on the source's own
      one-line punishment summary, field by field.
E11e  repeal detection against S1's Deleted IPC sections; AUC of 1 - carry-over.
E11f  carry-over by S2 category (does the score fall as the change grows?).
E12   hazard audit: layer-1 offence ids that equate an IPC section with a BNS section whose
      carry-over is below 0.5 -- the places where version normalisation silently transfers
      precedent across a changed law.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from kanoon_bridge.drift import classify, gold
from kanoon_bridge.drift import punishment as pun
from kanoon_bridge.drift.model import FEATURES, DriftAnalyzer, DriftClass, DriftTable

# --------------------------------------------------------------------------- metrics


def binary_metrics(y: list[int], p: list[int]) -> dict[str, float]:
    y, p = np.asarray(y, int), np.asarray(p, int)
    tp, fp = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum())
    fn, tn = int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum())
    n = len(y) or 1

    def f1(tp_, fp_, fn_):
        return 2 * tp_ / (2 * tp_ + fp_ + fn_) if tp_ else 0.0

    f_pos, f_neg = f1(tp, fp, fn), f1(tn, fn, fp)
    return {"n": len(y), "accuracy": (tp + tn) / n, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0, "f1_changed": f_pos, "macro_f1": (f_pos + f_neg) / 2,
            "kappa": cohen_kappa(y.tolist(), p.tolist())}


def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def auc(y: list[int], score: list[float]) -> float:
    """P(score of a positive > score of a negative), ties half (Mann-Whitney)."""
    pos = [s for s, t in zip(score, y) if t == 1]
    neg = [s for s, t in zip(score, y) if t == 0]
    if not pos or not neg:
        return float("nan")
    allv = sorted((s, i) for i, s in enumerate(pos + neg))
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[allv[k][1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = sum(ranks[: len(pos)])
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def mcnemar(y: list[int], a: list[int], b: list[int]) -> float:
    """Exact two-sided McNemar p-value: do systems a and b err on different items more than chance?"""
    n01 = sum(1 for t, x, z in zip(y, a, b) if x == t and z != t)
    n10 = sum(1 for t, x, z in zip(y, a, b) if x != t and z == t)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def bootstrap_ci(y: list[int], p: list[int], stat: str = "f1_changed", reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        vals.append(binary_metrics(y[idx].tolist(), p[idx].tolist())[stat])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def split_of(section: str) -> str:
    """Stable dev/test assignment by section number."""
    return "dev" if int(hashlib.md5(section.encode()).hexdigest(), 16) % 2 == 0 else "test"


# --------------------------------------------------------------------------- baselines


def _tfidf_cosines(an: DriftAnalyzer, items: list[tuple[str, list[str]]]) -> dict[str, float]:
    from kanoon_bridge.drift.diff import normalise

    docs = [normalise(p.text) for p in an.ipc.values()] + [normalise(p.text) for p in an.bns.values()]
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)

    def vec(tokens):
        tf = Counter(tokens)
        v = {t: (1 + math.log(c)) * math.log(n / df.get(t, 1)) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    out = {}
    for b, olds in items:
        a = vec(normalise(" ".join(an.ipc[s].text for s in olds)))
        c = vec(normalise(an.bns[b].text))
        out[b] = sum(w * c.get(t, 0.0) for t, w in a.items())
    return out


def _best_threshold(scores: list[float], y: list[int], below_is_changed: bool = True) -> float:
    """Threshold maximising macro-F1 on the given (dev) rows."""
    best, best_t = -1.0, 0.5
    for t in sorted(set(scores)) + [1.01]:
        pred = [int(s < t) if below_is_changed else int(s >= t) for s in scores]
        m = binary_metrics(y, pred)["macro_f1"]
        if m > best:
            best, best_t = m, t
    return best_t


# --------------------------------------------------------------------------- the experiment


@dataclass
class Population:
    sections: list[str]                       # BNS sections with an IPC predecessor and a gold label
    olds: dict[str, list[str]]
    y: dict[str, dict[str, int]]              # target -> section -> 0/1
    s2: dict[str, set[str]]


def population(an: DriftAnalyzer, cfg=None) -> Population:
    s1 = gold.s1_section_labels(cfg)
    s2 = gold.s2_section_labels(cfg)
    secs, olds = [], {}
    for b in an.bns:
        o = [s for s in an.predecessors.get(b, []) if s in an.ipc]
        if not o or s1.get(b) in (None, "new"):
            continue
        secs.append(b)
        olds[b] = o
    # S2 says "NEW: ..." about added content of a section that does have an IPC predecessor
    s2_fixed = {b: ({"substance"} | (v - {"new"})) if "new" in v else v for b, v in s2.items()}
    y = {"S1": {b: int(s1[b] == "changed") for b in secs},
         "S2": {b: int(gold.s2_changed(s2_fixed.get(b, set()))) for b in secs}}
    return Population(secs, olds, y, s2_fixed)


def _rules_pred(table: DriftTable, secs: list[str], subst_only: bool = False) -> list[int]:
    want = {DriftClass.SUBSTANTIVE} if subst_only else {DriftClass.SUBSTANTIVE, DriftClass.PUNISHMENT}
    return [int(table.sections[b].drift_class in want) for b in secs]


RULE_GRID = {"novelty_substantive": [0.25, 0.35, 0.45, 0.55, 0.65],
             "retention_substantive": [0.4, 0.55, 0.7],
             "added_min": [1, 2, 3, 99]}


def _rules_with(an: DriftAnalyzer, table: DriftTable, secs: list[str], params: dict) -> list[int]:
    saved = dict(an.params)
    an.params.update(params)
    try:
        return [int(an.classify(table.sections[b].features) in (DriftClass.SUBSTANTIVE, DriftClass.PUNISHMENT))
                for b in secs]
    finally:
        an.params = saved


def tune_rules(an: DriftAnalyzer, table: DriftTable, secs: list[str], y: list[int]) -> dict:
    """Grid search of the three substantive-change thresholds, macro-F1 on the given (dev) rows."""
    import itertools

    best, best_p = -1.0, {}
    for vals in itertools.product(*RULE_GRID.values()):
        params = dict(zip(RULE_GRID, vals))
        m = binary_metrics(y, _rules_with(an, table, secs, params))["macro_f1"]
        if m > best + 1e-12:
            best, best_p = m, params
    return best_p


def change_detection(an: DriftAnalyzer, table: DriftTable, pop: Population) -> tuple[list[dict], dict]:
    """E11a. Returns (rows, extras) where extras keeps per-system predictions for plots/tests."""
    secs = pop.sections
    rel_changed = {b: int(any(an.relation.get((s, b), "same") != "same" for s in pop.olds[b])) for b in secs}
    edit = {b: table.sections[b].features["edit_similarity"] for b in secs}
    cos = _tfidf_cosines(an, [(b, pop.olds[b]) for b in secs])
    X = np.array([[table.sections[b].features[k] for k in FEATURES] for b in secs])
    rows: list[dict] = []
    extras: dict = {"coef": {}}
    for target in ("S1", "S2"):
        y_all = [pop.y[target][b] for b in secs]
        dev = [i for i, b in enumerate(secs) if split_of(b) == "dev"]
        test = [i for i, b in enumerate(secs) if split_of(b) == "test"]
        yd, yt = [y_all[i] for i in dev], [y_all[i] for i in test]
        preds_all: dict[str, list[int]] = {}
        scores_all: dict[str, list[float]] = {}
        preds_all["v1_crosswalk"] = [rel_changed[b] for b in secs]
        for name, sc in (("edit_similarity", edit), ("tfidf_cosine", cos)):
            vals = [sc[b] for b in secs]
            t = _best_threshold([vals[i] for i in dev], yd)
            preds_all[name] = [int(v < t) for v in vals]
            scores_all[name] = [-v for v in vals]
        preds_all["drift_rules"] = _rules_pred(table, secs)
        preds_all["drift_rules_subst"] = _rules_pred(table, secs, subst_only=True)
        tuned = tune_rules(an, table, [secs[i] for i in dev], yd)
        extras.setdefault("tuned", {})[target] = tuned
        preds_all["drift_rules_dev"] = _rules_with(an, table, secs, tuned)
        scores_all["drift_rules"] = [1 - table.sections[b].carry_over for b in secs]
        oof = classify.cross_val_predict(X, np.array(y_all), folds=5, seed=0)
        dev_model = classify.fit(X[dev], np.array(yd))
        test_prob = dev_model.predict_proba(X[test])
        lr_test = {i: int(p >= 0.5) for i, p in zip(test, test_prob)}
        preds_all["drift_lr"] = [int(p >= 0.5) for p in oof]
        scores_all["drift_lr"] = oof.tolist()
        extras["coef"][target] = classify.fit(X, np.array(y_all)).coefficients(FEATURES)
        for system, pred in preds_all.items():
            for split, idx in (("all", list(range(len(secs)))), ("test", test)):
                if system == "drift_lr" and split == "test":
                    p_ = [lr_test[i] for i in idx]
                    s_ = [float(test_prob[test.index(i)]) for i in idx]
                else:
                    p_ = [pred[i] for i in idx]
                    s_ = [scores_all[system][i] for i in idx] if system in scores_all else None
                y_ = [y_all[i] for i in idx]
                m = binary_metrics(y_, p_)
                lo, hi = bootstrap_ci(y_, p_)
                base = [preds_all["v1_crosswalk"][i] for i in idx]
                rows.append({"target": target, "split": split, "system": system, **{k: round(v, 4) for k, v in m.items()},
                             "f1_ci_low": round(lo, 4), "f1_ci_high": round(hi, 4),
                             "auc": round(auc(y_, s_), 4) if s_ is not None else "",
                             "p_vs_v1": round(mcnemar(y_, p_, base), 4) if system != "v1_crosswalk" else ""})
        extras[target] = {"y": y_all, "preds": preds_all, "scores": scores_all}
    return rows, extras


def ablations(cfg, pop: Population, ipc, bns, normalizer) -> list[dict]:
    """E11b: switch off one component of the drift model at a time."""
    rows = []
    for name, kw in (("full", {}), ("- renumbering", {"renumbering": False}), ("- punishment", {"use_punishment": False}),
                     ("- elements (mens rea, thresholds)", {"use_elements": False}),
                     ("- clause alignment", {"clause_alignment": False})):
        an = DriftAnalyzer.from_sources(cfg, ipc=ipc, bns=bns, normalizer=normalizer)
        for k, v in kw.items():
            setattr(an, k, v)
        reps = {b: an.report(b) for b in pop.sections}
        for target in ("S1", "S2"):
            y = [pop.y[target][b] for b in pop.sections]
            pred = [int(reps[b].drift_class in (DriftClass.SUBSTANTIVE, DriftClass.PUNISHMENT)) for b in pop.sections]
            m = binary_metrics(y, pred)
            rows.append({"target": target, "variant": name, **{k: round(v, 4) for k, v in m.items()},
                         "auc_carry_over": round(auc(y, [1 - reps[b].carry_over for b in pop.sections]), 4)})
    return rows


def punishment_detection(table: DriftTable, pop: Population) -> list[dict]:
    """E11c: our punishment_changed flag vs S2's 'punishment' category."""
    secs = [b for b in pop.sections if not table.sections[b].features.get("old_incomplete")]
    y = [int("punishment" in pop.s2.get(b, set())) for b in secs]
    p = [int(table.sections[b].features["punishment_changed"] > 0) for b in secs]
    m = binary_metrics(y, p)
    return [{"task": "punishment changed (vs S2)", **{k: round(v, 4) for k, v in m.items()}}]


def punishment_extraction(an: DriftAnalyzer, cfg=None) -> list[dict]:
    """E11d: extract() on the bare act vs on the source's punishment summary, per field."""
    summ = gold.bns_punishment_summaries(cfg)
    agree: dict[str, list[int]] = defaultdict(list)
    for b, prov in an.bns.items():
        s = summ.get(b, "")
        if not s or s.lower().startswith("not applicable"):
            continue
        a, g = pun.extract(prov.text), pun.extract(s)
        if g.empty:
            continue
        agree["death"].append(int(a.death == g.death))
        agree["life"].append(int(a.life == g.life))
        agree["fine"].append(int(a.fine == g.fine))
        agree["community_service"].append(int(a.community_service == g.community_service))
        if g.max_years is not None:
            agree["max_years"].append(int(a.max_years is not None and abs(a.max_years - g.max_years) < 1e-6))
        if g.min_years is not None:
            agree["min_years"].append(int(a.min_years is not None and abs(a.min_years - g.min_years) < 1e-6))
        if g.fine_max is not None:
            agree["fine_max"].append(int(a.fine_max is not None and abs(a.fine_max - g.fine_max) < 1e-6))
    return [{"field": k, "n": len(v), "agreement": round(sum(v) / len(v), 4)} for k, v in agree.items() if v]


def repeal_detection(an: DriftAnalyzer, table: DriftTable, cfg=None) -> list[dict]:
    """E11e: which IPC sections did the BNS drop? Gold: S1 Deleted."""
    deleted = gold.s1_deleted_ipc(cfg)
    pop = sorted(gold.s1_ipc_sections(cfg) & set(an.ipc))
    y = [int(s in deleted) for s in pop]
    pred = [int(s in table.repealed) for s in pop]
    best = {s: max((table.pairs[(s, b)].carry_over for b in an.successors.get(s, []) if (s, b) in table.pairs), default=None)
            for s in pop}
    score = [1.0 if s in table.repealed else 1 - (best[s] if best[s] is not None else 0.5) for s in pop]
    rows = [{"system": "drift: no successor (crosswalk + text search)", **{k: round(v, 4) for k, v in binary_metrics(y, pred).items()},
             "auc": round(auc(y, score), 4)}]
    pred_low = [int(s in table.repealed or (best[s] is not None and best[s] < 0.5)) for s in pop]
    rows.append({"system": "drift: no successor or carry-over < 0.5", **{k: round(v, 4) for k, v in binary_metrics(y, pred_low).items()},
                 "auc": round(auc(y, score), 4)})
    v1 = [int(not an.successors.get(s)) for s in pop]
    rows.append({"system": "v1: not in the crosswalk", **{k: round(v, 4) for k, v in binary_metrics(y, v1).items()}, "auc": ""})
    return rows


def carry_over_by_category(table: DriftTable, pop: Population) -> list[dict]:
    """E11f: mean carry-over of the section by its S2 category (most severe category wins)."""
    order = ["unchanged", "structure", "punishment", "substance"]
    groups: dict[str, list[float]] = defaultdict(list)
    for b in pop.sections:
        cats = pop.s2.get(b, {"unchanged"})
        cat = max((c for c in cats if c in order), key=order.index, default="unchanged")
        groups[cat].append(table.sections[b].carry_over)
    return [{"s2_category": c, "n": len(groups[c]), "mean_carry_over": round(float(np.mean(groups[c])), 4),
             "median_carry_over": round(float(np.median(groups[c])), 4)} for c in order if groups[c]]


def hazard_audit(an: DriftAnalyzer, table: DriftTable, normalizer, threshold: float = 0.5) -> list[dict]:
    """E12: offence ids of layer 1 that equate provisions whose carry-over is below `threshold`."""
    rows = []
    for oid, info in normalizer.offences.items():
        ipc = [re.sub(r"\(.*", "", s) for s in info.get("ipc", [])]
        bns = [re.sub(r"\(.*", "", s) for s in info.get("bns", [])]
        pairs = [(i, b, table.pairs[(i, b)]) for i in ipc for b in bns if (i, b) in table.pairs]
        if not pairs:
            continue
        i, b, worst = min(pairs, key=lambda x: x[2].carry_over)
        rep = table.sections.get(b)
        rows.append({"offence_id": oid, "label": info.get("label", ""), "ipc": ";".join(ipc), "bns": ";".join(bns),
                     "worst_pair": f"ipc:{i} -> bns:{b}", "carry_over": round(worst.carry_over, 4),
                     "class": worst.drift_class.value, "hazard": int(worst.carry_over < threshold),
                     "why": rep.reasons[1:3] if rep else ""})
    rows.sort(key=lambda r: r["carry_over"])
    return rows


__all__ = ["binary_metrics", "cohen_kappa", "auc", "mcnemar", "split_of", "population", "change_detection",
           "ablations", "punishment_detection", "punishment_extraction", "repeal_detection",
           "carry_over_by_category", "hazard_audit"]
