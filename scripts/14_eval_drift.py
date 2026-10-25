"""Run the drift experiments E11-E12 (eval/drift_eval.py) and draw their figures.  [owner: Gaurav — working]

Needs the bare acts and the two gold sources on disk (make fetch). Writes
    results/tables/drift_change_detection.csv   E11a, every system x target x split
    results/tables/drift_rule_tuning.csv        thresholds the dev half chose for drift_rules_dev
    results/tables/drift_lr_coefficients.csv    standardised weights of the learned detector
    results/tables/drift_source_agreement.csv   S1 vs S2 (how much the two gold sources agree)
    results/tables/drift_ablation.csv           E11b
    results/tables/drift_punishment.csv         E11c (change detection) + E11d (extraction)
    results/tables/drift_repeal.csv             E11e
    results/tables/drift_carry_over.csv         E11f
    results/tables/drift_hazard_offences.csv    E12
    results/tables/drift_classes.csv            how many sections fall in each class
    results/figures/drift_*.png

    python scripts/14_eval_drift.py
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from pathlib import Path

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.drift.model import DriftAnalyzer, DriftTable
from kanoon_bridge.drift.provisions import load_bns, load_ipc
from kanoon_bridge.eval import drift_eval as E
from kanoon_bridge.eval.plots import INK, INK_MUTED, GRID, PALETTE, _plt
from kanoon_bridge.text.version_norm import VersionNormalizer

SYSTEM_LABEL = {"v1_crosswalk": "v1 crosswalk relation", "edit_similarity": "Edit similarity (tuned)",
                "tfidf_cosine": "tf-idf cosine (tuned)", "drift_rules": "Drift rules (hand-set)",
                "drift_rules_subst": "Drift rules, substantive only", "drift_rules_dev": "Drift rules (dev-tuned)",
                "drift_lr": "Drift vector + logistic reg."}


def write(rows: list[dict], name: str) -> Path:
    path = project_path("results/tables") / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_change_detection(rows: list[dict], out: Path) -> None:
    plt = _plt()
    systems = ["v1_crosswalk", "edit_similarity", "tfidf_cosine", "drift_rules", "drift_rules_dev", "drift_lr"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    h = 0.36
    for k, target in enumerate(("S1", "S2")):
        vals, lo, hi = [], [], []
        for s in systems:
            r = next(r for r in rows if r["target"] == target and r["split"] == "test" and r["system"] == s)
            vals.append(r["f1_changed"])
            lo.append(r["f1_changed"] - r["f1_ci_low"])
            hi.append(r["f1_ci_high"] - r["f1_changed"])
        ys = [i + (k - 0.5) * h for i in range(len(systems))]
        label = "S1: indian-legal-mcp status" if target == "S1" else "S2: bns-study-platform notes"
        ax.barh(ys, vals, height=h - 0.04, color=PALETTE[k], label=label, xerr=[lo, hi],
                error_kw={"ecolor": INK_MUTED, "elinewidth": 0.8, "capsize": 2})
        for y, v, e in zip(ys, vals, hi):
            ax.text(v + e + 0.015, y, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(systems)), [SYSTEM_LABEL[s] for s in systems], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 of 'changed', test half (95% bootstrap CI)", color=INK_MUTED)
    ax.set_title("The drift vector finds the provisions the BNS changed", loc="left", color=INK, fontsize=11, pad=24)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_single(labels: list[str], vals: list[float], title: str, xlabel: str, out: Path, fmt: str = "{:.2f}",
               xmax: float | None = None) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(labels) + 1.2))
    ax.barh(range(len(labels)), vals, height=0.6, color=PALETTE[0])
    for i, v in enumerate(vals):
        ax.text(v + (xmax or max(vals)) * 0.01, i, fmt.format(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(labels)), labels, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, (xmax or max(vals)) * 1.12)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_ablation(rows: list[dict], out: Path) -> None:
    plt = _plt()
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    h = 0.36
    for k, target in enumerate(("S1", "S2")):
        vals = [next(r["f1_changed"] for r in rows if r["target"] == target and r["variant"] == v) for v in variants]
        ys = [i + (k - 0.5) * h for i in range(len(variants))]
        ax.barh(ys, vals, height=h - 0.04, color=PALETTE[k], label=target)
        for y, v in zip(ys, vals):
            ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(variants)), variants, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 of 'changed' (hand-set rules, all sections)", color=INK_MUTED)
    ax.set_title("Which parts of the drift vector carry the signal", loc="left", color=INK, fontsize=11, pad=24)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    t0 = time.perf_counter()
    ipc, bns = load_ipc(cfg, docs_path="/nonexistent"), load_bns(cfg)
    norm = VersionNormalizer.load(cfg)
    an = DriftAnalyzer.from_sources(cfg, ipc=ipc, bns=bns, normalizer=norm)
    table = DriftTable.build(an)
    pop = E.population(an, cfg)
    figs = project_path("results/figures")
    figs.mkdir(parents=True, exist_ok=True)

    rows, extras = E.change_detection(an, table, pop)
    write(rows, "drift_change_detection")
    write([{"target": t, **p} for t, p in extras["tuned"].items()], "drift_rule_tuning")
    write([{"target": t, "feature": f, "coefficient": round(c, 4)} for t, cs in extras["coef"].items() for f, c in cs],
          "drift_lr_coefficients")
    y1 = [pop.y["S1"][b] for b in pop.sections]
    y2 = [pop.y["S2"][b] for b in pop.sections]
    m = E.binary_metrics(y1, y2)
    write([{"n": len(y1), "s1_changed": sum(y1), "s2_changed": sum(y2), "agreement": round(m["accuracy"], 4),
            "kappa": round(m["kappa"], 4)}], "drift_source_agreement")
    abl = E.ablations(cfg, pop, ipc, bns, norm)
    write(abl, "drift_ablation")
    write(E.punishment_detection(table, pop) + [{"task": f"extraction: {r['field']}", "n": r["n"], "accuracy": r["agreement"]}
                                                for r in E.punishment_extraction(an, cfg)], "drift_punishment")
    write(E.repeal_detection(an, table, cfg), "drift_repeal")
    carry = E.carry_over_by_category(table, pop)
    write(carry, "drift_carry_over")
    hazard = E.hazard_audit(an, table, norm)
    write(hazard, "drift_hazard_offences")
    classes = Counter(r.drift_class.value for r in [*table.sections.values(), *table.repealed.values()])
    write([{"class": c, "sections": n} for c, n in classes.most_common()], "drift_classes")

    fig_change_detection(rows, figs / "drift_change_detection.png")
    fig_ablation(abl, figs / "drift_ablation.png")
    fig_single([c["s2_category"] for c in carry], [c["mean_carry_over"] for c in carry],
               "Carry-over falls once a provision changes", "mean carry-over of IPC-era precedent, by S2 category",
               figs / "drift_carry_over.png", xmax=1.0)
    order = ["identical", "cosmetic", "restructured", "punishment", "substantive", "new", "repealed", "unverified"]
    fig_single(order, [classes.get(c, 0) for c in order], "How the 358 BNS sections moved from the IPC",
               "sections (repealed: IPC sections with no successor)", figs / "drift_classes.png", fmt="{:.0f}")

    n_haz = sum(r["hazard"] for r in hazard)
    print(f"E11a ({len(pop.sections)} sections, test half):")
    for r in rows:
        if r["split"] == "test":
            print(f"  {r['target']} {r['system']:18s} F1 {r['f1_changed']:.3f}  macro-F1 {r['macro_f1']:.3f}  "
                  f"kappa {r['kappa']:.3f}  p vs v1 {r['p_vs_v1']}")
    print(f"S1 vs S2 kappa {m['kappa']:.3f}")
    print(f"E12: {n_haz} of {len(hazard)} offence ids equate an IPC section with a BNS section at carry-over < 0.5")
    print(f"done in {time.perf_counter() - t0:.1f}s; tables in results/tables/drift_*.csv, figures in results/figures/drift_*.png")


if __name__ == "__main__":
    main()
