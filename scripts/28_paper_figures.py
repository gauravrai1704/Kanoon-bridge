"""Figures for the papers, from results/tables/*.csv, as vector PDFs.  [owner: Gaurav — working]

    python scripts/28_paper_figures.py
    -> docs/paper_acm/figures/*.pdf and docs/paper/figures/*.pdf

One style for every chart: single-column width (3.33 in), a fixed categorical order (blue, orange,
aqua, yellow; validated for colour-blind separation), thin bars with a gap, direct labels, a
recessive grid. Every number comes from a CSV written by an experiment script.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kanoon_bridge.config import project_path  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, LIGHT, INK, MUTED = "#8c8a83", "#d9d7d0", "#0b0b0b", "#52514e"
W = 3.33

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": GREY,
    "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": INK, "axes.labelcolor": MUTED,
    "xtick.major.width": 0.5, "ytick.major.width": 0, "legend.frameon": False, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

TABLES = project_path("results/tables")
OUT = project_path("docs/paper_acm/figures")


def rows(name):
    with open(TABLES / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=LIGHT, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    path = OUT / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------------- Fig: citation shift
def citation_shift():
    r = rows("citation_shift.csv")
    months = [x["month"] for x in r]
    ipc = [float(x["ipc_share_of_judgments"]) for x in r]
    bns = [float(x["bns_share_of_judgments"]) for x in r]
    xs = range(len(months))
    fig, ax = plt.subplots(figsize=(W, 1.55))
    ax.plot(xs, ipc, color=ORANGE, lw=1.4, marker="o", ms=2.2)
    ax.plot(xs, bns, color=BLUE, lw=1.4, marker="o", ms=2.2)
    cut = months.index("2024-07")
    ax.axvline(cut - 0.5, color=GREY, lw=0.7, ls=(0, (3, 2)))
    ax.text(cut - 0.3, 0.93, "BNS in force\n1 Jul 2024", color=MUTED, fontsize=6, va="top")
    ax.text(1.5, 0.93, "cite the IPC", color=ORANGE, fontsize=6.5, va="top")
    ax.text(1.5, 0.05, "cite the BNS", color=BLUE, fontsize=6.5, va="bottom")
    ticks = [i for i, m in enumerate(months) if m.endswith(("-01", "-07"))]
    ax.set_xticks(ticks, [months[i].replace("-01", "\nJan").replace("-07", "\nJul") for i in ticks])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("share of judgments")
    ax.set_xlim(-0.5, len(months) - 0.5)
    grid(ax, "y")
    return save(fig, "citation_shift")


# --------------------------------------------------------------------------- Fig: E4 languages
def language_e4():
    r = [x for x in rows("language_e4.csv") if not x.get("note")]          # "+dense" was off: same as previous
    steps = [x["step"].replace("none", "no typed-\nquery steps").replace("+", "+ ") for x in r]
    langs = [("P@5_en", "English", BLUE), ("P@5_hi", "Hindi (Devanagari)", ORANGE), ("P@5_hinglish", "Hinglish", AQUA)]
    fig, ax = plt.subplots(figsize=(W, 1.6))
    width = 0.25
    for j, (key, label, color) in enumerate(langs):
        vals = [float(x[key]) for x in r]
        ax.bar([i + (j - 1) * (width + 0.02) for i in range(len(r))], vals, width, color=color, label=label)
    ax.set_xticks(range(len(r)), steps)
    ax.set_ylabel("P@5")
    ax.set_ylim(0, 0.2)
    for i, x in enumerate(r):
        ax.text(i, 0.185, f"MAP {float(x['MAP']):.2f}", ha="center", fontsize=6, color=MUTED)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.22), handlelength=1, columnspacing=1)
    grid(ax, "y")
    return save(fig, "language_e4")


# --------------------------------------------------------------------------- Fig: drift ablation
def drift_ablation():
    r = [x for x in rows("drift_ablation.csv") if x["family"] == "ipc"]
    variants = list(dict.fromkeys(x["variant"] for x in r))
    labels = {"full": "full drift vector", "- renumbering": "− renumbering", "- punishment": "− punishment",
              "- elements (mens rea, thresholds)": "− elements (mens rea, thresholds)",
              "- clause alignment": "− clause alignment"}
    fig, ax = plt.subplots(figsize=(W, 1.7))
    h = 0.36
    for j, (target, color, name) in enumerate((("S1", BLUE, "S1 mapping"), ("S2", ORANGE, "S2 study notes"))):
        vals = [next(float(x["f1_changed"]) for x in r if x["variant"] == v and x["target"] == target) for v in variants]
        ys = [i + (j - 0.5) * (h + 0.04) for i in range(len(variants))]
        ax.barh(ys, vals, h, color=color, label=name)
        for y, v in zip(ys, vals):
            ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=6, color=INK)
    ax.set_yticks(range(len(variants)), [labels.get(v, v) for v in variants])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 of “changed” (hand-set rules, all 346 IPC sections)")
    ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0))
    grid(ax)
    return save(fig, "drift_ablation")


# --------------------------------------------------------------------------- Fig: error injection
def error_injection():
    r = [x for x in rows("layers_error_injection.csv") if x["base"] == "RAG v2" and not x["error"].startswith("none")]
    names = {"years": "term of years changed", "fine": "fine ×10", "negation": "negation flipped",
             "wrong_code": "old-code section", "section": "section number changed", "unsupported": "unsupported sentence"}
    fig, ax = plt.subplots(figsize=(W, 1.75))
    h = 0.36
    for j, (key, color, name) in enumerate((("v1_detected", GREY, "cosine support + code check (v1)"),
                                            ("v2_detected", BLUE, "claim verification (v2)"))):
        vals = [float(x[key]) for x in r]
        ys = [i + (j - 0.5) * (h + 0.04) for i in range(len(r))]
        ax.barh(ys, vals, h, color=color, label=name)
        for y, v in zip(ys, vals):
            ax.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=6, color=INK)
    ax.set_yticks(range(len(r)), [names.get(x["error"], x["error"]) for x in r])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of injected errors flagged")
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(-0.02, 1.0))
    grid(ax)
    return save(fig, "error_injection")


# --------------------------------------------------------------------------- Fig: E20 outcomes
def e20_outcomes():
    r = {x["condition"]: x for x in rows("misgrounding_llama3_2_3b.csv") if x["set"] == "all"}
    conds = [("closed-book", "closed-book"), ("naive RAG", "naive RAG"), ("RAG v1", "RAG v1 (date prompt)"),
             ("RAG v2", "RAG v2 (verified)"), ("structured", "structured (no LLM)")]
    parts = [("right@1", "right provision", BLUE), ("other", "other section", LIGHT),
             ("no_section", "no section", GREY), ("misgrounded@1", "code not in force", ORANGE)]
    fig, ax = plt.subplots(figsize=(W, 1.65))
    for i, (key, label) in enumerate(conds):
        x = r[key]
        vals = {k: float(x[k]) for k in ("right@1", "no_section", "misgrounded@1")}
        vals["other"] = max(0.0, 1 - sum(vals.values()))
        left = 0.0
        for k, _, color in parts:
            v = vals[k]
            ax.barh(i, v, 0.62, left=left, color=color, edgecolor="white", linewidth=0.8)
            if v >= 0.07:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=6,
                        color="white" if color in (BLUE, ORANGE, GREY) else INK)
            left += v
        if vals["misgrounded@1"] < 0.07:
            ax.text(1.01, i, f"{vals['misgrounded@1']:.1%}".replace(".0%", "%"), va="center", fontsize=6, color=ORANGE)
    ax.set_yticks(range(len(conds)), [c[1] for c in conds])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of 300 answers (first section named)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts]
    ax.legend(handles, [p[1] for p in parts], ncol=4, loc="lower left", bbox_to_anchor=(-0.32, 1.0),
              handlelength=0.9, columnspacing=0.8)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    return save(fig, "e20_outcomes")


# --------------------------------------------------------------------------- Fig: E25 ablation
def _kind(r):
    if r["mislabel"]:
        return "mislabel"
    if r["first_section_in_context"]:
        return "copied"
    if r.get("question_number") and r["set"].startswith("e2"):
        return "question number"
    return "recalled"


def e25_ablation():
    with open(TABLES / "context_ablation_llama3_2_3b_rows.jsonl", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    kinds = ["copied", "question number", "mislabel", "recalled"]
    labels = {"copied": "copied from context", "question number": "bare number read as IPC",
              "mislabel": "right number, IPC's name", "recalled": "recalled"}
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.45), sharey=True)
    for ax, cond, title in zip(axes, ("naive RAG", "RAG v1"), ("naive RAG", "RAG v1 (date prompt)")):
        rs = [r for r in data if r["condition"] == cond and r.get("control_misgrounded@1")]
        fixed = Counter(_kind(r) for r in rs if not r["ablated_misgrounded@1"])
        kept = Counter(_kind(r) for r in rs if r["ablated_misgrounded@1"])
        ys = range(len(kinds))
        a = [fixed[k] for k in kinds]
        b = [kept[k] for k in kinds]
        ax.barh(ys, a, 0.6, color=BLUE, edgecolor="white", linewidth=0.8, label="fixed by removal")
        ax.barh(ys, b, 0.6, left=a, color=ORANGE, edgecolor="white", linewidth=0.8, label="still misgrounded")
        for y, (u, v) in enumerate(zip(a, b)):
            ax.text(u + v + 0.6, y, f"{u}/{u + v}", va="center", fontsize=6, color=INK)
        n, nf = len(rs), sum(a)
        ax.set_title(f"{title}\n{nf} of {n} fixed", loc="left", color=INK, fontsize=6.8)
        ax.set_xlim(0, 30)
        ax.set_xticks([0, 10, 20])
        grid(ax)
    axes[0].set_yticks(range(len(kinds)), [labels[k] for k in kinds])
    axes[0].invert_yaxis()
    fig.supxlabel("misgrounded answers (label: fixed / all)", fontsize=6.8, color=MUTED, y=-0.1)
    axes[1].legend(loc="lower right", fontsize=6)
    fig.subplots_adjust(wspace=0.08)
    return save(fig, "e25_ablation")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    made = [citation_shift(), language_e4(), drift_ablation(), error_injection(), e20_outcomes(), e25_ablation()]
    acl = project_path("docs/paper/figures")
    acl.mkdir(parents=True, exist_ok=True)
    for p in made:
        if p.stem in ("citation_shift", "e20_outcomes", "e25_ablation"):
            shutil.copy(p, acl / p.name)


if __name__ == "__main__":
    main()
