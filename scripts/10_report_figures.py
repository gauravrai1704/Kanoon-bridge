"""Figures for the report, from results/tables/*.csv.  [owner: Gaurav]

    python scripts/10_report_figures.py        # -> docs/report/figures/*.png (and results/figures/)

Style: the baseline in one quiet grey, our system in one blue accent; values printed on the marks,
no gridlines; each title states the finding with its number.
"""

from __future__ import annotations

import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kanoon_bridge.config import load_config, project_path  # noqa: E402
from kanoon_bridge.eval.plots import read_table  # noqa: E402

GREY, BLUE, INK, MUTED = "#a3a8b4", "#2563c9", "#1d2033", "#5d6375"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#c9ccd4",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 14,
                     "xtick.color": MUTED, "ytick.color": INK})

HEADLINE = [  # set, metric, label, higher is better
    ("e1_ilpcsr", "MAP", "E1 precedents (IL-PCSR test) · MAP"),
    ("e1s_ilpcsr_statutes", "MAP", "E1 statutes (IL-PCSR test) · MAP"),
    ("e3_cross_version", "MAP", "E3 BNS-worded queries · MAP"),
    ("e4_multilingual", "MAP", "E4 en / hi / Hinglish · MAP"),
    ("e6_jurisdiction", "nDCG@10", "E6 jurisdiction (graded) · nDCG@10"),
    ("e7_temporal", "code_accuracy@1", "E7 right code at rank 1"),
    ("e2_collision", "MAP", "E2 colliding numbers · MAP"),
]


def _val(rows, set_, system, metric):
    for r in rows:
        if r["set"] == set_ and r["system"] == system and r.get(metric) not in (None, ""):
            return float(r[metric])
    return None


def fig_main(main, out):
    items = [(lab, _val(main, s, "baseline", m), _val(main, s, "full", m)) for s, m, lab in HEADLINE]
    items = [x for x in items if x[1] is not None and x[2] is not None]
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(items) + 1.3))
    for i, (lab, b, f) in enumerate(reversed(items)):
        ax.plot([b, f], [i, i], color="#d5d8df", lw=3, zorder=1, solid_capstyle="round")
        ax.scatter([b], [i], s=70, color=GREY, zorder=2)
        ax.scatter([f], [i], s=90, color=BLUE, zorder=3)
        close = abs(f - b) < 0.06
        ax.text(b, i + 0.22, f"{b:.2f}", ha="right" if close and b < f else ("left" if close else "center"),
                va="bottom", color=MUTED, fontsize=9.5)
        ax.text(f, i + 0.22, f"{f:.2f}", ha="left" if close and f > b else ("right" if close else "center"),
                va="bottom", color=BLUE, fontsize=9.5, fontweight="bold")
    ax.set_yticks(range(len(items)), [x[0] for x in reversed(items)])
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    e1 = items[0]
    wins = sum(1 for _, b, f in items if f > b)
    fig.suptitle(f"Better than BM25 on {wins} of {len(items)} sets; E1 precedent MAP {e1[1]:.2f} → {e1[2]:.2f}",
                 x=0.02, ha="left", fontsize=12.5, fontweight="bold")
    ax.scatter([], [], s=70, color=GREY, label="BM25, text as written (baseline)")
    ax.scatter([], [], s=90, color=BLUE, label="Kanoon-Bridge (full)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_ylim(-0.6, len(items) - 0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_ablation(rows, out):
    rows = [r for r in rows if r.get("MAP") not in (None, "")]
    names = [r["step"] for r in rows]
    maps = [float(r["MAP"]) for r in rows]
    gains = [maps[0]] + [maps[i] - maps[i - 1] for i in range(1, len(maps))]
    big = max(range(1, len(maps)), key=lambda i: gains[i]) if len(maps) > 1 else 0
    fig, ax = plt.subplots(figsize=(8.6, 0.45 * len(rows) + 1.3))
    for i, (n, m, r) in enumerate(zip(names, maps, rows)):
        y = len(rows) - 1 - i
        ax.barh(y, m, color=BLUE if i == big or i == len(rows) - 1 else GREY, height=0.62)
        note = "  not run (no GPU)" if "dense off" in (r.get("note") or "") and n in ("+dense", "+qpp") else ""
        d = f"  {m:.3f}" + (f"  ({gains[i]:+.3f})" if i and not note else "") + note
        ax.text(m, y, d, va="center", ha="left", fontsize=9.3, color=INK)
    ax.set_yticks(range(len(rows)), list(reversed(names)))
    ax.set_xlim(0, max(maps) * 1.45)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("MAP on IL-PCSR test (627 queries)", color=MUTED)
    ax.set_title(f"E1 ablation: trigrams add {gains[big]:+.3f} MAP, learning to rank {gains[-1]:+.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_e3(main, out):
    vals = {(sys, s): _val(main, s, sys, "MAP") for sys in ("baseline", "full") for s in ("e3_control", "e3_cross_version")}
    if None in vals.values():
        return False
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    xs = [0, 1, 3, 4]
    keys = [("baseline", "e3_control"), ("baseline", "e3_cross_version"), ("full", "e3_control"), ("full", "e3_cross_version")]
    cols = [GREY, GREY, BLUE, BLUE]
    alphas = [1.0, 0.55, 1.0, 0.55]
    for x, k, c, a in zip(xs, keys, cols, alphas):
        ax.bar(x, vals[k], color=c, alpha=a, width=0.85)
        ax.text(x, vals[k] + 0.01, f"{vals[k]:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
        ax.text(x, -0.035, "IPC wording" if k[1] == "e3_control" else "BNS wording", ha="center", va="top", fontsize=9, color=MUTED)
    ax.text(0.5, -0.11, "BM25, text as written", ha="center", va="top", fontsize=10, color=INK, fontweight="bold")
    ax.text(3.5, -0.11, "Kanoon-Bridge", ha="center", va="top", fontsize=10, color=BLUE, fontweight="bold")
    ax.set_xticks([])
    ax.set_ylim(0, max(vals.values()) * 1.25)
    ax.set_ylabel("MAP", color=MUTED)
    drop_b = vals[("baseline", "e3_control")] - vals[("baseline", "e3_cross_version")]
    gap_f = vals[("full", "e3_control")] - vals[("full", "e3_cross_version")]
    ax.set_title(f"E3: asking in BNS numbers costs BM25 {drop_b:.2f} MAP; Kanoon-Bridge loses {gap_f:.2f}")
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_language(main, lang, out):
    steps = [r for r in lang if r["step"] in ("none", "+translit", "+lexicon", "+phonetic")]
    if not steps:
        return False
    langs = [("en", "English"), ("hi", "Hindi (Devanagari)"), ("hinglish", "Hinglish (Roman)")]
    shades = [0.25, 0.45, 0.75, 1.0]
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    w = 0.8
    for li, (code, name) in enumerate(langs):
        for si, r in enumerate(steps):
            v = float(r.get(f"P@5_{code}") or 0)
            x = li * (len(steps) + 1.2) + si
            ax.bar(x, v, width=w, color=BLUE, alpha=shades[si], label=r["step"] if li == 0 else None)
            ax.text(x, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks([li * (len(steps) + 1.2) + (len(steps) - 1) / 2 for li in range(3)], [n for _, n in langs])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("P@5 (E4, 20 needs per language)", color=MUTED)
    ax.set_ylim(0, 0.3)
    ax.legend(title="analyser steps switched on", frameon=False, fontsize=8.5, title_fontsize=8.5, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, 1.02))
    hi = [float(r.get("P@5_hi") or 0) for r in steps]
    fig.suptitle(f"E4: Hindi P@5 {hi[0]:.2f} → {hi[-1]:.2f}, level with English, once the legal lexicon is on",
                 x=0.02, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return True


def main() -> None:
    ev = load_config("eval.yaml")
    tables = project_path(ev.outputs.tables)
    out = project_path("docs/report/figures")
    out.mkdir(parents=True, exist_ok=True)
    main_rows = read_table(tables / "main_results.csv")
    made = []
    fig_main(main_rows, out / "fig_main.png")
    made.append("fig_main.png")
    if (tables / "ablation_e1_ilpcsr.csv").exists():
        fig_ablation(read_table(tables / "ablation_e1_ilpcsr.csv"), out / "fig_ablation.png")
        made.append("fig_ablation.png")
    if fig_e3(main_rows, out / "fig_e3.png"):
        made.append("fig_e3.png")
    lang = read_table(tables / "language_e4.csv") if (tables / "language_e4.csv").exists() else []
    if fig_language(main_rows, lang, out / "fig_language.png"):
        made.append("fig_language.png")
    figs = project_path(ev.outputs.figures)
    for m in made:
        shutil.copy(out / m, figs / f"report_{m}")
    print("wrote", ", ".join(made), "->", out)


if __name__ == "__main__":
    main()
