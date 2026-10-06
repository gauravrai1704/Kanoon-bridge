"""Charts for the report and video, saved to results/figures/.  [owner: Gaurav — working]

One finding per chart; the title states it. Built from results/tables/*.csv, so they can be
regenerated without re-running retrieval:

    ablation_<set>.png    MAP and F1@k per ladder step
    main_results.png      baseline vs full system on each set's headline metric
    agent_<set>.png       core vs agent (RRF) vs agent (CombSUM)
    language_e4.png       P@5 by query language, per language-ablation step
    efficiency.png        latency (left) and Recall@20 vs exhaustive (right) per scoring mode
    typos.png             known-item MRR: clean vs misspelled vs misspelled + spelling correction
    rag.png               supported-sentence rate: RAG vs closed-book

    python -m kanoon_bridge.eval.plots

Style: one y-axis per panel, thin bars, a legend whenever there are 2+ series, text in ink
colours (never the series colour), recessive grid, fixed categorical order of colours.
"""

from __future__ import annotations

import csv
from pathlib import Path

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"

HEADLINE = {            # set -> (metric, higher is better?)
    "e1_ilpcsr": ("MAP", True), "e1s_ilpcsr_statutes": ("MAP", True), "e2_collision": ("wrong_hit@10", False),
    "e3_cross_version": ("MAP", True), "e4_multilingual": ("P@5", True), "e6_jurisdiction": ("binding_share@5", True),
    "e7_temporal": ("code_accuracy@1", True),
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
    return plt


def read_table(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bars(ax, categories: list[str], series: dict[str, list[float | None]], label_values: bool = True) -> None:
    n = max(1, len(series))
    width = 0.8 / n
    for i, (name, vals) in enumerate(series.items()):
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(categories))]
        ys = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(xs, ys, width * 0.92, color=PALETTE[i % len(PALETTE)], label=name, zorder=2)
        # selective labels: every bar when the chart is small, else only the first series
        if label_values and (len(series) * len(categories) <= 12 or i == 0):
            for b, v in zip(bars, vals):
                if v is not None:
                    ax.annotate(f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                                ha="center", va="bottom", fontsize=7, color=INK_MUTED, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=20 if max(map(len, categories), default=0) > 8 else 0, ha="right" if
                       max(map(len, categories), default=0) > 8 else "center")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8, ncol=len(series), loc="lower right", bbox_to_anchor=(1.0, 1.0))


def bar_chart(rows: list[dict], x: str, y: str | list[str], title: str, out: str | Path,
              ylabel: str = "", ylim: tuple[float, float] | None = (0, 1)) -> Path | None:
    """Bar chart from table rows: one bar group per row[x], one series per metric in `y`."""
    ys = [y] if isinstance(y, str) else y
    ys = [m for m in ys if any(_num(r.get(m)) is not None for r in rows)]
    if not rows or not ys:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(rows) + 2), 3.6))
    _bars(ax, [r[x] for r in rows], {m: [_num(r.get(m)) for r in rows] for m in ys})
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=22 if len(ys) > 1 else 8)
    ax.set_ylabel(ylabel or (ys[0] if len(ys) == 1 else "score"))
    if ylim:
        top = max((_num(r.get(m)) or 0 for r in rows for m in ys), default=1)
        ax.set_ylim(ylim[0], max(ylim[1], top) * 1.1)       # headroom for value labels
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _delta_title(rows, metric, first, last, what):
    a, b = _num(rows[0].get(metric)), _num(rows[-1].get(metric))
    if a is None or b is None:
        return what
    return f"{what}: {metric} {a:.3f} -> {b:.3f} ({first} -> {last})"


def make_all(tables: str | Path | None = None, figures: str | Path | None = None) -> list[Path]:
    from kanoon_bridge.config import load_config, project_path

    ev = load_config("eval.yaml")
    tables = Path(tables) if tables else project_path(ev.outputs.tables)
    figures = Path(figures) if figures else project_path(ev.outputs.figures)
    made: list[Path] = []

    for path in sorted(tables.glob("ablation_*.csv")):
        rows = read_table(path)
        if rows:
            f1 = "F1@k_val" if "F1@k_val" in rows[0] else "F1@10"
            name = path.stem.removeprefix("ablation_")
            made.append(bar_chart(rows, "step", ["MAP", f1, "MRR"], _delta_title(rows, "MAP", rows[0]["step"],
                                  rows[-1]["step"], f"Ablation on {name}"), figures / f"{path.stem}.png"))

    main = read_table(tables / "main_results.csv")
    if main:
        sets = list(dict.fromkeys(r["set"] for r in main))
        cats, base, full, labels = [], [], [], []
        for s in sets:
            metric, _ = HEADLINE.get(s, ("MAP", True))
            by = {r["system"]: r for r in main if r["set"] == s}
            if metric not in next(iter(by.values())) or _num(next(iter(by.values())).get(metric)) is None:
                metric = "MAP"
            cats.append(f"{s}\n{metric}")
            base.append(_num(by.get("baseline", {}).get(metric)))
            full.append(_num(by.get("full", {}).get(metric)))
        plt = _plt()
        fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(cats) + 2), 3.8))
        _bars(ax, cats, {"BM25 baseline": base, "Kanoon-Bridge (full)": full})
        ax.set_xticklabels(cats, rotation=0, ha="center", fontsize=8)
        ax.set_title("Baseline vs full system, headline metric per test set (wrong_hit: lower is better)",
                     loc="left", fontsize=10, pad=22)
        ax.set_ylim(0, 1.1)
        fig.tight_layout()
        made.append(figures / "main_results.png")
        figures.mkdir(parents=True, exist_ok=True)
        fig.savefig(made[-1], dpi=160)
        plt.close(fig)

    for path in sorted(tables.glob("agent_*.csv")):
        rows = read_table(path)
        if rows:
            made.append(bar_chart(rows, "system", ["MAP", "MRR", "nDCG@10"],
                                  _delta_title(rows, "MAP", rows[0]["system"], rows[-1]["system"],
                                               f"Layer 2 on {path.stem.removeprefix('agent_')}"),
                                  figures / f"{path.stem}.png"))

    ty = read_table(tables / "typos.csv")
    if ty:
        a, b = (_num(r.get("MRR@10")) for r in (ty[1], ty[2])) if len(ty) > 2 else (None, None)
        title = ("Typo robustness (known-item statute search)" if a is None else
                 f"Spelling correction: MRR@10 {a:.2f} -> {b:.2f} on misspelled queries")
        made.append(bar_chart(ty, "condition", ["MRR@10", "Success@1", "Success@10"], title, figures / "typos.png"))

    lang = read_table(tables / "language_e4.csv")
    if lang:
        cols = [c for c in ("P@5_en", "P@5_hi", "P@5_hinglish") if c in lang[0]]
        made.append(bar_chart(lang, "step", cols or ["P@5"], "E4: P@5 by query language as analyzer steps switch on",
                              figures / "language_e4.png", ylabel="P@5"))

    eff = read_table(tables / "efficiency.csv")
    if eff:
        plt = _plt()
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4))
        modes = [r["mode"] for r in eff]
        _bars(a1, modes, {"median": [_num(r["latency_ms_median"]) for r in eff],
                          "p95": [_num(r["latency_ms_p95"]) for r in eff]})
        a1.set_ylabel("latency (ms per query)")
        a1.set_title("Latency", loc="left", fontsize=10, pad=22)
        _bars(a2, modes, {"recall": [_num(r["recall@20_vs_exhaustive"]) for r in eff]})
        a2.set_ylim(0, 1.15)
        a1.set_ylim(0, max((_num(r["latency_ms_p95"]) or 0) for r in eff) * 1.15 or 1)
        a2.set_title("Top-20 kept (Recall@20 vs exhaustive)", loc="left", fontsize=10, pad=22)
        fig.suptitle("Tiered and champion-list scoring: speed vs top-20 overlap", x=0.02, ha="left", fontsize=11)
        fig.tight_layout()
        made.append(figures / "efficiency.png")
        fig.savefig(made[-1], dpi=160)
        plt.close(fig)

    rag = [r for r in read_table(tables / "rag.csv") if r.get("id") == "MEAN"]
    if rag:
        r = rag[0]
        cats, vals = ["RAG (" + r.get("generator", "") + ")"], [_num(r.get("supported_rate"))]
        if _num(r.get("closed_book_supported_rate")) is not None:
            cats.append("closed-book LLM")
            vals.append(_num(r["closed_book_supported_rate"]))
        made.append(bar_chart([{"system": c, "supported": v} for c, v in zip(cats, vals)], "system", "supported",
                              "Share of answer sentences supported by a retrieved source", figures / "rag.png",
                              ylabel="supported-sentence rate"))
    made = [m for m in made if m]
    for m in made:
        print("  figure", m)
    return made


if __name__ == "__main__":
    make_all()
