"""Run E14-E16 (eval/layers_eval.py): agent v2 and RAG v2 on the open corpus.  [owner: Gaurav — working]

Needs the open corpus (make open: all six codes from India Code + the BNS, synthetic precedents)
and the bns-study-platform checkout (gold punishment summaries). No IL-PCSR, no API key.

    results/tables/layers_punishment_qa.csv     E14
    results/tables/layers_error_injection.csv   E15 (base answers: RAG v2 and RAG v1)
    results/tables/layers_agent_statutes.csv    E16
    results/figures/layers_error_injection.png
    results/figures/layers_agent_statutes.png

    python scripts/17_eval_layers.py
    python scripts/17_eval_layers.py --limit 60      # quick run
"""

from __future__ import annotations

import argparse
import copy
import csv
import time

from kanoon_bridge.agent.research import ResearchAgent, ResearchAgentV2
from kanoon_bridge.config import project_path
from kanoon_bridge.eval import layers_eval as L
from kanoon_bridge.eval.open_eval import score_set
from kanoon_bridge.eval.plots import INK, INK_MUTED, GRID, PALETTE, _plt
from kanoon_bridge.rag.answer import RagPipeline
from kanoon_bridge.search import SearchEngine, SearchOptions


def write(rows: list[dict], name: str) -> None:
    path = project_path("results/tables") / f"{name}.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _style(ax, axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_injection(rows: list[dict], out) -> None:
    plt = _plt()
    rows = [r for r in rows if r["base"] == "RAG v2" and not r["error"].startswith("none")]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    h = 0.36
    labels = [r["error"].replace("_", " ") for r in rows]
    for k, (key, name) in enumerate((("v1_detected", "v1 checks (cosine + code)"), ("v2_detected", "v2 claim verification"))):
        vals = [float(r[key]) for r in rows]
        ys = [i + (k - 0.5) * h for i in range(len(rows))]
        ax.barh(ys, vals, height=h - 0.04, color=PALETTE[k], label=name)
        for y, v in zip(ys, vals):
            ax.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), labels, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("share of injected errors flagged", color=INK_MUTED)
    ax.set_title("Claim verification catches the errors cosine support misses", loc="left", color=INK,
                 fontsize=11, pad=24)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_agent(rows: list[dict], out) -> None:
    plt = _plt()
    systems = ["core", "agent v1", "agent v2"]
    sets = [("e2_collision", "expected@1", "E2 collisions"), ("e7_temporal", "both@1", "E7 penal"),
            ("e9_procedure_temporal", "both@1", "E9 procedure")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.26
    for k, s in enumerate(systems):
        vals = [next(float(r[m]) for r in rows if r["set"] == name and r["system"] == s) for name, m, _ in sets]
        xs = [i + (k - 1) * w for i in range(len(sets))]
        ax.bar(xs, vals, width=w - 0.03, color=PALETTE[k], label=s)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(sets)), [lab for _, _, lab in sets], color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("right provision of the right code at rank 1", color=INK_MUTED, fontsize=8)
    ax.set_title("The verified agent finds the right provision more often", loc="left", color=INK, fontsize=11, pad=24)
    _style(ax, "y")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()
    engine = SearchEngine.load()
    rag = RagPipeline.load(engine)
    v2 = ResearchAgentV2.load(engine)
    figs = project_path("results/figures")

    # E14 + E15
    qs = L.punishment_questions(limit=args.limit)
    qa, answers = L.punishment_qa(engine, rag, v2, qs)
    write(qa, "layers_punishment_qa")
    inj = []
    for base, system in (("RAG v2", "core + RAG v2"), ("RAG v1", "core + RAG v1")):
        inj += [{"base": base, **r} for r in L.error_injection(rag, answers[system])]
    write(inj, "layers_error_injection")
    fig_injection(inj, figs / "layers_error_injection.png")

    # E16
    def agent(cls=ResearchAgentV2, **over):
        cfg = copy.deepcopy(engine.cfg)
        for k, v in over.items():
            cfg.agent[k] = v
        a = cls.load(engine, cfg=cfg)
        return lambda q: a.run(q, SearchOptions.full_drift())

    systems = {"baseline BM25": lambda q: engine.search(q, SearchOptions.baseline()),
               "core": lambda q: engine.search(q, SearchOptions.full_drift()),
               "agent v1": agent(ResearchAgent),
               "agent v2": agent(),
               "v2 - verification": agent(verify_family=0, verify_force=0, verify_heading=0, verify_general=0),
               "v2 - citation votes": agent(vote_weight=0),
               "v2, votes ungated": agent(vote_gate=0),
               "v2 - family search": agent(family_weight=0)}
    rows = []
    for name in ("e2_collision", "e7_temporal", "e9_procedure_temporal"):
        for s, fn in systems.items():
            rows.append({"set": name, "system": s, **score_set(name, fn, engine, args.limit)})
    write(rows, "layers_agent_statutes")
    fig_agent(rows, figs / "layers_agent_statutes.png")

    for r in qa:
        print(f"E14 {r['system']:20s} statute@1 {r['statute@1']:.3f}  punishment_ok {r['punishment_ok']:.3f}  "
              f"(given statute {r['punishment_ok|statute']:.3f})  corrections/answer {r['corrections/answer']:.3f}")
    for r in inj:
        print(f"E15 [{r['base']}] {r['error']:38s} v1 {r['v1_detected']}  v2 {r['v2_detected']}  (n={r['n']})")
    for r in rows:
        print(f"E16 {r['set']:22s} {r['system']:20s} " + "  ".join(f"{k} {v}" for k, v in r.items() if k not in ("set", "system")))
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
