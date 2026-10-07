"""Statistical significance for every comparison we report.  [owner: Gaurav — working]

    python -m kanoon_bridge.eval.significance --set e1_ilpcsr --a baseline --b full
    python -m kanoon_bridge.eval.significance --set e1_ilpcsr --ladder     # each ablation step vs the previous
    python -m kanoon_bridge.eval.significance --e3                          # BNS wording vs IPC wording (E3)

Method (Smucker, Allan & Carterette, "A comparison of statistical significance tests for
information retrieval evaluation", CIKM 2007, recommend the randomization test for IR):

    per-query scores     AP, nDCG@10, P@10, RR for both systems on the same judged queries
                         (a query a run does not return counts as 0)
    paired randomization two-sided sign-flip test on the per-query differences, 10,000
                         permutations, p = (#{|mean*| >= |observed|} + 1) / (B + 1)
    bootstrap CI         95% percentile interval of the mean difference, 10,000 resamples of queries
    multiple testing     Holm-Bonferroni across all comparisons made in one call (each ablation
                         step is a separate hypothesis), reported as p_holm

Reads the TREC run files the evaluation already writes (results/runs/), so nothing is re-run.
Output: results/tables/significance.csv. In the report, mark a difference significant only when
p_holm < 0.05, and give the CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kanoon_bridge.config import Config, load_config, project_path

METRICS = ("AP", "nDCG@10", "P@10", "RR")


def per_query(ranking: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """metric -> {query: score}, over queries with at least one relevant document."""
    from kanoon_bridge.eval.metrics import average_precision, ndcg_at_k, precision_at_k, reciprocal_rank

    out = {m: {} for m in METRICS}
    for q, rels in qrels.items():
        if not any(g > 0 for g in rels.values()):
            continue
        r = ranking.get(q, [])
        out["AP"][q] = average_precision(r, rels)
        out["nDCG@10"][q] = ndcg_at_k(r, rels, 10)
        out["P@10"][q] = precision_at_k(r, rels, 10)
        out["RR"][q] = reciprocal_rank(r, rels)
    return out


def randomization_test(diffs: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided paired sign-flip randomization test on the mean difference."""
    if len(diffs) == 0 or not np.any(diffs):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())
    signs = rng.choice((-1.0, 1.0), size=(n_perm, len(diffs)))
    perm = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(perm >= obs - 1e-12) + 1) / (n_perm + 1))


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    if len(diffs) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as given)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def compare(scores_a: dict[str, float], scores_b: dict[str, float], n_perm: int = 10000) -> dict:
    """Paired comparison over the queries both have (B - A)."""
    common = sorted(set(scores_a) & set(scores_b))
    a = np.array([scores_a[q] for q in common])
    b = np.array([scores_b[q] for q in common])
    diffs = b - a
    lo, hi = bootstrap_ci(diffs)
    return {"queries": len(common), "mean_a": round(float(a.mean()), 4) if len(a) else 0.0,
            "mean_b": round(float(b.mean()), 4) if len(b) else 0.0,
            "diff": round(float(diffs.mean()), 4) if len(diffs) else 0.0,
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "wins": int((diffs > 0).sum()), "losses": int((diffs < 0).sum()),
            "p": round(randomization_test(diffs, n_perm), 5)}


# --------------------------------------------------------------------------- run files -> rows


def _qrels_for(name: str, ev: Config) -> dict[str, dict[str, int]]:
    from kanoon_bridge.eval.run_eval import load_test_set

    ts = load_test_set(name, ev=ev)
    qrels = ts.qrels
    if name.startswith("e6"):
        from kanoon_bridge.eval.qrels import jurisdiction_grades
        from kanoon_bridge.index import store

        qrels = jurisdiction_grades(qrels, store.load("facets").metas, {q.query_id: q.state for q in ts.queries})
    return qrels


def _run(path: Path) -> dict[str, list[str]]:
    from kanoon_bridge.eval.run_eval import read_run

    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the evaluation first (make eval)")
    return read_run(path)


def pairs_rows(name: str, pairs: list[tuple[str, Path, str, Path]], qrels, metrics=METRICS, n_perm=10000) -> list[dict]:
    rows = []
    for label_a, path_a, label_b, path_b in pairs:
        sa, sb = per_query(_run(path_a), qrels), per_query(_run(path_b), qrels)
        for m in metrics:
            rows.append({"set": name, "a": label_a, "b": label_b, "metric": m, **compare(sa[m], sb[m], n_perm)})
    return rows


def e3_rows(runs_dir: Path, ev: Config, system: str = "full", metrics=METRICS, n_perm=10000) -> list[dict]:
    """Same questions, IPC wording (control) vs BNS wording: is the version gap closed?"""
    rows = []
    for sys_name in dict.fromkeys(("baseline", system)):
        qa = _qrels_for("e3_control", ev)
        qb = _qrels_for("e3_cross_version", ev)
        sa = per_query(_run(runs_dir / f"e3_control.{sys_name}.run"), qa)
        sb = per_query(_run(runs_dir / f"e3_cross_version.{sys_name}.run"), qb)
        for m in metrics:
            a = {k.split("-", 1)[1]: v for k, v in sa[m].items()}         # E3C-<id> / E3-<id> -> <id>
            b = {k.split("-", 1)[1]: v for k, v in sb[m].items()}
            rows.append({"set": "e3 (IPC vs BNS wording)", "a": f"{sys_name}: IPC wording", "b": f"{sys_name}: BNS wording",
                         "metric": m, **compare(a, b, n_perm)})
    return rows


def finish(rows: list[dict], out: Path | None) -> list[dict]:
    adj = holm([r["p"] for r in rows])
    for r, p in zip(rows, adj):
        r["p_holm"] = round(p, 5)
        r["significant"] = "yes" if p < 0.05 else "no"
    if out is not None:
        from kanoon_bridge.eval.ablation import write_table

        write_table(rows, out)
    return rows


def run_all(ev: Config | None = None, sets: list[str] | None = None, ladder_sets=("e1_ilpcsr",), n_perm: int = 10000) -> list[dict]:
    """Every comparison the report makes: baseline vs full per set, the ablation ladder, E3 gap."""
    ev = ev or load_config("eval.yaml")
    runs = project_path(ev.outputs.runs)
    rows: list[dict] = []
    for name in sets or list(ev.test_sets):
        a, b = runs / f"{name}.baseline.run", runs / f"{name}.full.run"
        if a.exists() and b.exists():
            pairs = [("baseline", a, "full", b)]
            il = runs / f"{name}.ilpcsr_bm25_3gram.run"
            if il.exists():                                  # vs IL-PCSR's strongest lexical baseline
                pairs.append(("ilpcsr_bm25_3gram", il, "full", b))
            rows += pairs_rows(name, pairs, _qrels_for(name, ev), ("AP", "nDCG@10"), n_perm)
    for name in ladder_sets:
        steps = [s["name"] for s in ev.ablation_ladder]
        paths = [runs / "ablation" / f"{name}.{s}.run" for s in steps]
        if all(p.exists() for p in paths):
            q = _qrels_for(name, ev)
            rows += pairs_rows(name, [(steps[i], paths[i], steps[i + 1], paths[i + 1]) for i in range(len(steps) - 1)],
                               q, ("AP",), n_perm)
    if (runs / "e3_control.full.run").exists() and (runs / "e3_cross_version.full.run").exists():
        rows += e3_rows(runs, ev, metrics=("AP",), n_perm=n_perm)
    rows = finish(rows, project_path(ev.outputs.tables) / "significance.csv")
    for r in rows:
        star = "†" if r["significant"] == "yes" else " "
        print(f"  {star} {r['set']:22s} {r['a']:>22s} -> {r['b']:<22s} {r['metric']:8s} "
              f"{r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  p={r['p']:.4f}  p_holm={r['p_holm']:.4f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired randomization tests over existing run files")
    ap.add_argument("--set", help="test set, e.g. e1_ilpcsr")
    ap.add_argument("--a", default="baseline", help="system A run name (results/runs/<set>.<A>.run)")
    ap.add_argument("--b", default="full")
    ap.add_argument("--ladder", action="store_true", help="each ablation step vs the previous one")
    ap.add_argument("--e3", action="store_true", help="IPC wording vs BNS wording on the E3 queries")
    ap.add_argument("--all", action="store_true", help="everything the report needs (also run by make eval)")
    ap.add_argument("--perm", type=int, default=10000)
    args = ap.parse_args()

    ev = load_config("eval.yaml")
    runs = project_path(ev.outputs.runs)
    out = project_path(ev.outputs.tables) / "significance.csv"
    if args.all:
        run_all(ev, n_perm=args.perm)
        return
    rows: list[dict] = []
    if args.e3:
        rows += e3_rows(runs, ev, n_perm=args.perm)
    if args.set and args.ladder:
        steps = [s["name"] for s in ev.ablation_ladder]
        paths = [runs / "ablation" / f"{args.set}.{s}.run" for s in steps]
        rows += pairs_rows(args.set, [(steps[i], paths[i], steps[i + 1], paths[i + 1]) for i in range(len(steps) - 1)],
                           _qrels_for(args.set, ev), n_perm=args.perm)
    elif args.set:
        rows += pairs_rows(args.set, [(args.a, runs / f"{args.set}.{args.a}.run", args.b, runs / f"{args.set}.{args.b}.run")],
                           _qrels_for(args.set, ev), n_perm=args.perm)
    if not rows:
        ap.error("give --set (with --a/--b or --ladder), --e3, or --all")
    for r in finish(rows, out):
        star = "†" if r["significant"] == "yes" else " "
        print(f"{star} {r['set']} {r['a']} -> {r['b']} {r['metric']}: {r['mean_a']:.4f} -> {r['mean_b']:.4f} "
              f"(diff {r['diff']:+.4f}, 95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], "
              f"{r['wins']} wins / {r['losses']} losses, p={r['p']:.4f}, Holm p={r['p_holm']:.4f}, n={r['queries']})")
    print("written", out)


if __name__ == "__main__":
    main()
