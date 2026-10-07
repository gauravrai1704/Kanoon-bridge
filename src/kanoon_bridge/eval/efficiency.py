"""Efficiency: exhaustive vs tiered vs champion-list scoring.  [owner: Gaurav — working]

For each candidate mode, over the first `n_queries` E1 test queries (capped to 100 terms like E1):
    latency_ms_median, latency_ms_p95   wall time of SearchEngine.search
    recall@20_vs_exhaustive             overlap of the mode's top-20 with exhaustive top-20

Output: results/tables/efficiency.csv (+ a chart from eval/plots.py).
"""

from __future__ import annotations

import copy
import math
import statistics
import time

from kanoon_bridge.config import Config, load_config, project_path


def compare_modes(n_queries: int | None = None, engine=None, ev: Config | None = None,
                  queries=None) -> list[dict]:
    from kanoon_bridge.eval.ablation import write_table
    from kanoon_bridge.eval.run_eval import load_test_set
    from kanoon_bridge.search import SearchEngine, SearchOptions

    ev = ev or load_config("eval.yaml")
    engine = engine or SearchEngine.load()
    eff = ev.get("efficiency", {})
    n = n_queries or eff.get("n_queries", 200)
    if queries is None:
        queries = load_test_set("e1_ilpcsr", ev=ev, limit=n).queries
    if engine.tiers is None:
        print("  efficiency: no tiers.pkl (run scripts/03_build_graph.py) - only exhaustive timed")
    modes = eff.get("modes", ["all", "tiers", "champions"])

    tops: dict[str, dict[str, list[str]]] = {}
    rows = []
    for mode in modes:
        if mode != "all" and engine.tiers is None:
            continue
        opt = SearchOptions(candidate_mode=mode, top_k=20, max_query_terms=ev.get("e1_max_query_terms", 100))
        times, tops[mode] = [], {}
        for q in queries:
            t0 = time.perf_counter()
            res = engine.search(copy.deepcopy(q), opt)
            times.append((time.perf_counter() - t0) * 1000)
            tops[mode][q.query_id] = [h.doc_id for h in res.precedents]
        rows.append({"mode": mode, "queries": len(queries),
                     "latency_ms_median": round(statistics.median(times), 2) if times else 0.0,
                     "latency_ms_p95": round(sorted(times)[min(len(times) - 1, math.ceil(0.95 * len(times)) - 1)], 2) if times else 0.0})
    base = tops.get("all", {})
    for row in rows:
        got = tops[row["mode"]]
        overlaps = [len(set(got[q]) & set(base[q])) / len(base[q]) for q in base if base[q]]
        row["recall@20_vs_exhaustive"] = round(statistics.fmean(overlaps), 4) if overlaps else 1.0
    write_table(rows, project_path(ev.outputs.tables) / "efficiency.csv")
    for r in rows:
        print(f"  efficiency {r['mode']:10s} median={r['latency_ms_median']}ms p95={r['latency_ms_p95']}ms "
              f"recall@20={r['recall@20_vs_exhaustive']}")
    return rows
