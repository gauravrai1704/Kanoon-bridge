"""Tune BM25F zone weights and b on the IL-PCSR VALIDATION split.  [owner: Gaurav]

    python scripts/09_tune_on_val.py               # ~5 min; prints MAP per setting, best last
    python scripts/09_tune_on_val.py --limit 200
    python scripts/09_tune_on_val.py --ngram       # beta of the trigram channel (precedents, statutes)

Only val queries are used (the test split is never seen while tuning). Copy the winning values
into configs/default.yaml (zones.*, bm25.b), then retrain LTR (make ltr) because its features
include the zone scores. Results go to results/tables/tuning_val.csv for the report.
"""

from __future__ import annotations

import argparse
import itertools
import time

PROFILES = {
    "no zones (BM25)":   None,               # plain BM25 on the whole judgment (the baseline)
    "hand-set (before)": {"facts": 1.0, "arguments": 0.8, "ratio": 1.5, "decision": 1.2, "other": 0.5},
    "uniform":           {"facts": 1.0, "arguments": 1.0, "ratio": 1.0, "decision": 1.0, "other": 1.0},
    "facts-heavy":       {"facts": 2.0, "arguments": 0.8, "ratio": 1.0, "decision": 0.6, "other": 0.5},
    "reasoning-heavy":   {"facts": 1.0, "arguments": 0.5, "ratio": 2.0, "decision": 0.8, "other": 0.3},
    "facts+reasoning":   {"facts": 1.5, "arguments": 0.6, "ratio": 1.5, "decision": 0.6, "other": 0.3},
}
B_VALUES = (0.5, 0.7, 0.85, 0.95, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--profiles", nargs="*", help="subset of PROFILES (default: all)")
    ap.add_argument("--b", nargs="*", type=float, help="subset of b values (default: all)")
    ap.add_argument("--ngram", action="store_true", help="tune ngram.beta / ngram.beta_statutes instead")
    args = ap.parse_args()
    if args.ngram:
        tune_ngram(args.limit)
        return

    from kanoon_bridge.config import load_config, project_path
    from kanoon_bridge.eval import metrics
    from kanoon_bridge.eval.ablation import write_table
    from kanoon_bridge.eval.run_eval import options_for, ranked_ids, run_queries
    from kanoon_bridge.search import SearchEngine

    ev = load_config("eval.yaml")
    engine = SearchEngine.load()
    from kanoon_bridge.eval.run_eval import _ilpcsr_queries
    from kanoon_bridge.ingest import load_ilpcsr

    queries = _ilpcsr_queries("val", engine.cfg)[: args.limit]
    qrels = {q: {d: g for d, g in r.items()} for q, r in load_ilpcsr.load_qrels("val", "precedent", engine.cfg).items()}
    qrels = {q.query_id: qrels.get(q.query_id, {}) for q in queries}
    opt = options_for("+zones", ev)
    rows = []
    profiles = {k: v for k, v in PROFILES.items() if not args.profiles or k in args.profiles}
    for (name, weights), b in itertools.product(profiles.items(), args.b or B_VALUES):
        engine.cfg.zones.update(weights or {})
        engine.cfg.bm25.b = b
        engine._scorers.clear()
        t = time.time()
        o = options_for("bm25", ev) if weights is None else opt
        run = ranked_ids(run_queries(engine, queries, o, "precedent", ev.depth, ev.e1_max_query_terms))
        m = metrics.mean_over_queries(run, qrels, metrics.average_precision)
        rows.append({"profile": name, "b": b, **(weights or {}), "MAP_val": round(m, 4)})
        print(f"{name:20s} b={b:<5} MAP(val, {len(queries)} q) = {m:.4f}   ({time.time() - t:.0f}s)", flush=True)
    rows.sort(key=lambda r: -r["MAP_val"])
    out = project_path(ev.outputs.tables) / "tuning_val.csv"
    if out.exists() and (args.profiles or args.b):             # partial grid: merge with earlier rows
        from kanoon_bridge.eval.plots import read_table

        seen = {(r["profile"], float(r["b"])) for r in rows}
        rows += [r for r in read_table(out) if (r["profile"], float(r["b"])) not in seen]
        rows.sort(key=lambda r: -float(r["MAP_val"]))
    write_table(rows, out)
    print("best:", rows[0])


def tune_ngram(limit: int) -> None:
    from kanoon_bridge.config import load_config, project_path
    from kanoon_bridge.eval import metrics
    from kanoon_bridge.eval.ablation import write_table
    from kanoon_bridge.eval.run_eval import _ilpcsr_queries, options_for, ranked_ids, run_queries
    from kanoon_bridge.ingest import load_ilpcsr
    from kanoon_bridge.search import SearchEngine

    ev = load_config("eval.yaml")
    engine = SearchEngine.load()
    queries = _ilpcsr_queries("val", engine.cfg)[:limit]
    opt = options_for("+zones", ev)                      # unigram BM25F + trigram channel
    rows = []
    for target, key in (("precedent", "beta"), ("statute", "beta_statutes")):
        raw = load_ilpcsr.load_qrels("val", target, engine.cfg)
        qrels = {q.query_id: raw.get(q.query_id, {}) for q in queries}
        for beta in (0.0, 0.3, 0.5, 0.7, 0.85, 1.0):
            engine.cfg.ngram[key] = beta
            run = ranked_ids(run_queries(engine, queries, opt, target, ev.depth, ev.e1_max_query_terms))
            m = metrics.mean_over_queries(run, qrels, metrics.average_precision)
            rows.append({"target": target, "beta": beta, "MAP_val": round(m, 4)})
            print(f"{target:10s} beta={beta:<5} MAP(val, {len(queries)} q) = {m:.4f}", flush=True)
        best = max((r for r in rows if r["target"] == target), key=lambda r: r["MAP_val"])
        engine.cfg.ngram[key] = best["beta"]
        print("best", best, flush=True)
    write_table(rows, project_path(ev.outputs.tables) / "tuning_ngram_val.csv")


if __name__ == "__main__":
    main()
