"""E23: the core against BM25 that knows the crosswalk (eval/baselines.py).  [owner: Gaurav — working]

Answers the obvious objection to E2/E3/E7-E10: "plain BM25 cannot know the crosswalk". Systems:
BM25, BM25 + crosswalk expansion, BM25 + crosswalk + date code filter, and the core
(SearchOptions.full()). Paired randomization tests on per-query AP, Holm-corrected over all
comparisons of the run.

    python scripts/23_eval_fair_baselines.py                     # IL-PCSR corpus (make build)
    python scripts/23_eval_fair_baselines.py --sets e10_bns_era --tag hc   # corpus with the eCourts judgments

    results/tables/fair_baselines_<tag>.csv, fair_baselines_<tag>_significance.csv
"""

from __future__ import annotations

import argparse
import csv

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.eval import significance as sig
from kanoon_bridge.eval.baselines import CrosswalkBM25
from kanoon_bridge.eval.run_eval import evaluate_set, load_test_set, read_run
from kanoon_bridge.search import SearchEngine, SearchOptions

SETS = ["e2_collision", "e3_cross_version", "e3_control", "e8_procedure_cross_version", "e8_procedure_control",
        "e7_temporal", "e9_procedure_temporal"]
KEEP = ("MAP", "MRR", "nDCG@10", "P@1", "wrong_hit@10", "code@1", "n_queries")


def write(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--tag", default="ilpcsr")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    ev = load_config("eval.yaml")
    engine = SearchEngine.load()
    systems = {"BM25": (SearchOptions.baseline(), None),
               "BM25 + crosswalk": (SearchOptions.baseline(), CrosswalkBM25(engine)),
               "BM25 + crosswalk + date": (SearchOptions.baseline(), CrosswalkBM25(engine, date_filter=True)),
               "core": (SearchOptions.full(), None)}
    out_dir = project_path(ev.outputs.runs) / "fair"
    rows, comps = [], []
    for name in args.sets:
        ts = load_test_set(name, ev=ev, limit=args.limit)
        per = {}
        for system, (opt, agent) in systems.items():
            tag = system.replace(" ", "_").replace("+", "plus")
            scores = evaluate_set(engine, ts, opt, tag, ev, out_dir, agent=agent)
            rows.append({"set": name, "system": system, **{k: round(v, 4) for k, v in scores.items() if k in KEEP}})
            per[system] = sig.per_query(read_run(out_dir / f"{name}.{tag}.run"), ts.qrels)["AP"]
            print(f"E23 {name:28s} {system:24s} MAP {scores.get('MAP', 0):.4f}")
        for a, b in (("BM25", "BM25 + crosswalk"), ("BM25 + crosswalk", "BM25 + crosswalk + date"),
                     ("BM25 + crosswalk + date", "core")):
            comps.append({"set": name, "a": a, "b": b, "metric": "AP", **sig.compare(per[a], per[b])})
    for c, p in zip(comps, sig.holm([c["p"] for c in comps])):
        c["p_holm"] = round(p, 5)
        print(f"E23 {c['set']:28s} {c['a']:24s} -> {c['b']:24s} {c['mean_a']:.4f} -> {c['mean_b']:.4f}  "
              f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  p_holm {c['p_holm']}")
    tables = project_path("results/tables")
    write(rows, tables / f"fair_baselines_{args.tag}.csv")
    write(comps, tables / f"fair_baselines_{args.tag}_significance.csv")


if __name__ == "__main__":
    main()
