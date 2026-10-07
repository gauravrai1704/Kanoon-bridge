"""Run every test set, the ablations, efficiency, agent and RAG evals; write tables and figures.
[owner: Gaurav — working]

    python scripts/05_run_all_evals.py                       # everything in configs/eval.yaml
    python scripts/05_run_all_evals.py --sets e2_collision e7_temporal
    python scripts/05_run_all_evals.py --limit 50            # quick pass: first 50 queries per set
    python scripts/05_run_all_evals.py --skip-ablation --skip-efficiency --skip-agent --skip-rag --skip-typos --skip-significance
    python scripts/05_run_all_evals.py --only-plots          # redraw figures from existing tables
    python scripts/05_run_all_evals.py --sample              # pipeline check on the synthetic sample

E1/E1s queries are whole judgments, capped to their 100 highest-idf terms (eval.yaml
e1_max_query_terms); expect a few minutes per system on the full 627-query test split.
Hand-built sets with no queries file / no qrels yet are skipped with a note.
"""

from __future__ import annotations

import argparse
import time

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.eval import ablation, agent_eval, efficiency, plots
from kanoon_bridge.eval.run_eval import evaluate_set, load_test_set
from kanoon_bridge.search import SearchEngine, SearchOptions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", help="test sets (default: all in configs/eval.yaml)")
    ap.add_argument("--limit", type=int, help="first N queries per set")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-efficiency", action="store_true")
    ap.add_argument("--skip-agent", action="store_true")
    ap.add_argument("--skip-rag", action="store_true")
    ap.add_argument("--skip-typos", action="store_true")
    ap.add_argument("--skip-significance", action="store_true")
    ap.add_argument("--rag-generator", help="auto | claude | extractive")
    ap.add_argument("--only-plots", action="store_true")
    ap.add_argument("--sample", action="store_true", help="evaluate on the synthetic sample (after make sample)")
    args = ap.parse_args()
    if args.sample:
        import os

        os.environ["KB_ILPCSR_DIR"] = "tests/data/ilpcsr_sample"

    # sample runs write to results/sample/ so synthetic numbers never mix with real ones
    ev = load_config("eval.yaml", overrides={"outputs": {"runs": "results/sample/runs", "tables": "results/sample/tables",
                                                         "figures": "results/sample/figures"}} if args.sample else None)
    tables, figures = project_path(ev.outputs.tables), project_path(ev.outputs.figures)
    if args.only_plots:
        plots.make_all(tables, figures)
        return
    sets = args.sets or list(ev.test_sets)
    engine = SearchEngine.load()
    runs = project_path(ev.outputs.runs)

    rows = []
    for name in sets:
        t0 = time.time()
        try:
            ts = load_test_set(name, ev=ev, limit=args.limit)
        except FileNotFoundError as err:
            print(f"{name}: skipped ({err})")
            continue
        if not ts.queries or not (ts.judged or name.startswith(("e2", "e7"))):
            print(f"{name}: skipped ({len(ts.queries)} queries, {ts.judged} judged - fill the qrels first)")
            continue
        print(f"{name}: {len(ts.queries)} queries, {ts.judged} judged"
              + ("" if ts.judged else " (qrels empty: only the qrels-free metric is meaningful)"))
        for system, opt in (("baseline", SearchOptions.baseline()), ("full", SearchOptions.full())):
            scores = evaluate_set(engine, ts, opt, system, ev, runs)
            rows.append({"set": name, "system": system, "queries": len(ts.queries),
                         **{k: round(v, 4) for k, v in scores.items()}})
            shown = {k: v for k, v in rows[-1].items() if k in ("MAP", "MRR", "F1@k_val", "nDCG@10", "P@5")
                     or k.startswith(("wrong", "binding", "code_acc", "P@5_"))}
            print(f"  {system:9s} {shown}")
        print(f"  ({time.time() - t0:.1f}s)")
    if engine.ltr_short is not None and not args.skip_ablation:
        # the short-query LTR model is not part of "full" (see SearchOptions.ltr_short): report it beside
        lrows = []
        for name in ev.get("ltr_short_sets", []):
            if name not in sets:
                continue
            ts = load_test_set(name, ev=ev, limit=args.limit)
            for system, opt in (("full", SearchOptions.full()), ("full+ltr_short", SearchOptions(ltr=True, ngram=True, ltr_short=True))):
                scores = evaluate_set(engine, ts, opt, system, ev, runs / "ltr_short")
                lrows.append({"set": name, "system": system, **{k: round(v, 4) for k, v in scores.items()
                                                                  if k in ("MAP", "nDCG@10", "P@5", "MRR", "binding_share@5")}})
                print(f"  ltr_short {name} {system:15s} {lrows[-1]}")
        if lrows:
            ablation.write_table(lrows, tables / "ltr_short.csv")
    if rows:
        out = project_path(ev.outputs.tables) / "main_results.csv"
        if args.sets and out.exists():              # a subset was re-run: keep the other sets' rows
            done = {r["set"] for r in rows}
            rows = [r for r in plots.read_table(out) if r["set"] not in done] + rows
        ablation.write_table(rows, out)

    if not args.skip_ablation:
        for name in ("e1_ilpcsr", "e1q_ilpcsr_short", "e2_collision"):
            if name in sets:
                ts = load_test_set(name, ev=ev, limit=args.limit)
                if ts.judged:
                    ablation.run_ladder(name, engine, args.limit, ev)
        ablation.run_language_ablation(engine=engine, ev=ev)
    if not args.skip_efficiency and "e1_ilpcsr" in sets:
        efficiency.compare_modes(min(args.limit or 10**9, ev.efficiency.n_queries), engine, ev)
    if not args.skip_typos:
        from kanoon_bridge.eval import typos

        try:
            typos.run(engine=engine, ev=ev, n_items=min(args.limit or 10**9, ev.typos.n_items) if args.limit else None)
        except FileNotFoundError as err:
            print(f"typos: skipped ({err})")
    if not args.skip_agent:
        for name in ("e1_ilpcsr", "e6_jurisdiction"):
            if name in sets and load_test_set(name, ev=ev, limit=args.limit).judged:
                agent_eval.compare_agent(name, engine, args.limit, ev)
    if not args.skip_rag:
        try:
            agent_eval.evaluate_rag(engine=engine, generator=args.rag_generator, ev=ev, limit=args.limit)
        except RuntimeError as err:
            print(f"rag: skipped ({err})")
    if not args.skip_significance:
        from kanoon_bridge.eval import significance

        print("significance (paired randomization test, Holm-corrected; † = p_holm < 0.05):")
        significance.run_all(ev, sets=sets)
    plots.make_all(tables, figures)


if __name__ == "__main__":
    main()
