"""Run every test set, the ablations and the efficiency check; write tables and figures.  [owner: D]

    python scripts/05_run_all_evals.py              # everything in configs/eval.yaml
    python scripts/05_run_all_evals.py e2_collision # one set
"""

from __future__ import annotations

import csv
import sys

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.eval import ablation, efficiency, plots
from kanoon_bridge.eval.metrics import evaluate
from kanoon_bridge.eval.run_eval import load_test_set, ranked_ids, run_queries, write_run
from kanoon_bridge.search import SearchEngine, SearchOptions


def main() -> None:
    ev = load_config("eval.yaml")
    sets = sys.argv[1:] or list(ev.test_sets)
    engine = SearchEngine.load()
    tables = project_path(ev.outputs.tables)
    tables.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in sets:
        queries, qrels, target = load_test_set(name)
        for system, opt in (("baseline", SearchOptions.baseline()), ("full", SearchOptions())):
            run = run_queries(engine, queries, opt, target)
            write_run(run, project_path(ev.outputs.runs) / f"{name}.{system}.run", system)
            rows.append({"set": name, "system": system, **evaluate(ranked_ids(run), qrels, ev.k_values)})
            print(name, system, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)})

    with open(tables / "main_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    if not sys.argv[1:]:
        ablation.run_ladder("e1_ilpcsr")
        ablation.run_language_ablation()
        efficiency.compare_modes()
        plots.make_all()


if __name__ == "__main__":
    main()
