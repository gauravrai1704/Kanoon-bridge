"""Run a system on a test set and save the ranked run.  [owner: D]

Working: TREC run file I/O, and `run_queries`, which calls search.SearchEngine (rule 4).
TODO(D): `load_test_set` for each set in configs/eval.yaml.

TREC run format (one line per result):
    query_id  Q0  doc_id  rank  score  system_name

    python -m kanoon_bridge.eval.run_eval --set e2_collision --system full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.schema import Query

Run = dict[str, list[tuple[str, float]]]


def write_run(run: Run, path: str | Path, system: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q, ranked in run.items():
            for i, (d, s) in enumerate(ranked, start=1):
                f.write(f"{q}\tQ0\t{d}\t{i}\t{s:.6f}\t{system}\n")


def read_run(path: str | Path) -> dict[str, list[str]]:
    out: dict[str, list[tuple[int, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            q, _, d, rank, _score, _sys = line.rstrip("\n").split("\t")
            out.setdefault(q, []).append((int(rank), d))
    return {q: [d for _, d in sorted(v)] for q, v in out.items()}


def ranked_ids(run: Run) -> dict[str, list[str]]:
    return {q: [d for d, _ in r] for q, r in run.items()}


def run_queries(engine, queries: list[Query], options, target: str = "precedent", k: int = 100) -> Run:
    """Search every query; target 'precedent' or 'statute' picks which result list to keep."""
    options.top_k = k
    run: Run = {}
    for q in queries:
        res = engine.search(q, options)
        hits = res.precedents if target == "precedent" else res.statutes
        run[q.query_id or q.text] = [(h.doc_id, h.score) for h in hits]
    return run


def load_test_set(name: str) -> tuple[list[Query], dict[str, dict[str, int]], str]:
    """(queries, qrels, target) for a set named in configs/eval.yaml.

    TODO(D): JSONL sets -> schema.read_queries + eval.qrels.load_tsv; 'ilpcsr_test' ->
    ingest.load_ilpcsr.load_queries('test') turned into Query objects (use the judgment
    text, or its first N paragraphs, as the query) + load_qrels('test', target).
    """
    raise NotImplementedError("TODO(D): load named test sets")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--system", default="full", help="'full', 'baseline', or an ablation step name")
    args = ap.parse_args()

    from kanoon_bridge.eval.metrics import evaluate
    from kanoon_bridge.search import SearchEngine, SearchOptions

    cfg_eval = load_config("eval.yaml")
    queries, qrels, target = load_test_set(args.set)
    if args.system == "baseline":
        opt = SearchOptions.baseline()
    elif args.system == "full":
        opt = SearchOptions()
    else:
        step = next(s for s in cfg_eval.ablation_ladder if s["name"] == args.system)
        opt = SearchOptions.from_dict(step)
    run = run_queries(SearchEngine.load(), queries, opt, target)
    out = project_path(cfg_eval.outputs.runs) / f"{args.set}.{args.system}.run"
    write_run(run, out, args.system)
    print(json.dumps(evaluate(ranked_ids(run), qrels, cfg_eval.k_values), indent=2))


if __name__ == "__main__":
    main()
