"""Terminal tool for judging results.  [working]

Shows each (query, result) pair from one or more run files, pooled and de-duplicated (so a judge
cannot tell which system found a document), asks for a grade, and appends to YOUR OWN qrels
file. Already-judged pairs are skipped, so you can stop and resume.

    python scripts/judge_queries.py --set e4_multilingual --judge gaurav
        # = queries data/queries/e4_multilingual.jsonl
        #   runs    results/runs/e4_multilingual.baseline.run + .full.run (whichever exist)
        #   writes  data/queries/qrels/e4_multilingual.gaurav.tsv

    python scripts/judge_queries.py --queries data/queries/e2_collision.jsonl \\
        --run results/runs/e2_collision.full.run --out my.tsv --judge gaurav --depth 10

Grades: 0 = not relevant, 1 = partly relevant, 2 = relevant.  s = skip, q = quit.
Two members judge each set; then:  python scripts/merge_judgments.py --set e4_multilingual
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.eval.run_eval import read_run
from kanoon_bridge.schema import read_documents, read_queries


def pooled(runs: list[dict[str, list[str]]], depth: int) -> dict[str, list[str]]:
    """Interleave the top `depth` of every run per query, without duplicates."""
    out: dict[str, list[str]] = {}
    for qid in dict.fromkeys(q for r in runs for q in r):
        seen: list[str] = []
        for i in range(depth):
            for r in runs:
                lst = r.get(qid, [])
                if i < len(lst) and lst[i] not in seen:
                    seen.append(lst[i])
        out[qid] = seen
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", help="test-set name, e.g. e4_multilingual (fills in the paths below)")
    ap.add_argument("--queries")
    ap.add_argument("--run", nargs="*", help="one or more run files (pooled)")
    ap.add_argument("--out", help="default: data/queries/qrels/<set>.<judge>.tsv")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--depth", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config()
    ev = load_config("eval.yaml")
    if args.set:
        args.queries = args.queries or str(project_path(f"data/queries/{args.set}.jsonl"))
        if not args.run:
            runs_dir = project_path(ev.outputs.runs)
            args.run = [str(p) for p in sorted(runs_dir.glob(f"{args.set}.*.run"))]
        args.out = args.out or str(project_path(f"data/queries/qrels/{args.set}.{args.judge}.tsv"))
    if not (args.queries and args.run and args.out):
        ap.error("give --set, or all of --queries, --run and --out")
    if not args.run:
        raise SystemExit(f"no run files for {args.set} - run: python -m kanoon_bridge.eval.run_eval --set {args.set} "
                         "--system baseline (and --system full)")

    queries = {q.query_id: q for q in read_queries(args.queries)}
    run = pooled([read_run(r) for r in args.run], args.depth)
    docs = {d.doc_id: d for d in read_documents(project_path(cfg.paths.docs))}

    out = Path(args.out)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            for row in csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"):
                if row["judge"] == args.judge:
                    done.add((row["query_id"], row["doc_id"]))
    todo = sum(1 for q, ds in run.items() for d in ds if (q, d) not in done)
    print(f"{len(run)} queries, {todo} pairs left to judge -> {out}")
    new_file = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if new_file:
            w.writerow(["query_id", "doc_id", "grade", "judge"])
        for qid, ranked in run.items():
            q = queries.get(qid)
            for doc_id in ranked:
                if (qid, doc_id) in done:
                    continue
                n += 1
                d = docs.get(doc_id)
                print("\n" + "=" * 80 + f"  [{n}/{todo}]")
                print(f"QUERY {qid}: {q.text if q else '?'}  (state={q.state if q else '-'}, date={q.incident_date if q else '-'})")
                print(f"DOC   {doc_id}: {d.title if d else ''}  [{d.court_name if d else ''}]")
                print(textwrap.fill((d.text if d else "")[:900], 100))
                ans = input("grade 0/1/2, s=skip, q=quit > ").strip().lower()
                if ans == "q":
                    return
                if ans in {"0", "1", "2"}:
                    w.writerow([qid, doc_id, ans, args.judge])
                    f.flush()


if __name__ == "__main__":
    main()
