"""Terminal tool for judging results.  [working]

Shows each (query, result) pair from a run file with a snippet, asks for a grade, and appends
to a qrels TSV under your judge name. Already-judged pairs are skipped, so you can stop and
resume. Two members judge each set; then run eval/agreement.py.

    python scripts/judge_queries.py --queries data/queries/e2_collision.jsonl \
        --run results/runs/e2_collision.full.run --out data/queries/qrels/e2_collision.tsv \
        --judge gaurav --depth 10

Grades: 0 = not relevant, 1 = partly relevant, 2 = relevant.  s = skip, q = quit.
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.eval.run_eval import read_run
from kanoon_bridge.schema import read_documents, read_queries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--depth", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config()
    queries = {q.query_id: q for q in read_queries(args.queries)}
    run = read_run(args.run)
    docs = {d.doc_id: d for d in read_documents(project_path(cfg.paths.docs))}

    out = Path(args.out)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["judge"] == args.judge:
                    done.add((row["query_id"], row["doc_id"]))
    new_file = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if new_file:
            w.writerow(["query_id", "doc_id", "grade", "judge"])
        for qid, ranked in run.items():
            q = queries.get(qid)
            for doc_id in ranked[: args.depth]:
                if (qid, doc_id) in done:
                    continue
                d = docs.get(doc_id)
                print("\n" + "=" * 80)
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
