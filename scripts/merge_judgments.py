"""Agreement between judges, a list of disagreements, and the merged qrels file.  [working]

    python scripts/merge_judgments.py --set e4_multilingual

Reads every data/queries/qrels/<set>.<judge>.tsv, prints percent agreement and Cohen's kappa
for each pair of judges (over the pairs both graded), writes the pairs where two judges differ
by 2 grades to results/tables/<set>_disagreements.tsv (settle these together, then edit the
merged file), and writes the merged qrels to data/queries/qrels/<set>.tsv (judge "merged").
With one judge only, that judge's grades are copied.
"""

from __future__ import annotations

import argparse
import csv
import itertools

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.eval.agreement import cohens_kappa, percent_agreement
from kanoon_bridge.eval.qrels import load_tsv, merge_judges, save_tsv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    args = ap.parse_args()
    qdir = project_path("data/queries/qrels")
    files = sorted(p for p in qdir.glob(f"{args.set}.*.tsv"))
    if not files:
        raise SystemExit(f"no judge files {qdir}/{args.set}.<judge>.tsv - run scripts/judge_queries.py first")
    judges = {p.stem.split(".", 1)[1]: load_tsv(p) for p in files}
    print(f"{args.set}: judges {', '.join(judges)}")
    tables = project_path(load_config("eval.yaml").outputs.tables)
    tables.mkdir(parents=True, exist_ok=True)
    disagreements = []
    for (ja, a), (jb, b) in itertools.combinations(judges.items(), 2):
        print(f"  {ja} vs {jb}: agreement {percent_agreement(a, b):.3f}, Cohen's kappa {cohens_kappa(a, b):.3f}")
        for q in a:
            for d, g in a[q].items():
                if d in b.get(q, {}) and abs(g - b[q][d]) >= 2:
                    disagreements.append((q, d, ja, g, jb, b[q][d]))
    if disagreements:
        out = tables / f"{args.set}_disagreements.tsv"
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["query_id", "doc_id", "judge_a", "grade_a", "judge_b", "grade_b"])
            w.writerows(disagreements)
        print(f"  {len(disagreements)} pairs differ by 2 grades -> {out}")
    merged = None
    for qrels in judges.values():
        merged = qrels if merged is None else merge_judges(merged, qrels)
    save_tsv(merged, qdir / f"{args.set}.tsv", judge="merged")
    print(f"  merged qrels -> {qdir / (args.set + '.tsv')}  ({sum(len(v) for v in merged.values())} judgments)")


if __name__ == "__main__":
    main()
