"""Build and evaluate judicial-treatment signals (rank/treatment.py), E21.  [owner: Gaurav — working]

    python scripts/20_build_treatment.py            # writes data/processed/index/treatment.json
    python scripts/20_build_treatment.py --eval     # also E21 on E1 (needs make build)

    results/tables/treatment_flags.csv    every judgment flagged overruled / doubted, with the
                                          citing judgment and sentence (for checking by hand:
                                          annotation/treatment_check.tsv is generated from it)
    results/tables/treatment_e1.csv       E21: E1 with and without the treatment penalty:
                                          MAP, flagged@10 (share of test queries whose top 10
                                          holds a flagged judgment), moved@10
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.rank import treatment as T
from kanoon_bridge.schema import read_documents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    docs = list(read_documents(project_path(cfg.paths.docs)))
    titles = {d.doc_id: d.title for d in docs}
    table = T.build(docs)
    T.save(table, project_path(cfg.paths.index_dir) / "treatment.json")
    st = Counter(t.status for t in table.values())
    print(f"treatment: {len(table)} judgments with a treatment found; {dict(st)}; "
          f"{sum(len(t.positive) for t in table.values())} positive mentions")
    tables = project_path("results/tables")
    with open(tables / "treatment_flags.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "title", "status", "cue", "citing_id", "citing_title", "citing_date", "sentence"])
        for t in table.values():
            if t.status != "good":
                for e in t.strong + t.weak:
                    w.writerow([t.doc_id, titles.get(t.doc_id, ""), t.status, e[2], e[0], titles.get(e[0], ""), e[1], e[3]])

    if args.eval:
        from kanoon_bridge.eval.metrics import average_precision
        from kanoon_bridge.eval.run_eval import load_test_set
        from kanoon_bridge.search import SearchEngine, SearchOptions

        engine = SearchEngine.load()
        ts = load_test_set("e1_ilpcsr", limit=args.limit)
        flagged = {d for d, t in table.items() if t.status != "good"}
        rows = {"off": [], "on": []}
        moved = []
        for q in ts.queries:
            rels = ts.qrels.get(q.query_id, {})
            tops = {}
            for mode in ("off", "on"):
                opt = SearchOptions(ltr=True, ngram=True, treatment=mode == "on", top_k=100,
                                    max_query_terms=ts.max_query_terms)
                ids = [h.doc_id for h in engine.search(q.copy() if hasattr(q, "copy") else q, opt).precedents]
                tops[mode] = ids
                rows[mode].append((average_precision(ids, rels) if rels else None, any(d in flagged for d in ids[:10])))
            moved.append(len(set(tops["off"][:10]) ^ set(tops["on"][:10])) / 20)
        out = []
        for mode, rs in rows.items():
            aps = [a for a, _ in rs if a is not None]
            out.append({"treatment": mode, "queries": len(rs), "MAP": round(sum(aps) / len(aps), 4),
                        "flagged@10": round(sum(f for _, f in rs) / len(rs), 4),
                        "moved@10": round(sum(moved) / len(moved), 4) if mode == "on" else 0.0})
        with open(tables / "treatment_e1.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        for r in out:
            print("E21", r)


if __name__ == "__main__":
    main()
