"""E13: drift-aware precedent transfer inside retrieval.  [owner: Gaurav — working]

For every new-code section (BNS, BNSS, BSA) with a predecessor that some indexed precedent cites,
one question in the E3 style ("cases under section 152 BNS", "cases under section 482 BNSS",
incident date 2025-01-01). Two runs,
full system with drift off and on, compared per drift class of the asked section:

    MAP_cited    gold = precedents that cite a predecessor IPC section (what E3 measures:
                 citation relevance, which ignores whether the law changed)
    stale@10     share of the top 10 whose transfer score is below 0.5: IPC-era judgments on a
                 provision the BNS changed substantively or dropped
    moved@10     share of the top 10 that differs between the two runs

The expected pattern, and the claim we test: on IDENTICAL / COSMETIC sections nothing moves
(no harm); on SUBSTANTIVE sections stale@10 falls, at some cost in citation-MAP, because
citation relevance counts a stale precedent as relevant. The queries are built from the
indexed corpus, so the committed data/queries/ sets are not touched.

    python scripts/15_drift_retrieval.py                 # needs the built indexes (make build)
    python scripts/15_drift_retrieval.py --limit 40
    python scripts/15_drift_retrieval.py --family crpc
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from datetime import date

from kanoon_bridge.config import project_path
from kanoon_bridge.eval.metrics import average_precision
from kanoon_bridge.schema import Query
from kanoon_bridge.search import SearchEngine, SearchOptions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="questions per drift class")
    ap.add_argument("--out", default="results/tables/drift_retrieval.csv")
    ap.add_argument("--family", choices=["ipc", "crpc", "iea", "all"], default="all")
    args = ap.parse_args()
    engine = SearchEngine.load()
    if engine.drift is None:
        raise SystemExit("no drift table: run scripts/13_build_drift.py")
    by_section = engine.facets.by_section
    questions: dict[tuple[str, str], list[tuple[str, set[str]]]] = defaultdict(list)
    for fam, table in engine.drift.tables.items():
        if args.family != "all" and fam != args.family:
            continue
        for b, rep in table.sections.items():
            gold = set().union(*(by_section.get(f"{fam}:{i}", set()) for i in rep.old_sections))
            if gold:
                questions[(fam, rep.drift_class.value)].append((b, gold))
    rng = random.Random(0)
    rows = []
    for (fam, cls), qs in sorted(questions.items()):
        label = engine.drift.tables[fam].family.new_label
        if args.limit and len(qs) > args.limit:
            qs = rng.sample(qs, args.limit)
        agg = defaultdict(list)
        for b, gold in qs:
            q = Query(f"cases under section {b} {label}", incident_date=date(2025, 1, 1))
            runs = {}
            for name, drift in (("off", False), ("on", True)):
                opt = SearchOptions.full()
                opt.drift, opt.top_k = drift, 100
                res = engine.search(q, opt)
                ids = [h.doc_id for h in res.precedents]
                targets = engine.drift.targets(res.query)
                stale = [engine.drift.transfer(engine.facets.metas[d].statutes_cited, targets).score < 0.5
                         for d in ids[:10] if d in engine.facets.metas]
                runs[name] = ids
                agg[f"map_{name}"].append(average_precision(ids, {d: 1 for d in gold}))
                agg[f"stale10_{name}"].append(sum(stale) / max(1, len(stale)))
            agg["moved10"].append(1 - len(set(runs["off"][:10]) & set(runs["on"][:10])) / max(1, len(runs["off"][:10])))
        mean = {k: sum(v) / len(v) for k, v in agg.items()}
        rows.append({"family": fam, "drift_class": cls, "questions": len(qs), **{k: round(v, 4) for k, v in mean.items()}})
        print(f"{fam:5s} {cls:13s} n={len(qs):3d}  MAP {mean['map_off']:.3f} -> {mean['map_on']:.3f}   "
              f"stale@10 {mean['stale10_off']:.3f} -> {mean['stale10_on']:.3f}   moved@10 {mean['moved10']:.3f}")
    path = project_path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
