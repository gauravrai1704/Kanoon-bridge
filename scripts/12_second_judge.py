"""Second judge for E4 and E6: pools to judge, then Cohen's kappa and results under each judge.  [owner: Gaurav]

    python scripts/12_second_judge.py pool       # -> data/queries/judging/{e4,e6}_pool.jsonl
    (judge the pools: one TSV per judge, data/queries/judging/<set>.<judge>.tsv with
     unit_id <tab> doc_id <tab> grade 0/1/2 - a teammate with scripts/judge_queries.py, or an
     independent blind judge; the judge must not see the first judge's grades)
    python scripts/12_second_judge.py agree      # kappa + MAP/nDCG of baseline and full under each judge

Units: E4 is judged per legal NEED (the three language versions share one need) and E6 per
TOPIC (state only changes binding vs persuasive, which is a fact about the court, not a
judgment). The pool for a unit is the union of the top 10 of the BM25 baseline and the full
system over every query of the unit, plus every document the first judge marked relevant.

First judge ("team"): E4 = the sections chosen per need (scripts/08); E6 = citation-based
topical relevance (the precedent applied the topic's section). Kappa is computed over the pooled
(unit, document) pairs, on binary relevance and on the 0/1/2 grades (E4).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

POOL_DEPTH = 10
GUIDE = {
    "e4": ("Incident date: March 2025, so the BNS / BNSS / BSA are in force. Grade each provision for the "
           "legal need: 2 = the provision that directly defines or punishes this situation (the one a lawyer "
           "would cite first); 1 = closely related and useful (an aggravated form, the definition the "
           "punishing section relies on, a companion procedural section); 0 = not about this need. A "
           "section of the OLD codes (IPC / CrPC / Indian Evidence Act) is 0 because it no longer applies."),
    "e6": ("Grade each judgment for the topic, ignoring which court decided it: 1 = the judgment is "
           "substantially about this topic (the issue is decided or discussed on its merits); 0 = it is not, "
           "or only mentions it in passing."),
}


def _excerpt(doc, n: int) -> str:
    parts = [p.text for p in doc.paragraphs if p.zone in ("facts", "statute")][:6]
    parts += [p.text for p in doc.paragraphs if p.zone in ("ratio", "decision")][:6]
    text = " ".join(" ".join(parts).split())
    return text[:n]


def pool() -> None:
    from kanoon_bridge.config import load_config, project_path
    from kanoon_bridge.eval.run_eval import load_test_set, ranked_ids, run_queries
    from kanoon_bridge.index.docstore import DocStore
    from kanoon_bridge.search import SearchEngine, SearchOptions

    cfg, ev = load_config(), load_config("eval.yaml")
    engine = SearchEngine.load(cfg)
    docs = DocStore.load(cfg)
    out_dir = project_path("data/queries/judging")
    out_dir.mkdir(parents=True, exist_ok=True)
    for short, name, unit_key, text_len in (("e4", "e4_multilingual", "need_id", 700), ("e6", "e6_jurisdiction", "topic", 2200)):
        ts = load_test_set(name, ev=ev)
        units: dict[str, dict] = {}
        for q in ts.queries:
            row = ts.rows[q.query_id]
            u = units.setdefault(row[unit_key], {"unit_id": row[unit_key], "queries": [], "docs": set(),
                                                 "sections": row.get("sections", [])})
            u["queries"].append(q.text)
            u["docs"].update(d for d, g in ts.qrels.get(q.query_id, {}).items() if g > 0)
        for opt in (SearchOptions.baseline(), SearchOptions.full()):
            run = ranked_ids(run_queries(engine, ts.queries, opt, ts.target, POOL_DEPTH))
            for q in ts.queries:
                units[ts.rows[q.query_id][unit_key]]["docs"].update(run.get(q.query_id, [])[:POOL_DEPTH])
        path = out_dir / f"{short}_pool.jsonl"
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for u in units.values():
                cands = []
                for d in sorted(u["docs"]):
                    doc = docs.get(d)
                    if doc is None:
                        continue
                    cands.append({"doc_id": d, "title": doc.title, "ref": (doc.meta or {}).get("ref", ""),
                                  "court": doc.court_name or "", "text": _excerpt(doc, text_len)})
                n += len(cands)
                f.write(json.dumps({"unit_id": u["unit_id"], "queries": u["queries"], "guide": GUIDE[short],
                                    "candidates": cands}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(units)} units, {n} (unit, document) pairs -> {path}")
        # the first judge's grades over the same pool
        team = out_dir / f"{name}.team.tsv"
        with open(team, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["unit_id", "doc_id", "grade"])
            grades = _team_grades(ts, unit_key)
            for u in units.values():
                for d in sorted(u["docs"]):
                    w.writerow([u["unit_id"], d, grades.get((u["unit_id"], d), 0)])
        print(f"  first judge over the pool -> {team}")


def _team_grades(ts, unit_key: str) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for q in ts.queries:
        u = ts.rows[q.query_id][unit_key]
        for d, g in ts.qrels.get(q.query_id, {}).items():
            out[(u, d)] = max(out.get((u, d), 0), int(g))
    return out


def _read(path: Path) -> dict[tuple[str, str], int]:
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.reader(f, delimiter="\t") if r and r[0] != "unit_id"]
    return {(r[0], r[1]): int(float(r[2])) for r in rows}


def kappa(a: list[int], b: list[int]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    if not n:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agree() -> None:
    from kanoon_bridge.config import load_config, project_path
    from kanoon_bridge.eval import metrics
    from kanoon_bridge.eval.ablation import write_table
    from kanoon_bridge.eval.qrels import jurisdiction_grades
    from kanoon_bridge.eval.run_eval import load_test_set, ranked_ids, run_queries
    from kanoon_bridge.index import store
    from kanoon_bridge.search import SearchEngine, SearchOptions

    ev = load_config("eval.yaml")
    jdir = project_path("data/queries/judging")
    rows, res_rows = [], []
    engine = None
    for short, name, unit_key in (("e4", "e4_multilingual", "need_id"), ("e6", "e6_jurisdiction", "topic")):
        team_p = jdir / f"{name}.team.tsv"
        others = sorted(p for p in jdir.glob(f"{name}.*.tsv") if p != team_p)
        if not team_p.exists() or not others:
            print(f"{name}: need {team_p.name} and a second judge's file - skipped")
            continue
        team = _read(team_p)
        for op in others:
            judge = op.stem.split(".", 1)[1]
            other = _read(op)
            keys = sorted(set(team) & set(other))
            a, b = [team[k] for k in keys], [other[k] for k in keys]
            ab, bb = [int(x > 0) for x in a], [int(x > 0) for x in b]
            r = {"set": name, "judges": f"team vs {judge}", "pairs": len(keys),
                 "agreement_binary": round(sum(x == y for x, y in zip(ab, bb)) / len(keys), 3) if keys else 0,
                 "kappa_binary": round(kappa(ab, bb), 3),
                 "kappa_graded": round(kappa(a, b), 3) if short == "e4" else "",
                 "relevant_team": sum(ab), "relevant_second": sum(bb)}
            rows.append(r)
            print(f"{name}: team vs {judge}: {len(keys)} pairs, agreement {r['agreement_binary']}, "
                  f"kappa (binary) {r['kappa_binary']}" + (f", kappa (graded) {r['kappa_graded']}" if r["kappa_graded"] != "" else ""))
            # results under each judge
            engine = engine or SearchEngine.load()
            ts = load_test_set(name, ev=ev)
            per_q_second = {q.query_id: {d: g for (u, d), g in other.items() if u == ts.rows[q.query_id][unit_key]}
                            for q in ts.queries}
            for jname, qrels in (("team", ts.qrels), (judge, per_q_second)):
                if name.startswith("e6"):
                    qrels = jurisdiction_grades(qrels, store.load("facets").metas,
                                                {q.query_id: q.state for q in ts.queries})
                for sysname, opt in (("baseline", SearchOptions.baseline()), ("full", SearchOptions.full())):
                    run = ranked_ids(run_queries(engine, ts.queries, opt, ts.target, ev.depth))
                    res_rows.append({"set": name, "qrels": jname, "system": sysname,
                                     "MAP": round(metrics.mean_over_queries(run, qrels, metrics.average_precision), 4),
                                     "nDCG@10": round(metrics.mean_over_queries(run, qrels, metrics.ndcg_at_k, 10), 4),
                                     "P@5": round(metrics.mean_over_queries(run, qrels, metrics.precision_at_k, 5), 4)})
                    print(f"  qrels={jname:6s} {sysname:8s} {res_rows[-1]}")
    tables = project_path(ev.outputs.tables)
    if rows:
        write_table(rows, tables / "agreement.csv")
        write_table(res_rows, tables / "results_by_judge.csv")
        print("written", tables / "agreement.csv", tables / "results_by_judge.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pool", "agree"])
    args = ap.parse_args()
    pool() if args.step == "pool" else agree()


if __name__ == "__main__":
    main()
