"""Error analysis of E2 (colliding numbers) and E19 (bail facts -> charges).  [owner: Gaurav — working]

    python scripts/26_error_analysis.py            # needs the IL-PCSR build and the E17 run files
    results/tables/error_analysis_e2.csv, error_analysis_e19.csv

E2  from the saved runs (results/runs/layers/e2_collision.<system>.run): the top statute of each
    query is right, the colliding section of the other code, another section of the right code,
    another section of the wrong code, or missing; by query style (bare number / with context
    words) and incident era (before / after 1 July 2024).
E19 agent v2 on the 995 bail orders: recall@10 of the charged sections split into GENERAL
    provisions (joint liability, abetment, conspiracy, attempt: IPC 34, 35, 107-120B, 149, 511 and
    their BNS counterparts) and OFFENCE provisions; and how often the top statute belongs to
    procedure or evidence law instead of penal law.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.eval.open_eval import load_rows, statute_ref
from kanoon_bridge.eval.run_eval import read_run

GENERAL_IPC = {"34", "35", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118",
               "119", "120", "120a", "120b", "149", "511"}


def write(rows, name):
    path = project_path("results/tables") / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def e2(engine) -> None:
    rows = {r["id"]: r for r in load_rows("e2_collision")}
    out = []
    for system in ("core", "agent_v2"):
        path = project_path("results/runs/layers") / f"e2_collision.{system}.run"
        if not path.exists():
            print(f"  {path} missing - run scripts/18 first")
            continue
        run = read_run(path)
        acc = defaultdict(Counter)
        for qid, ranking in run.items():
            r = rows.get(qid)
            if r is None:
                continue
            top = statute_ref(engine, ranking[0]) if ranking else ""
            exp, wrong = r["expected"], set(r.get("wrong_refs") or [])
            code = exp.split(":")[0]
            if not top:
                kind = "no statute"
            elif top == exp:
                kind = "right"
            elif top in wrong:
                kind = "colliding section of the other code"
            elif top.split(":")[0] == code:
                kind = "other section, right code"
            else:
                kind = "other section, other code"
            era = "after" if r.get("incident_date", "") >= "2024-07-01" else "before"
            for key in ("all", f"style={r.get('style')}", f"era={era}"):
                acc[key][kind] += 1
        for key, c in acc.items():
            n = sum(c.values())
            out.append({"system": system, "group": key, "n": n,
                        **{k: round(v / n, 4) for k, v in sorted(c.items())}})
            print(f"E2 {system:9s} {key:16s} n={n:3d} " + "  ".join(f"{k}: {v / n:.2f}" for k, v in sorted(c.items())))
    write(out, "error_analysis_e2")


def e19(engine) -> None:
    from kanoon_bridge.agent.research import ResearchAgentV2
    from kanoon_bridge.eval import facts_eval as F
    from kanoon_bridge.schema import Query
    from kanoon_bridge.search import SearchOptions

    norm = engine.analyzer.text_res.normalizer
    agent = ResearchAgentV2.load(engine)
    rows = F.load_bail(words=60)
    hit = defaultdict(list)
    top_family = Counter()
    for row in rows:
        refs = [statute_ref(engine, h.doc_id) for h in agent.run(Query(row["text"]), SearchOptions.full()).statutes[:10]]
        bases = {re.sub(r"\(.*$", "", r) for r in refs}
        fam = refs[0].split(":")[0] if refs else "none"
        top_family[{"ipc": "penal", "bns": "penal", "crpc": "procedure", "bnss": "procedure",
                    "iea": "evidence", "bsa": "evidence"}.get(fam, "other")] += 1
        for g in row["gold"]:
            g = re.sub(r"\(.*$", "", g)
            aliases = {g, *(re.sub(r"\(.*$", "", e) for e in norm.equivalents(g))}
            kind = "general" if g.split(":")[1] in GENERAL_IPC else "offence"
            hit[kind].append(bool(aliases & bases))
    out = []
    for kind, v in hit.items():
        out.append({"what": f"recall@10 of {kind} provisions", "n": len(v), "value": round(sum(v) / len(v), 4)})
    n = sum(top_family.values())
    for fam, c in top_family.most_common():
        out.append({"what": f"top statute is {fam} law", "n": n, "value": round(c / n, 4)})
    for r in out:
        print(f"E19 {r['what']:36s} n={r['n']:5d} {r['value']:.3f}")
    write(out, "error_analysis_e19")


def main() -> None:
    from kanoon_bridge.search import SearchEngine

    engine = SearchEngine.load()
    e2(engine)
    e19(engine)


if __name__ == "__main__":
    main()
