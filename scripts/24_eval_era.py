"""E24: era-aware retrieval (query/era.py) on E10, against the fair baselines.  [owner: Gaurav — working]

E10 asks "cases under section 503 IPC" and counts only judgments citing the BNS counterpart. The
core treats IPC 503 and BNS 351 as one offence, so literal crosswalk expansion beat it (E23). Here
the same need is asked three ways:

    as is          "cases under section 503 IPC"                       (no era intent)
    era filter     the same, with the explicit filter era:new          (a UI switch / era:new)
    era in words   "cases under section 503 IPC decided under the BNS" (detected automatically)

Systems: BM25, BM25 + crosswalk, core. The baselines receive the same text (and, for the explicit
filter, the same text without it: BM25 has no era filter). Paired randomization on per-query AP,
Holm-corrected. Needs the corpus with the eCourts judgments (scripts/11 ingest, then make data
index graph).

    python scripts/24_eval_era.py
    python scripts/24_eval_era.py --answers     # era-aware answers on E14's 273 questions
    results/tables/era_e10.csv, era_e10_significance.csv, era_answers.csv
"""

from __future__ import annotations

import copy
import csv

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.eval import significance as sig
from kanoon_bridge.eval.baselines import CrosswalkBM25
from kanoon_bridge.eval.run_eval import evaluate_set, load_test_set, read_run
from kanoon_bridge.search import SearchEngine, SearchOptions

WORDING = " decided under the BNS"


def variant(ts, how: str):
    t = copy.deepcopy(ts)
    for q in t.queries:
        if how == "era filter":
            q.filters = {**q.filters, "era": "new"}
        elif how == "era in words":
            q.text = q.text + WORDING
    return t


def write(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def answers(engine) -> None:
    """E14's questions (incident date 2025) through agent v2 + RAG v2, preferring a judgment decided
    under the new codes for the quote or not: which era the quoted judgment is from, how often an
    old-era quote is noted and warned, and that the punishment is unaffected."""
    from kanoon_bridge.agent.research import ResearchAgentV2
    from kanoon_bridge.drift.punishment import extract
    from kanoon_bridge.eval import layers_eval as L
    from kanoon_bridge.query.era import of_judgment
    from kanoon_bridge.rag.answer import RagPipeline
    from kanoon_bridge.schema import Query

    rag = RagPipeline.load(engine)
    agent = ResearchAgentV2.load(engine, docs=rag.docs)
    qs = L.punishment_questions()
    rows = []
    for prefer in (False, True):
        rag.cfg.rag["prefer_new_era"] = prefer
        acc = {"quotes": 0, "new_era": 0, "old_era": 0, "old_noted": 0, "old_warned": 0, "punishment_ok": 0}
        for row in qs:
            ans = rag.answer(agent.run(Query.from_dict(row), SearchOptions.interactive()), generator="extractive", version=2)
            acc["punishment_ok"] += L.same_punishment(L.answer_punishment(ans), extract(row["gold"]))
            by_n = {c.n: c for c in ans.chunks}
            quoted = [by_n[n] for _, n in ans.sentences if n in by_n and by_n[n].zone != "statute"]
            if not quoted:
                continue
            acc["quotes"] += 1
            doc = rag.docs.get(quoted[0].doc_id)
            era = of_judgment(getattr(doc, "statutes_cited", None))
            acc["new_era"] += era == "new"
            if era == "old":
                acc["old_era"] += 1
                notes = " ".join(getattr(ans, "notes", []) or [])
                acc["old_noted"] += "decided under the old codes" in notes
                acc["old_warned"] += "check that its reasoning still applies" in notes
        n = len(qs)
        rows.append({"prefer_new_era": prefer, "questions": n, "answers_quoting_a_judgment": acc["quotes"],
                     "quote_new_era": round(acc["new_era"] / max(1, acc["quotes"]), 4),
                     "quote_old_era": round(acc["old_era"] / max(1, acc["quotes"]), 4),
                     "old_era_quotes_noted": round(acc["old_noted"] / max(1, acc["old_era"]), 4),
                     "old_era_quotes_warned": round(acc["old_warned"] / max(1, acc["old_era"]), 4),
                     "punishment_ok": round(acc["punishment_ok"] / n, 4)})
        print("E24 answers", rows[-1])
    write(rows, project_path("results/tables") / "era_answers.csv")


def main() -> None:
    import sys

    ev = load_config("eval.yaml")
    engine = SearchEngine.load()
    if "--answers" in sys.argv:
        answers(engine)
        return
    base = load_test_set("e10_bns_era", ev=ev)
    out_dir = project_path(ev.outputs.runs) / "era"
    systems = {"BM25": (SearchOptions.baseline(), None), "BM25 + crosswalk": (SearchOptions.baseline(), CrosswalkBM25(engine)),
               "core": (SearchOptions.full(), None)}
    rows, per = [], {}
    for how in ("as is", "era filter", "era in words"):
        ts = variant(base, how)
        for system, (opt, agent) in systems.items():
            tag = f"{how}_{system}".replace(" ", "_").replace("+", "plus")
            s = evaluate_set(engine, ts, opt, tag, ev, out_dir, agent=agent)
            rows.append({"query form": how, "system": system,
                         **{k: round(v, 4) for k, v in s.items() if k in ("MAP", "MRR", "P@10", "nDCG@10", "n_queries")}})
            per[(how, system)] = sig.per_query(read_run(out_dir / f"e10_bns_era.{tag}.run"), ts.qrels)["AP"]
            print(f"E24 {how:14s} {system:18s} MAP {s['MAP']:.4f}  P@10 {s['P@10']:.4f}")
    comps = []
    for how in ("era filter", "era in words"):
        for a, b in ((("as is", "core"), (how, "core")), (("as is", "BM25 + crosswalk"), (how, "core")),
                     ((how, "BM25 + crosswalk"), (how, "core"))):
            comps.append({"a": " / ".join(a), "b": " / ".join(b), "metric": "AP", **sig.compare(per[a], per[b])})
    for c, p in zip(comps, sig.holm([c["p"] for c in comps])):
        c["p_holm"] = round(p, 5)
        print(f"E24 {c['a']:34s} -> {c['b']:24s} {c['mean_a']:.4f} -> {c['mean_b']:.4f}  "
              f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  p_holm {c['p_holm']}")
    tables = project_path("results/tables")
    write(rows, tables / "era_e10.csv")
    write(comps, tables / "era_e10_significance.csv")


if __name__ == "__main__":
    main()
