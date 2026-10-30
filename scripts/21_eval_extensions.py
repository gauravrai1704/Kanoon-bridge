"""E19 (facts -> charged sections), the STAIR-style ToC variant, E22 (KV reuse).  [owner: Gaurav — working]

Needs a built corpus (make build for IL-PCSR, or make open) and data/raw/bail (scripts/00 --only bail).

    python scripts/21_eval_extensions.py facts   # E19: core / agent v1 / agent v2 / v2 + ToC on bail facts (and ILSI if present)
    python scripts/21_eval_extensions.py toc     # agent v2 with and without ToC navigation on E2/E7/E9/E14
    python scripts/21_eval_extensions.py kv      # E22: chunk reuse and cache hit rates
    python scripts/21_eval_extensions.py all --tag ilpcsr
    python scripts/21_eval_extensions.py e10 --tag hc   # after scripts/11 fetch + ingest and a rebuild

    results/tables/facts_<tag>.csv, toc_<tag>.csv, kv_reuse_<tag>.csv, kv_cache_<tag>.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import time

from kanoon_bridge.agent.research import ResearchAgent, ResearchAgentV2
from kanoon_bridge.config import project_path
from kanoon_bridge.search import SearchEngine, SearchOptions


def write(rows: list[dict], name: str) -> None:
    path = project_path("results/tables") / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def v2(engine, **over):
    cfg = copy.deepcopy(engine.cfg)
    for k, v in over.items():
        cfg.agent[k] = v
    return ResearchAgentV2.load(engine, cfg=cfg)


def facts(engine, tag, limit, words):
    from kanoon_bridge.eval import facts_eval as F

    systems = {"core": lambda q: engine.search(q, SearchOptions.full()),
               "agent v1": (lambda a: lambda q: a.run(q, SearchOptions.full()))(ResearchAgent.load(engine)),
               "agent v2": (lambda a: lambda q: a.run(q, SearchOptions.full()))(v2(engine)),
               "v2 + ToC navigation": (lambda a: lambda q: a.run(q, SearchOptions.full()))(v2(engine, toc_weight=1.0))}
    rows, sig_rows = [], []
    for source, loader in (("bail", F.load_bail), ("ilsi", F.load_ilsi)):
        data = loader(words=words, limit=limit)
        if not data:
            print(f"  {source}: no data - skipped")
            continue
        per = {}
        for name, run in systems.items():
            per[name] = {}
            r = {"source": source, "system": name, "words": words, **F.score(data, run, engine, per[name])}
            rows.append(r)
            print("E19", r)
        from kanoon_bridge.eval import significance as sig

        comps = [{"source": source, "a": a, "b": b, "metric": m, **sig.compare(per[a][m], per[b][m])}
                 for a, b in (("core", "agent v2"), ("agent v1", "agent v2"), ("agent v2", "v2 + ToC navigation"))
                 for m in ("MAP", "hit@1")]
        for c, p in zip(comps, sig.holm([c["p"] for c in comps])):
            c["p_holm"] = round(p, 5)
            print("E19 sig", c)
        sig_rows.extend(comps)
    write(rows, f"facts_{tag}")
    if sig_rows:
        write(sig_rows, f"facts_{tag}_significance")


def toc(engine, tag, limit):
    from kanoon_bridge.eval import layers_eval as L
    from kanoon_bridge.eval.open_eval import score_set
    from kanoon_bridge.rag.answer import RagPipeline

    rows = []
    agents = {"agent v2": v2(engine), "v2 + ToC navigation": v2(engine, toc_weight=1.0)}
    for name in ("e2_collision", "e7_temporal", "e9_procedure_temporal"):
        for s, a in agents.items():
            r = {"set": name, "system": s, **score_set(name, lambda q, a=a: a.run(q, SearchOptions.full_drift()),
                                                        engine, limit)}
            rows.append(r)
            print("TOC", r)
    rag = RagPipeline.load(engine)
    qs = L.punishment_questions(limit=limit)
    for s, a in agents.items():
        acc = {"statute@1": 0, "punishment_ok": 0}
        for row in qs:
            from kanoon_bridge.schema import Query

            ans = rag.answer(a.run(Query.from_dict(row), SearchOptions.interactive()), generator="extractive", version=2)
            from kanoon_bridge.drift.punishment import extract

            acc["statute@1"] += L.answer_law_ref(ans, rag.docs) == row["section"]
            acc["punishment_ok"] += L.same_punishment(L.answer_punishment(ans), extract(row["gold"]))
        r = {"set": "e14_punishment", "system": s, **{k: round(v / len(qs), 4) for k, v in acc.items()}, "n": len(qs)}
        rows.append(r)
        print("TOC", r)
    write(rows, f"toc_{tag}")


def kv(engine, tag, limit):
    from kanoon_bridge.eval import kv_reuse as K
    from kanoon_bridge.eval import layers_eval as L
    from kanoon_bridge.eval.agent_eval import load_questions
    from kanoon_bridge.eval.facts_eval import load_bail
    from kanoon_bridge.eval.open_eval import load_rows
    from kanoon_bridge.rag.answer import RagPipeline

    questions = (L.punishment_questions() + load_rows("e7_temporal") + load_rows("e9_procedure_temporal")
                 + load_rows("e2_collision") + load_bail() + load_questions("data/queries/rag_questions.jsonl"))
    questions = questions[:limit] if limit else questions
    rag = RagPipeline.load(engine)
    trace = K.collect(engine, rag, v2(engine), questions, log=print)
    r = K.reuse(trace)
    print("E22 reuse", r)
    write([r], f"kv_reuse_{tag}")
    rows = K.sweep(trace)
    for x in rows:
        print("E22 cache", x)
    write(rows, f"kv_cache_{tag}")


def e10(engine, tag, limit):
    """E10 on a corpus with the BNS-era High Court judgments (scripts/11 ingest, then make data
    index graph): IPC-numbered questions, gold = post-2024 judgments citing the BNS counterpart."""
    from kanoon_bridge.config import load_config
    from kanoon_bridge.eval import significance as sig
    from kanoon_bridge.eval.run_eval import evaluate_set, load_test_set, read_run

    ev = load_config("eval.yaml")
    ts = load_test_set("e10_bns_era", ev=ev, limit=limit)
    out_dir = project_path(ev.outputs.runs) / "layers"
    systems = {"baseline BM25": (SearchOptions.baseline(), None), "core": (SearchOptions.full(), None),
               "agent v2": (SearchOptions.full(), v2(engine))}
    rows, per = [], {}
    for name, (opt, agent) in systems.items():
        tagname = name.replace(" ", "_")
        scores = evaluate_set(engine, ts, opt, tagname, ev, out_dir, agent=agent)
        rows.append({"set": "e10_bns_era", "system": name, **{k: round(v, 4) for k, v in scores.items()}})
        per[name] = sig.per_query(read_run(out_dir / f"e10_bns_era.{tagname}.run"), ts.qrels)
        print("E10", rows[-1])
    for a, b in (("baseline BM25", "core"), ("core", "agent v2")):
        c = sig.compare(per[a]["AP"], per[b]["AP"])
        rows.append({"set": "e10_bns_era", "system": f"{a} -> {b} (AP)", **c})
        print("E10", rows[-1])
    write(rows, f"e10_{tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["facts", "toc", "kv", "e10", "all"])
    ap.add_argument("--tag", default="ilpcsr")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--words", type=int, default=60, help="E19 query length in words (0 = whole facts)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    engine = SearchEngine.load()
    if args.what in ("facts", "all"):
        facts(engine, args.tag, args.limit, args.words)
    if args.what in ("toc", "all"):
        toc(engine, args.tag, args.limit)
    if args.what in ("kv", "all"):
        kv(engine, args.tag, args.limit)
    if args.what == "e10":
        e10(engine, args.tag, args.limit)
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
