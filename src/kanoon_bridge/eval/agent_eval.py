"""Evaluate layers 2 and 3.  [owner: Gaurav — working; agent internals: Shaurya]

Agent (layer 2), on E1 (and E6 if judged):
    core single query  vs  agent (RRF)  vs  agent (CombSUM), same planner and sub-results
    metrics: MAP, MRR, F1@k (k from val), nDCG@10; plus searches per question, sub-query
    kinds used, share of questions reformulated and share where reformulation improved AP.
    Sanity: an agent whose plan has only the 'original' sub-query == core ranking order.

RAG (layer 3), on data/queries/rag_questions.jsonl (rows: id, text, incident_date, state,
answerable):
    supported-sentence rate   RAG answer  vs  the same LLM closed-book (no retrieval)
    version flags per answer  wrong code for the date / bare colliding numbers (rag/version_check.py)
    abstention                precision and recall on rows with "answerable": false
The closed-book baseline needs the Claude generator (an API key); with the extractive generator
only the RAG side is reported.

Output: results/tables/agent.csv, results/tables/rag.csv, results/tables/rag_answers.jsonl
"""

from __future__ import annotations

import json
import statistics

from kanoon_bridge.config import Config, load_config, project_path


def compare_agent(test_set: str = "e1_ilpcsr", engine=None, limit: int | None = None,
                  ev: Config | None = None, diagnostics: int = 50) -> list[dict]:
    """core vs agent (RRF) vs agent (CombSUM) on one set, plus diagnostics of the RRF agent on
    the first `diagnostics` queries: searches per question, sub-query kinds used, share of
    questions that got a reformulation round and share where that round improved AP."""
    import copy

    from kanoon_bridge.agent.fuse import fuse
    from kanoon_bridge.agent.research import ResearchAgent
    from kanoon_bridge.eval.ablation import write_table
    from kanoon_bridge.eval.metrics import average_precision
    from kanoon_bridge.eval.run_eval import evaluate_set, load_test_set
    from kanoon_bridge.search import SearchEngine, SearchOptions

    ev = ev or load_config("eval.yaml")
    engine = engine or SearchEngine.load()
    ts = load_test_set(test_set, ev=ev, limit=limit)
    rrf = ResearchAgent.load(engine, fusion="rrf")
    comb = ResearchAgent(engine=engine, cfg=rrf.cfg, planner=rrf.planner, docs=rrf.docs, fusion="combsum")
    out_dir = project_path(ev.outputs.runs) / "agent"
    rows = []
    for system, ag in (("core", None), ("agent_rrf", rrf), ("agent_combsum", comb)):
        scores = evaluate_set(engine, ts, SearchOptions(), system, ev, out_dir, agent=ag)
        rows.append({"set": test_set, "system": system, **{k: round(v, 4) for k, v in scores.items()}})
        print(f"  agent {test_set} {system:14s} MAP={scores.get('MAP', 0):.4f}")

    # diagnostics on the RRF agent
    searches, kinds, rounds2, improved = [], {}, 0, 0
    judged = [q for q in ts.queries if any(g > 0 for g in ts.qrels.get(q.query_id, {}).values())][:diagnostics]
    opt = SearchOptions(top_k=ev.get("depth", 100), max_query_terms=ts.max_query_terms)
    for q in judged:
        res = rrf.run(copy.deepcopy(q), opt)
        searches.append(res.n_searches)
        for sr in res.subresults:
            kinds[sr.subquery.kind] = kinds.get(sr.subquery.kind, 0) + 1
        if len(res.plans) > 1:
            rounds2 += 1
            first = [sr for sr in res.subresults if not sr.subquery.sq_id.startswith("r")]
            r1 = fuse(first, ts.target, "rrf", rrf.cfg.agent.rrf_k)
            final = [h.doc_id for h in (res.precedents if ts.target == "precedent" else res.statutes)]
            rels = ts.qrels[q.query_id]
            if average_precision(final, rels) > average_precision(sorted(r1, key=r1.get, reverse=True), rels):
                improved += 1
    if searches:
        rows[1]["searches_per_q"] = round(statistics.fmean(searches), 2)
        rows[1]["reformulated_share"] = round(rounds2 / len(searches), 3)
        rows[1]["reformulation_helped_share"] = round(improved / rounds2, 3) if rounds2 else ""
        rows[1]["subquery_kinds"] = " ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        print(f"  agent diagnostics: {rows[1]['searches_per_q']} searches/q, kinds {rows[1]['subquery_kinds']}, "
              f"reformulated {rounds2}/{len(searches)}, helped {improved}")
    write_table(rows, project_path(ev.outputs.tables) / f"agent_{test_set}.csv")
    return rows


def load_questions(path: str) -> list[dict]:
    p = project_path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("#")]


def evaluate_rag(questions_path: str | None = None, engine=None, generator: str | None = None,
                 ev: Config | None = None, limit: int | None = None) -> list[dict]:
    from kanoon_bridge.eval.ablation import write_table
    from kanoon_bridge.rag.answer import RagPipeline
    from kanoon_bridge.rag.generate import pick_generator
    from kanoon_bridge.schema import Query
    from kanoon_bridge.search import SearchEngine, SearchOptions

    ev = ev or load_config("eval.yaml")
    questions = load_questions(questions_path or ev.rag.questions)[:limit]
    if not questions:
        print("  rag: no questions file - skipped")
        return []
    engine = engine or SearchEngine.load()
    rag = RagPipeline.load(engine)
    gen = pick_generator(generator or ev.rag.get("generator", "auto"))
    closed = gen == "claude"
    rows, dump = [], []
    for row in questions:
        q = Query.from_dict(row)
        res = engine.search(q, SearchOptions(top_k=10))
        ans = rag.answer(res, generator=gen)
        out = {"id": row.get("id"), "answerable": row.get("answerable", True), "abstained": ans.abstained,
               "sentences": len(ans.sentences), "supported_rate": round(ans.supported_rate, 3) if ans.sentences else "",
               "version_flags": len(ans.version_flags)}
        rec = {"id": row.get("id"), "question": q.text, "answer": ans.text, "flags": ans.flags,
               "sources": [c.header() for c in ans.chunks], "generator": ans.generator}
        if closed:
            cb = rag.answer(res, generator="claude", closed_book=True)
            out["closed_book_supported_rate"] = round(cb.supported_rate, 3) if cb.sentences else ""
            out["closed_book_version_flags"] = len(cb.version_flags)
            rec["closed_book"] = {"answer": cb.text, "flags": cb.flags}
        rows.append(out)
        dump.append(rec)

    summary = {"id": "MEAN", "answerable": "", "abstained": sum(r["abstained"] for r in rows)}
    for key in ("supported_rate", "version_flags", "closed_book_supported_rate", "closed_book_version_flags"):
        vals = [r[key] for r in rows if key in r and r[key] != "" and not r["abstained"]]
        if vals:
            summary[key] = round(statistics.fmean(vals), 3)
    abst = [r for r in rows if r["abstained"]]
    unans = [r for r in rows if r["answerable"] is False]
    if abst:
        summary["abstain_precision"] = round(sum(r["answerable"] is False for r in abst) / len(abst), 3)
    if unans:
        summary["abstain_recall"] = round(sum(r["abstained"] for r in unans) / len(unans), 3)
    summary["generator"] = gen
    rows.append(summary)

    tables = project_path(ev.outputs.tables)
    write_table(rows, tables / "rag.csv")
    with open(tables / "rag_answers.jsonl", "w", encoding="utf-8") as f:
        for rec in dump:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  rag ({gen}): " + ", ".join(f"{k}={v}" for k, v in summary.items() if k not in ("id", "answerable")))
    return rows
