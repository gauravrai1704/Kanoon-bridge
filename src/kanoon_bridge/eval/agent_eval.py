"""Evaluate layers 2 and 3.  [owner: Gaurav — working; agent internals: Shaurya]

Agent (layer 2), on E1 (and E6 if judged):
    core single query  vs  agent (RRF over the plan's sub-queries)
    metrics: MAP, MRR, F1@k (k from val), nDCG@10; plus mean searches per question.
    Sanity: while the planner only emits the 'original' sub-query, agent == core ranking order.

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
                  ev: Config | None = None) -> list[dict]:
    from kanoon_bridge.agent.research import ResearchAgent
    from kanoon_bridge.eval.ablation import write_table
    from kanoon_bridge.eval.run_eval import evaluate_set, load_test_set
    from kanoon_bridge.schema import Query
    from kanoon_bridge.search import SearchEngine, SearchOptions

    ev = ev or load_config("eval.yaml")
    engine = engine or SearchEngine.load()
    ts = load_test_set(test_set, ev=ev, limit=limit)
    agent = ResearchAgent.load(engine)
    out_dir = project_path(ev.outputs.runs) / "agent"
    rows = []
    for system, ag in (("core", None), ("agent_rrf", agent)):
        scores = evaluate_set(engine, ts, SearchOptions(), system, ev, out_dir, agent=ag)
        row = {"set": test_set, "system": system, **{k: round(v, 4) for k, v in scores.items()}}
        if ag is not None and ts.queries:
            import copy

            row["searches_per_q"] = round(statistics.fmean(
                ag.run(copy.deepcopy(q), SearchOptions(top_k=10, max_query_terms=ts.max_query_terms)).n_searches
                for q in ts.queries[:20]), 2)
        rows.append(row)
        print(f"  agent {test_set} {system:10s} MAP={scores.get('MAP', 0):.4f}")
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
