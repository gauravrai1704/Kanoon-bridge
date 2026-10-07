"""Command-line search. Best for showing internals in the video (--debug).  [working wrapper]

    python app/cli.py "mere bhai ko chaku maara" --state delhi --date 2025-03-01 --debug
    python app/cli.py "section 302" --date 2023-05-10
    python app/cli.py "murder knife" --baseline          # plain BM25 for comparison
    python app/cli.py "dowery deth" --debug               # spelling correction ("did you mean")
    python app/cli.py "extort* AND threat"                # wildcard inside a Boolean query
    python app/cli.py "" --like P01                       # cases similar to one precedent
    python app/cli.py "BNS 103 knife" --state delhi --date 2025-01-03 --agent --debug      # layer 2
    python app/cli.py "punishment for murder" --date 2025-01-03 --answer                 # layer 3
    python app/cli.py "punishment for murder" --date 2025-01-03 --answer --generator extractive

--debug prints each query rewrite (language, normalised text, tokens, code in force,
cross-code sections), the statutes used by the bridge, and every result's score breakdown.
"""

from __future__ import annotations

import argparse
import sys

from kanoon_bridge.schema import Query
from kanoon_bridge.search import SearchEngine, SearchOptions


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Kanoon-Bridge search (not legal advice)")
    ap.add_argument("query")
    ap.add_argument("--state", help="user's state, e.g. delhi, maharashtra")
    ap.add_argument("--date", help="incident date YYYY-MM-DD (decides IPC vs BNS)")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--baseline", action="store_true", help="plain BM25 only")
    ap.add_argument("--no-jurisdiction", action="store_true")
    ap.add_argument("--no-ltr", action="store_true", help="hand-set weights instead of the learned re-ranker")
    ap.add_argument("--agent", action="store_true", help="layer 2: research agent (several sub-queries, fused)")
    ap.add_argument("--fusion", choices=["rrf", "combsum"], help="layer 2 fusion (default: config agent.fusion)")
    ap.add_argument("--planner", choices=["rules", "llm", "hybrid"], help="layer 2 planner (default: config agent.planner)")
    ap.add_argument("--answer", action="store_true", help="layer 3: cited answer on top of the results")
    ap.add_argument("--generator", default=None, help="auto | claude | extractive (default: configs rag.generator)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--snippets", type=int, default=5, help="show snippets + reasons for the top N precedents")
    ap.add_argument("--no-snippets", action="store_true", help="ids and scores only (no document store)")
    ap.add_argument("--similar", action="store_true", help="also list cases similar to the top precedent")
    ap.add_argument("--feedback", help="relevance feedback: comma-separated precedent ids you found relevant "
                                       "(the query is re-run moved toward them, Rocchio)")
    ap.add_argument("--like", help="find cases similar to this precedent id (no query needed: pass '' as query)")
    args = ap.parse_args(argv)

    try:
        engine = SearchEngine.load()
    except FileNotFoundError as err:
        print(f"{err}\nBuild the indexes first: make data && make index && make graph", file=sys.stderr)
        return 1

    if args.like and not args.query.strip():
        return _similar(engine, args.like, None, args.k)

    opt = SearchOptions.baseline() if args.baseline else SearchOptions.interactive()
    opt.jurisdiction = opt.jurisdiction and not args.no_jurisdiction
    if args.no_ltr:
        opt.ltr = False
    opt.top_k = args.k
    query = Query(args.query, state=args.state, incident_date=args.date)

    agent = None
    if args.agent:
        from kanoon_bridge.agent.research import ResearchAgent

        if args.planner:
            from kanoon_bridge.config import load_config

            engine.cfg = load_config(overrides={"agent": {"planner": args.planner}})
        agent = ResearchAgent.load(engine, fusion=args.fusion)
        res = agent.run(query, opt)
        trace = res.analyzed.trace + res.trace
    else:
        res = engine.search(query, opt)
        if args.feedback:
            from kanoon_bridge.index.docstore import DocStore
            from kanoon_bridge.rank.feedback import rocchio_options

            marked = [d.strip() for d in args.feedback.split(",") if d.strip()]
            shown = [h.doc_id for h in res.precedents]
            fb = rocchio_options(engine, res.query, DocStore.load(engine.cfg), marked,
                                 [d for d in shown[:5] if d not in marked])
            for key, value in fb.items():
                setattr(opt, key, value)
            res = engine.search(Query(args.query, state=args.state, incident_date=args.date), opt)
            print(f"feedback: +{' +'.join(fb['extra_terms'])}" + (f"  -{' -'.join(fb['drop_terms'])}" if fb["drop_terms"] else ""))
        trace = res.query.trace

    if args.debug:
        print("\n-- trace --")
        for step, out in trace:
            print(f"  {step:22s} {out}")

    analyzed = res.analyzed if agent is not None else res.query
    if analyzed.suggestion:
        print(f"\nDid you mean: {analyzed.suggestion}")

    docs = None
    if True:                                     # titles need the doc store even without snippets
        try:
            from kanoon_bridge.index.docstore import DocStore

            docs = getattr(agent, "docs", None) or DocStore.load(engine.cfg)
        except FileNotFoundError:
            docs = None

    from kanoon_bridge import present

    print("\n-- statutes --")
    for h in res.statutes:
        note = present.version_note(present.statute_ref(engine, h.doc_id), engine.analyzer.text_res.normalizer)
        title = present.title_of(docs, h.doc_id)
        print(f"  {h.rank:2d}. {h.doc_id:12s} {h.score:8.3f}  {title[:60]}" + (f"   [{note}]" if note else ""))
    print("\n-- precedents --")
    for h in res.precedents:
        extra = "  " + "  ".join(f"{k}={v:.3f}" for k, v in h.components.items()) if args.debug else ""
        if args.debug and agent is not None:
            extra += "  from " + ",".join(res.found_by(h.doc_id))
        meta = engine.facets.metas.get(h.doc_id)
        when = f"{meta.court_name or meta.court}, {meta.decision_date.year}" if meta and meta.decision_date else ""
        print(f"  {h.rank:2d}. {h.doc_id:12s} {h.score:8.3f}  {present.title_of(docs, h.doc_id)[:60]}  ({when}){extra}")
        if docs is not None and not args.no_snippets and h.rank <= args.snippets and h.doc_id in docs:
            print("      " + present.snippet(docs[h.doc_id], analyzed, engine.analyzer.text_res))
            why = present.explain(h, analyzed, engine)
            if why:
                print("      why: " + "; ".join(why))
    if args.similar and res.precedents:
        args.like = res.precedents[0].doc_id
    if args.like:
        _similar(engine, args.like, docs, args.k)

    if args.answer:
        from kanoon_bridge.rag.answer import RagPipeline, render

        try:
            rag = RagPipeline.load(engine, docs=getattr(agent, "docs", None))
            print("\n-- answer --\n" + render(rag.answer(res, generator=args.generator)))
        except (NotImplementedError, RuntimeError, FileNotFoundError) as err:
            print(f"\n-- answer -- not available ({err})")

    if args.debug and hasattr(res, "timings_ms"):
        print("\n-- timings (ms) --  " + "  ".join(f"{k}={v:.1f}" for k, v in res.timings_ms.items()))
    print("\nNot legal advice.")
    return 0


def _similar(engine, doc_id: str, docs, k: int) -> int:
    from kanoon_bridge import present
    from kanoon_bridge.rank.similar import SimilarCases

    sim = SimilarCases.load(engine, docs)
    print(f"\n-- cases similar to {doc_id} ({present.title_of(sim.docs, doc_id)}) --")
    for i, (d, s, parts) in enumerate(sim.find(doc_id, k), start=1):
        print(f"  {i:2d}. {d:12s} {s:6.3f}  {present.title_of(sim.docs, d)[:60]}  "
              + "  ".join(f"{n}={v:.2f}" for n, v in parts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
