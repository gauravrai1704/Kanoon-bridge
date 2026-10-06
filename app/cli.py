"""Command-line search. Best for showing internals in the video (--debug).  [working wrapper]

    python app/cli.py "mere bhai ko chaku maara" --state delhi --date 2025-03-01 --debug
    python app/cli.py "section 302" --date 2023-05-10
    python app/cli.py "murder knife" --baseline          # plain BM25 for comparison

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
    ap.add_argument("--agent", action="store_true", help="layer 2: research agent (several sub-queries, fused)")
    ap.add_argument("--answer", action="store_true", help="layer 3: cited answer on top (implies --agent)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    try:
        engine = SearchEngine.load()
    except FileNotFoundError as err:
        print(f"{err}\nBuild the indexes first: make data && make index && make graph", file=sys.stderr)
        return 1

    opt = SearchOptions.baseline() if args.baseline else SearchOptions(jurisdiction=not args.no_jurisdiction)
    opt.top_k = args.k
    query = Query(args.query, state=args.state, incident_date=args.date)

    if args.agent or args.answer:
        from kanoon_bridge.agent.research import ResearchAgent

        agent = ResearchAgent.load(engine)
        res = agent.run(query, opt)
        trace = res.trace
    else:
        res = engine.search(query, opt)
        trace = res.query.trace

    if args.debug:
        print("\n-- trace --")
        for step, out in trace:
            print(f"  {step:22s} {out}")

    print("\n-- statutes --")
    for h in res.statutes:
        print(f"  {h.rank:2d}. {h.doc_id:20s} {h.score:8.3f}")
    print("\n-- precedents --")
    for h in res.precedents:
        extra = "  " + "  ".join(f"{k}={v:.3f}" for k, v in h.components.items()) if args.debug else ""
        print(f"  {h.rank:2d}. {h.doc_id:20s} {h.score:8.3f}{extra}")

    if args.answer:
        from kanoon_bridge.rag.answer import answer

        try:
            ans = answer(res, agent.docs or {})
            print("\n-- answer --\n" + ans.text)
            for flag in ans.flags:
                print("  FLAG:", flag)
        except NotImplementedError as err:
            print(f"\n-- answer -- not available yet ({err})")

    if args.debug and hasattr(res, "timings_ms"):
        print("\n-- timings (ms) --  " + "  ".join(f"{k}={v:.1f}" for k, v in res.timings_ms.items()))
    print("\nNot legal advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
