"""Print the IR internals for one query: analysed terms, postings, idf, and a BM25F score worked
out term by term for one document. Built for the demo video and the viva.  [owner: Sharanya]

    python scripts/show_postings.py "dowry death"
    python scripts/show_postings.py "section 302 murder" --doc 1466771
    python scripts/show_postings.py "dahej hatya" --index statutes --date 2025-03-01
    python scripts/show_postings.py "IPC 302"      # shows the offence-id term shared by IPC 302 and BNS 103
"""

from __future__ import annotations

import argparse
from datetime import date


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--index", choices=["precedents", "statutes"], default="precedents")
    ap.add_argument("--doc", help="document to explain the BM25F score for (default: the top one)")
    ap.add_argument("--date", help="incident date YYYY-MM-DD")
    ap.add_argument("--postings", type=int, default=5, help="postings shown per term")
    args = ap.parse_args()

    from kanoon_bridge.index.positional import count
    from kanoon_bridge.rank.bm25f import BM25F
    from kanoon_bridge.schema import Query
    from kanoon_bridge.search import SearchEngine

    engine = SearchEngine.load()
    q = Query(text=args.query, query_id="demo", incident_date=date.fromisoformat(args.date) if args.date else None)
    a = engine.analyzer.analyze(q)
    zidx = engine.precedent_index if args.index == "precedents" else engine.statute_index
    bm = BM25F.from_config(zidx, engine.cfg, use_zones=True)
    if args.index == "statutes":
        bm.b = engine.cfg.bm25.get("b_statutes", bm.b)
    terms = a.weighted_terms()

    print(f"\nquery            {args.query!r}")
    print(f"language         {a.detected_lang}")
    print(f"tokens           {' '.join(a.tokens)}")
    if getattr(a, "expanded_terms", None):
        print(f"expansions       {a.expanded_terms}")
    if getattr(a, "offence_ids", None):
        print(f"offence ids      {a.offence_ids}")
    print(f"index            {args.index}: N = {zidx.n_docs} documents, avg length {zidx.whole.avg_doc_len:.0f} tokens")
    print(f"BM25F            k1 = {bm.k1}, b = {bm.b}, zone weights = {bm.zone_weights}")

    print("\n-- dictionary + postings (whole-document positional index) --")
    print(f"{'term':28s} {'w_q':>5s} {'df':>6s} {'idf':>7s}   postings: doc(tf)[first positions]")
    for t, w in sorted(terms.items(), key=lambda kv: -bm.idf(kv[0])):
        plist = zidx.whole.postings.get(t, {})
        shown = []
        for d, pos in sorted(plist.items(), key=lambda kv: -count(kv[1]))[: args.postings]:
            p = list(pos)[:4] if not isinstance(pos, int) else []
            shown.append(f"{d}({count(pos)}){p}")
        print(f"{t:28s} {w:5.2f} {len(plist):6d} {bm.idf(t):7.3f}   " + ("  ".join(shown) or "-"))

    scores = bm.score(terms)
    if not scores:
        print("\nno document contains a query term")
        return
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    print("\n-- top 5 by BM25F alone --")
    for i, (d, s) in enumerate(ranked[:5], 1):
        print(f"  {i}. {d:12s} {s:8.3f}")

    doc = args.doc or ranked[0][0]
    print(f"\n-- BM25F for document {doc}, term by term --")
    print("   tf~ = sum_z w_z * tf(t,d,z) / (1 - b + b * len(d,z) / avglen(z));  score += w_q * idf * tf~ (k1+1) / (tf~ + k1)")
    total = 0.0
    for t, w in terms.items():
        idf = bm.idf(t)
        if idf <= 0:
            continue
        parts, tft = [], 0.0
        for z, zw in bm.zone_weights.items():
            if zw <= 0 or z not in zidx.zones:
                continue
            tf = zidx.tf(t, doc, z)
            if tf:
                norm = bm._zone_norm(z).get(doc, 1.0)
                part = zw * tf / norm
                tft += part
                parts.append(f"{z}: {zw}*{tf}/{norm:.2f}={part:.2f}")
        if not tft:
            continue
        contrib = w * idf * tft * (bm.k1 + 1) / (tft + bm.k1)
        total += contrib
        print(f"  {t:24s} idf {idf:6.3f}  tf~ {tft:6.2f}  [{'; '.join(parts)}]  -> +{contrib:.3f}")
    print(f"  {'total':24s} {total:.3f}")


if __name__ == "__main__":
    main()
