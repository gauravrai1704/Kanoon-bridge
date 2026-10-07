"""Train the learning-to-rank re-ranker (rank/ltr.py) on IL-PCSR validation queries.  [owner: Gaurav]

    python scripts/06_train_ltr.py                 # all 627 val queries, top-100 candidates each
    python scripts/06_train_ltr.py --limit 200     # quicker
    python scripts/06_train_ltr.py --sample        # synthetic sample (pipeline check only)
    python scripts/06_train_ltr.py --short         # the typed-question model (ltr_short.json): trained on
                                                   # 40-word facts excerpts of VAL judgments. Not train: the
                                                   # citation graph behind `authority` is built from train
                                                   # qrels, so train queries would over-reward authority.
                                                   # bridge / offence_match / binding are kept >= 0.

Prints 5-fold cross-validated MAP (hand-set weights vs learned) on val, then trains on all of
val and saves data/processed/index/ltr.json. Test queries are never touched; the eval's "full"
system and the "+ltr" ablation step use the saved model.
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--short", action="store_true", help="train the typed-question model on val-split 40-word excerpts")
    args = ap.parse_args()
    if args.sample:
        os.environ["KB_ILPCSR_DIR"] = "tests/data/ilpcsr_sample"

    from kanoon_bridge.config import load_config
    from kanoon_bridge.eval.run_eval import _ilpcsr_queries
    from kanoon_bridge.index import store
    from kanoon_bridge.ingest import load_ilpcsr
    from kanoon_bridge.rank.ltr import FEATURES, NONNEG, LinearRanker, coordinate_ascent, cross_validate, feature_rows
    from kanoon_bridge.search import SearchEngine, SearchOptions

    cfg = load_config()
    ev = load_config("eval.yaml")
    engine = SearchEngine.load(cfg)
    splits = ["val", "train"] if args.sample else ["val"]
    limit = args.limit
    queries = [q for sp in splits for q in _ilpcsr_queries(sp, cfg, short=args.short)][:limit]
    qrels = {k: v for sp in splits for k, v in load_ilpcsr.load_qrels(sp, "precedent", cfg).items()}
    opt = SearchOptions(ngram=True, top_k=args.depth,
                        max_query_terms=None if args.short else ev.get("e1_max_query_terms", 100),
                        dense=engine.dense is not None, qpp=engine.dense is not None)

    t0 = time.time()
    data = []
    for q in queries:
        rels = qrels.get(q.query_id, {})
        n_rel = sum(1 for g in rels.values() if g > 0)
        if not n_rel:
            continue
        res = engine.search(copy.deepcopy(q), opt)
        if not res.precedents:
            continue
        X = feature_rows(res.precedents, res.query, engine)
        y = np.array([rels.get(h.doc_id, 0) for h in res.precedents], dtype=np.float64)
        data.append((X, y, n_rel))
    print(f"features for {len(data)} {'+'.join(splits)} queries in {time.time() - t0:.1f}s "
          f"({sum(int((y > 0).sum()) for _, y, _ in data)} relevant docs inside the top {args.depth})")
    if len(data) < 2:
        print("not enough judged queries to train - no model saved")
        return

    folds = min(args.folds, len(data))
    kw = {"nonneg": NONNEG} if args.short else {}
    cv = cross_validate(data, folds=folds, **kw)
    print(f"{cv['folds']}-fold CV MAP on {'+'.join(splits)} (top-{args.depth} re-ranking): hand-set {cv['cv_map_handset']:.4f} "
          f"-> learned {cv['cv_map_ltr']:.4f}")
    w, train_map = coordinate_ascent(data, **kw)
    model = LinearRanker(weights=w, depth=args.depth,
                         info={"trained_on": "il-pcsr val, 40-word facts excerpts; bridge/offence_match/binding >= 0" if args.short else "il-pcsr val",
                               "queries": len(data), "train_map": round(train_map, 4), "dense": engine.dense is not None,
                               **{k: round(v, 4) if isinstance(v, float) else v for k, v in cv.items()}})
    name = "ltr_short" if args.short else "ltr"
    store.save(model.to_dict(), name, "json")
    print("weights:")
    for f, x in zip(FEATURES, w):
        print(f"  {f:15s} {x:+.3f}")
    print("saved", store.index_dir() / f"{name}.json")


if __name__ == "__main__":
    main()
