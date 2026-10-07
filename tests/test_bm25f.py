"""BM25F and tf-idf.  [Gaurav]

`manual_zone_index` fills ZoneIndex's documented data structures directly, so these tests do
not depend on ZoneIndex.build (Sharanya's). `test_*_with_real_build` re-run the key check through
the real build once it exists (skipped until then).
"""

import math

from conftest import todo

from kanoon_bridge.config import load_config
from kanoon_bridge.index.positional import PositionalIndex
from kanoon_bridge.index.zones import ZoneIndex
from kanoon_bridge.rank.bm25f import BM25F
from kanoon_bridge.rank.topk import top_k
from kanoon_bridge.rank.vsm import TfidfScorer
from kanoon_bridge.text.pipeline import TextOptions, analyze_text

OPTS = TextOptions(resolve_collisions=False, version_normalise=False)


def analyze(text, doc=None):
    return analyze_text(text, options=OPTS)


def _fill(idx: PositionalIndex, doc_id: str, tokens: list[str]) -> None:
    for i, t in enumerate(tokens):
        idx.postings[t].setdefault(doc_id, []).append(i)
    idx.doc_len[doc_id] = len(tokens)


def manual_zone_index(docs) -> ZoneIndex:
    z = ZoneIndex()
    for d in docs:
        whole = []
        per_zone: dict[str, list[str]] = {}
        for p in d.paragraphs:
            toks = analyze(p.text)
            per_zone.setdefault(p.zone, []).extend(toks)
            whole += toks
        for zone, toks in per_zone.items():
            _fill(z.zones[zone], d.doc_id, toks)
        _fill(z.whole, d.doc_id, whole)
        z.doc_ids.append(d.doc_id)
    return z


def test_bm25f_ranks_murder_doc_first(toy_docs):
    scorer = BM25F.from_config(manual_zone_index(toy_docs), load_config())
    terms = {t: 1.0 for t in analyze("stabbed with a knife murder")}
    assert top_k(scorer.score(terms), 1)[0][0] == "p1"


def test_ratio_zone_weight_matters(toy_docs):
    zidx = manual_zone_index(toy_docs)
    cfg = load_config()
    heavy = BM25F(zidx, {**cfg.zones, "ratio": 5.0}, cfg.bm25.k1, cfg.bm25.b)
    light = BM25F(zidx, {**cfg.zones, "ratio": 0.1}, cfg.bm25.k1, cfg.bm25.b)
    term = analyze("intimidation")[0]
    assert heavy.score({term: 1.0})["p3"] > light.score({term: 1.0})["p3"]
    assert "ratio" in heavy.last_breakdown["p3"]


def test_idf_formula_and_never_negative(toy_docs):
    zidx = manual_zone_index(toy_docs)
    bm = BM25F.from_config(zidx, load_config())
    term = analyze("accused")[0]                 # in 2 of the 3 docs
    n, df = 3, zidx.df(term)
    assert math.isclose(bm.idf(term), math.log((n - df + 0.5) / (df + 0.5) + 1))
    assert bm.idf(term) >= 0 and bm.idf("unseen-term") == 0


def test_candidates_restrict_scoring(toy_docs):
    bm = BM25F.from_config(manual_zone_index(toy_docs), load_config())
    terms = {t: 1.0 for t in analyze("accused dowry")}
    assert set(bm.score(terms)) == {"p1", "p2", "p3"}
    assert set(bm.score(terms, candidates={"p2"})) == {"p2"}


def test_plain_bm25_without_zones(toy_docs):
    bm = BM25F.from_config(manual_zone_index(toy_docs), load_config(), use_zones=False)
    terms = {t: 1.0 for t in analyze("dowry")}
    assert top_k(bm.score(terms), 1)[0][0] == "p2"


def test_long_queries_keep_highest_idf_terms(toy_docs):
    bm = BM25F.from_config(manual_zone_index(toy_docs), load_config())
    bm.max_query_terms = 1
    kept = bm._select_terms({analyze("accused")[0]: 1.0, analyze("dowry")[0]: 1.0})
    assert list(kept) == [analyze("dowry")[0]]   # rarer term wins


def test_tfidf_lnc_ltc(toy_docs):
    zidx = manual_zone_index(toy_docs)
    vsm = TfidfScorer(zidx.whole).prepare()
    scores = vsm.score({t: 1.0 for t in analyze("dowry death")})
    assert max(scores, key=scores.get) == "p2"
    assert all(0 < s <= 1.0 + 1e-9 for s in scores.values())   # cosine


@todo
def test_bm25f_with_real_build(toy_docs):
    zidx = ZoneIndex.build(toy_docs, lambda text, doc: analyze(text))
    scorer = BM25F.from_config(zidx, load_config())
    assert top_k(scorer.score({t: 1.0 for t in analyze("stabbed knife murder")}), 1)[0][0] == "p1"
