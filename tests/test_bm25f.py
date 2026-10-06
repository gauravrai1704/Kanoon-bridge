"""BM25F and tf-idf on the toy corpus.  [D]"""

from conftest import todo

from kanoon_bridge.config import load_config
from kanoon_bridge.index.zones import ZoneIndex
from kanoon_bridge.rank.bm25f import BM25F
from kanoon_bridge.rank.topk import top_k
from kanoon_bridge.text.pipeline import TextOptions, analyze_text


def analyze(text, doc):
    return analyze_text(text, options=TextOptions(resolve_collisions=False, version_normalise=False))


@todo
def test_bm25f_ranks_murder_doc_first(toy_docs):
    zidx = ZoneIndex.build(toy_docs, analyze)
    scorer = BM25F.from_config(zidx, load_config())
    terms = {t: 1.0 for t in analyze("stabbed with a knife murder", None)}
    assert top_k(scorer.score(terms), 1)[0][0] == "p1"


@todo
def test_ratio_zone_weight_matters(toy_docs):
    zidx = ZoneIndex.build(toy_docs, analyze)
    cfg = load_config()
    heavy = BM25F(zidx, {**cfg.zones, "ratio": 5.0}, cfg.bm25.k1, cfg.bm25.b)
    light = BM25F(zidx, {**cfg.zones, "ratio": 0.1}, cfg.bm25.k1, cfg.bm25.b)
    terms = {"intimid": 1.0}
    assert heavy.score(terms)["p3"] > light.score(terms)["p3"]


@todo
def test_idf_never_negative(toy_docs):
    zidx = ZoneIndex.build(toy_docs, analyze)
    assert BM25F.from_config(zidx, load_config()).idf("accus") >= 0
