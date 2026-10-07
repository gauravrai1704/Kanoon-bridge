"""Word-trigram BM25 channel (rank/ngram.py)."""

from kanoon_bridge.rank.ngram import NgramIndex, words
from kanoon_bridge.schema import Document, DocType, Paragraph


def _doc(i, text):
    return Document(doc_id=str(i), doc_type=DocType.PRECEDENT, title="", paragraphs=[Paragraph(text, "facts", 0)])


def test_shared_phrasing_wins():
    idx = NgramIndex.build([_doc(0, "the accused stabbed the victim with a knife"),
                            _doc(1, "the victim was stabbed by the accused"),
                            _doc(2, "bail is the rule and jail the exception")])
    s = idx.score("he stabbed the victim with a knife near the house")
    assert max(s, key=s.get) == "0"
    assert "2" not in s


def test_masks_and_unknown_words_break_ngrams():
    assert words("under [SECTION] of [ACT] the Court") == ["under", "of", "the", "court"]
    idx = NgramIndex.build([_doc(0, "rule of law prevails"), _doc(1, "law and order")])
    assert idx.score("rule zzz of law") == {}            # 'zzz' is unseen: no trigram survives
    assert idx.score("the rule of law") .keys() == {"0"}


def test_candidates_filter():
    idx = NgramIndex.build([_doc(0, "a b c d"), _doc(1, "a b c e")])
    assert set(idx.score("a b c", candidates={"1"})) == {"1"}


def test_ilpcsr_baseline_ranks_by_trigrams_alone():
    from kanoon_bridge.search import SearchEngine, SearchOptions

    opt = SearchOptions.ilpcsr_baseline()
    assert opt.ngram and opt.ngram_only and not opt.version_norm and not opt.bridge and not opt.ltr
    eng = SearchEngine.__new__(SearchEngine)
    eng.cfg = type("C", (), {"ngram": type("N", (), {"min_query_words": 50})()})()
    from kanoon_bridge.schema import Query

    assert eng._use_ngram(Query("a short typed question"), opt)          # any length, unlike the full system
    assert not eng._use_ngram(Query("a short typed question"), SearchOptions.full())
