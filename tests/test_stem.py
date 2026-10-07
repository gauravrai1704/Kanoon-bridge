"""Stemming: Porter for English, light suffix stripping for Hindi.  [C]"""

from kanoon_bridge.text.stem import stem_hindi, stem_tokens


def test_hindi_suffixes():
    assert stem_hindi("लड़कियों") == "लड़क"
    assert stem_hindi("मारता") == "मार"
    assert stem_hindi("गवाहों") == "गवाह"
    assert stem_hindi("दहेज") == "दहेज"                      # nothing to strip


def test_min_stem_length():
    assert stem_hindi("को") == "को"                          # would leave one character


def test_mixed_tokens():
    assert stem_tokens(["murders", "हत्याओं", "sec:ipc:302", "302"]) == ["murder", "हत्य", "sec:ipc:302", "302"]
