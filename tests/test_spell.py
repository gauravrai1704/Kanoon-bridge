"""Tolerant retrieval: edit distance, k-gram index, speller, wildcards, typo simulation."""

import random

from kanoon_bridge.text.spell import KGramIndex, Speller, damerau_levenshtein


def test_damerau_levenshtein():
    assert damerau_levenshtein("murdr", "murder") == 1
    assert damerau_levenshtein("ansewr", "answer") == 1          # adjacent transposition = 1 edit
    assert damerau_levenshtein("bail", "jail") == 1
    assert damerau_levenshtein("abc", "xyz", max_dist=1) == 2       # early exit returns max + 1


def test_kgram_candidates_and_wildcards():
    idx = KGramIndex.build(["murder", "murderer", "border", "extortion", "extort", "export"])
    cands = {t for t, _ in idx.candidates("murdr")}
    assert "murder" in cands and "export" not in cands
    assert idx.wildcard("extort*") == {"extort", "extortion"}
    assert idx.wildcard("*order") == {"border"}
    assert idx.wildcard("mur*er") == {"murder", "murderer"}


def _speller():
    df = {"murder": 40, "knife": 12, "dowri": 9, "punish": 300, "bail": 50, "jail": 1, "anticipatori": 5}
    surface = {"display": {"dowri": "dowry", "punish": "punished", "anticipatori": "anticipatory"},
               "words": {"punishment": "punish", "punished": "punish", "dowry": "dowri", "anticipatory": "anticipatori"}}
    return Speller.build(df, surface)


def test_speller_corrections():
    sp = _speller()
    c = sp.check("murdr", "murdr")
    assert c.term == "murder" and c.applied and c.edits == 1
    c = sp.check("punishmnt", "punishmnt")                          # whole-word match beats the stem
    assert c.term == "punish" and c.display == "punishment"
    c = sp.check("dowery", "doweri")
    assert c.term == "dowri" and c.display == "dowry"
    assert sp.check("bail", "bail") is None                          # known word
    assert sp.check("kni", "kni") is None                            # too short
    rare = sp.check("jail", "jail")                                   # rare word with a frequent neighbour
    assert rare is not None and rare.term == "bail" and not rare.applied


def test_speller_wildcard_expansion():
    sp = _speller()
    assert sp.expand_wildcard("antic*") == ["anticipatori"]
    assert sp.expand_wildcard("*") == []


def test_typo_simulation_is_deterministic():
    from kanoon_bridge.eval.typos import corrupt

    a = corrupt("Punishment for criminal intimidation", random.Random(5))
    b = corrupt("Punishment for criminal intimidation", random.Random(5))
    assert a == b and a != "Punishment for criminal intimidation"
