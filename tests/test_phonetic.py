"""Phonetic matching and transliteration for Hinglish.  [C]"""

from conftest import todo

from kanoon_bridge.text.phonetic import hindi_soundex, soundex
from kanoon_bridge.text.transliterate import detect_lang, normalize_roman


def test_classic_soundex_baseline():
    assert soundex("Robert") == soundex("Rupert") == "R163"


def test_detect_lang():
    assert detect_lang("मेरे भाई को चाकू मारा") == "hi"
    assert detect_lang("mere bhai ko chaku maara") == "hinglish"
    assert detect_lang("my brother was stabbed") == "en"


@todo
def test_hatya_variants_match():
    assert hindi_soundex("hatya") == hindi_soundex("hathya") == hindi_soundex("hattya")


@todo
def test_distinct_words_stay_distinct():
    assert hindi_soundex("chori") != hindi_soundex("dhokha")


@todo
def test_normalize_roman_vowel_length():
    assert normalize_roman("maaraa") == normalize_roman("mara")
