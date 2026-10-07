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


def test_devanagari_and_roman_meet():
    from kanoon_bridge.text.transliterate import devanagari_to_roman

    def norm(text):
        return " ".join(normalize_roman(w) for w in text.split())

    assert norm(devanagari_to_roman("मेरे भाई को चाकू मारा")) == norm("mere bhai ko chaku maara")
    assert devanagari_to_roman("दहेज") == "dahej"                 # final schwa dropped
    assert devanagari_to_roman("ज़मानत") == "zamaanat"            # nukta
    assert devanagari_to_roman("धारा 302") == "dhaaraa 302"


def test_normalisation_rules():
    for a, b in [("zamanat", "jamanat"), ("chhura", "chura"), ("qatl", "katl"), ("wakil", "vakil"),
                 ("chouri", "chori"), ("phaansi", "fansi"), ("giraftaari", "giraftari")]:
        assert normalize_roman(a) == normalize_roman(b), (a, b)


def test_lexicon_lookup_layers():
    from kanoon_bridge.text.transliterate import LegalLexicon

    lex = LegalLexicon.load()
    assert ("murder", 1.0) in lex.lookup("hatya")
    assert ("murder", 1.0) in lex.lookup("hathya")                # normalised spelling
    assert ("knife", 0.8) in lex.lookup("चाकू")                   # Devanagari
    assert lex.lookup("atmhatya", use_phonetic=False) == []
    assert ("suicide", 0.8) in lex.lookup("atmhatya")             # phonetic fallback, weight x 0.8
    assert lex.lookup("mere") == []                               # short words never phonetic-match


def test_phonetic_index():
    from kanoon_bridge.text.phonetic import PhoneticIndex

    idx = PhoneticIndex.build(["ramesh", "rameshh", "suresh", "sec:ipc:302", "murder"])
    assert idx.matches("ramesh") == {"rameshh"}
    assert "sec:ipc:302" not in {w for b in idx.buckets.values() for w in b}


def test_detect_lang_hinglish_variants():
    assert detect_lang("padosi ne jaan se maarne ki dhamki di") == "hinglish"
    assert detect_lang("the main issue is bail") == "en"            # English words that clash
