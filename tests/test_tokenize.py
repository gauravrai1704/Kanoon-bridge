"""Tokeniser: section mentions must become single tokens with their code.  [A]"""

from kanoon_bridge.schema import Code
from kanoon_bridge.text.tokenize import extract_sections, tokenize


def test_section_with_code_after():
    assert "sec:ipc:302" in tokenize("Convicted under Section 302 IPC.")


def test_us_abbreviation():
    assert "sec:ipc:498a" in tokenize("Charged u/s 498A IPC")


def test_trailing_form():
    assert "sec:bns:103" in tokenize("the offence is 103 BNS now")


def test_subsection_kept():
    assert "sec:bns:103(1)" in tokenize("Section 103(1) of the Bharatiya Nyaya Sanhita")


def test_full_act_name():
    mentions = extract_sections("sections 302, 307 and 34 of the Indian Penal Code")
    assert [m.section for m in mentions] == ["302", "307", "34"]
    assert all(m.code == Code.IPC for m in mentions)


def test_bare_number_has_unknown_code():
    assert "sec:?:302" in tokenize("What is the punishment under section 302?")


def test_bnss_not_confused_with_bns():
    assert "sec:bnss:483" in tokenize("bail under Section 483 BNSS")


def test_plain_words_lowercased():
    assert tokenize("Murder KNIFE") == ["murder", "knife"]


def test_devanagari_kept():
    assert "हत्या" in tokenize("हत्या का मामला")
