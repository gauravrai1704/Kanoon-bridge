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


def test_code_first_forms():
    assert tokenize("punishment for murder, BNS 103")[-1] == "sec:bns:103"
    assert tokenize("IPC section 302 and 34") == ["sec:ipc:302", "sec:ipc:34"]
    assert tokenize("bns 103(1)") == ["sec:bns:103(1)"]


def test_year_is_not_a_section():
    assert "sec:bns:2023" not in tokenize("the BNS, 2023 came into force")


def test_indian_kanoon_style_title():
    assert tokenize("Section 302 in The Indian Penal Code")[0] == "sec:ipc:302"


def test_read_with_and_ranges():
    from kanoon_bridge.text.tokenize import tokenize

    assert [t for t in tokenize("convicted u/s 302 r/w 34 IPC") if t.startswith("sec:")] == ["sec:ipc:302", "sec:ipc:34"]
    assert [t for t in tokenize("sections 302 read with 149 of the IPC") if t.startswith("sec:")] == ["sec:ipc:302", "sec:ipc:149"]
    assert [t for t in tokenize("sections 323-325 IPC") if t.startswith("sec:")] == ["sec:ipc:323", "sec:ipc:324", "sec:ipc:325"]
    assert [t for t in tokenize("Section 103(1) BNS") if t.startswith("sec:")] == ["sec:bns:103(1)"]
    assert "2019" in tokenize("in 2019-20 the accused")
