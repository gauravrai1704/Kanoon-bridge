"""Layer 3 (RAG) checks, offline: no API key, no index needed."""

from __future__ import annotations

from datetime import date

import pytest

from kanoon_bridge.rag import abstain, chunker, citation_check, generate, version_check
from kanoon_bridge.rag._util import split_sentences
from kanoon_bridge.rag.chunker import Chunk
from kanoon_bridge.rag.generate import Answer, parse_answer
from kanoon_bridge.schema import Document, DocType, Paragraph, ScoredDoc


class _Result:
    def __init__(self, statutes, precedents, text="punishment for murder"):
        from kanoon_bridge.schema import Query

        self.query = Query(text, incident_date="2025-01-03")
        self.statutes, self.precedents = statutes, precedents


def _docs():
    bns = Document("bns:103", DocType.STATUTE, title="Punishment for murder", meta={"ref": "bns:103"},
                   paragraphs=[Paragraph("Section 103 BNS. Punishment for murder.", "statute", -1),
                               Paragraph("(1) Whoever commits murder shall be punished with death or "
                                         "imprisonment for life, and shall also be liable to fine.", "statute", 0)])
    p = Document("P1", DocType.PRECEDENT, title="State v. A",
                 paragraphs=[Paragraph("The accused was seen near the house at night.", "facts", 0),
                             Paragraph("We hold that the stab wounds show an intention to cause death. "
                                       "The conviction for murder is upheld.", "ratio", 1)])
    return {d.doc_id: d for d in (bns, p)}


def test_sentence_split_keeps_citation_with_sentence():
    s = split_sentences("Murder is punished under BNS 103. [1] The court upheld it [2]. Done.")
    assert s[0].endswith("[1]") and s[1].endswith("[2].")


def test_chunks_statutes_first_and_label():
    res = _Result([ScoredDoc("bns:103", 3.0)], [ScoredDoc("P1", 1.0)])
    chunks = chunker.make_chunks(res, _docs(), max_chunks=4, query_terms=["murder", "punish"])
    assert [c.doc_id for c in chunks] == ["bns:103", "P1"]
    assert chunks[0].title == "BNS Section 103 - Punishment for murder"
    assert not chunks[0].text.startswith("Section 103 BNS")        # heading line dropped
    assert chunks[1].zone == "ratio"


def test_truncate_at_sentence_boundary():
    out = chunker.truncate("One two three. Four five six. Seven eight nine.", 30)
    assert out == "One two three. Four five six."


def test_extractive_answer_is_cited_and_supported():
    res = _Result([ScoredDoc("bns:103", 3.0)], [ScoredDoc("P1", 1.0)])
    chunks = chunker.make_chunks(res, _docs(), query_terms=["murder"])
    ans = generate.generate("punishment for murder", chunks, generator="extractive")
    assert ans.sentences and all(n is not None for _, n in ans.sentences)
    ans = citation_check.check(ans, chunks, None, threshold=0.2)
    assert ans.supported_rate == 1.0 and not ans.flags


def test_citation_check_flags_wrong_or_missing_citation():
    chunks = [Chunk(1, "bns:103", "statute", "Whoever commits murder shall be punished with death.", 1, 1.0),
              Chunk(2, "bns:303", "statute", "Whoever commits theft shall be punished.", 2, 0.5)]
    ans = parse_answer("Murder is punished with death [2]. Theft is an offence.")
    ans = citation_check.check(ans, chunks, None, threshold=0.3)
    assert len(ans.flags) == 2 and ans.supported_rate == 0.0
    ok = citation_check.check(parse_answer("Murder is punished with death [1]."), chunks, None, threshold=0.3)
    assert not ok.flags


def test_version_check_wrong_code_and_bare_number():
    from kanoon_bridge.text.collision import CollisionResolver

    try:
        resolver = CollisionResolver.load()
    except FileNotFoundError:
        pytest.skip("crosswalk not generated")
    norm = resolver.normalizer
    ans = parse_answer("This is murder under Section 302 IPC [1]. Section 302 also applies [1].")
    ans = version_check.check(ans, date(2025, 1, 3), resolver, norm)
    assert any(f.startswith("wrong code: IPC 302") and "BNS 103" in f for f in ans.flags)
    assert any(f.startswith("bare number: 302") for f in ans.flags)
    old = version_check.check(parse_answer("Murder under Section 302 IPC [1]."), date(2020, 1, 1), resolver, norm)
    assert not old.flags


def test_abstain_on_empty_and_on_unspecific_query():
    empty = _Result([], [])
    assert abstain.decide(empty, {"murder": 2.0}, query_terms=["murder"])[0]
    weak = _Result([ScoredDoc("a", 1.0)], [ScoredDoc("b", 1.0)])
    assert abstain.decide(weak, {"the": 0.01}, query_terms=["the"])[0]
    clear = _Result([ScoredDoc("a", 5.0), ScoredDoc("b", 1.0)], [ScoredDoc("c", 3.0), ScoredDoc("d", 1.0)])
    assert not abstain.decide(clear, {"murder": 2.0}, query_terms=["murder"])[0]


def test_auto_generator_without_key_is_extractive(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(generate, "load_env", lambda path=None: None)
    assert generate.pick_generator("auto") == "extractive"


def test_claude_path_prompt_and_parse(monkeypatch):
    seen = {}

    def fake(prompt, model=None, max_tokens=600, system=generate.SYSTEM):
        seen["prompt"] = prompt
        return "Murder is punished under BNS Section 103 [1]. The court upheld a murder conviction [2]."

    monkeypatch.setattr(generate, "call_claude", fake)
    res = _Result([ScoredDoc("bns:103", 3.0)], [ScoredDoc("P1", 1.0)])
    chunks = chunker.make_chunks(res, _docs(), query_terms=["murder"])
    ans = generate.generate("punishment for murder", chunks, "2025-01-03", "delhi", generator="claude")
    assert "[1] BNS Section 103" in seen["prompt"] and "2025-01-03" in seen["prompt"]
    assert [n for _, n in ans.sentences] == [1, 2] and ans.generator == "claude"
