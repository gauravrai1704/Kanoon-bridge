"""Agent v2 (router, verifier, issues, memo) and RAG v2 (structured answers, claim verification,
correction) on hand-made inputs. The engine-level hand-offs are in test_layers_integration.py."""

from __future__ import annotations

from datetime import date

from kanoon_bridge.rag.chunker import Chunk
from kanoon_bridge.rag.generate import parse_answer
from kanoon_bridge.schema import AnalyzedQuery, Code, Document, DocType, Paragraph, Query, ScoredDoc

MURDER = "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine."


def _statute(ref, title, text):
    code, _, num = ref.partition(":")
    return Document(ref, DocType.STATUTE, title=title, code=Code(code), section=num, meta={"ref": ref},
                    paragraphs=[Paragraph(f"Section {num}.", "statute", -1), Paragraph(text, "statute", 0)])


DOCS = {"bns:103": _statute("bns:103", "Punishment for murder", MURDER)}
CHUNKS = [Chunk(1, "bns:103", "statute", MURDER, 1, 1.0, "BNS Section 103 - Punishment for murder"),
          Chunk(2, "P1", "ratio", "We hold that the offence of murder under Section 302 IPC is made out on these facts.",
                1, 1.0, "State v. Arjun")]


# --------------------------------------------------------------------------- router


def test_router_kinds_and_family():
    from kanoon_bridge.agent.router import refine_family, route

    aq = AnalyzedQuery(Query("x"), tokens=["bail"], sections=[])
    assert route(Query(" ".join(["word"] * 80)), aq).kind == "case_text"
    sec = AnalyzedQuery(Query("cases under section 482 BNSS"), tokens=["case", "sec:bnss:482"], sections=["bnss:482"])
    r = route(Query("cases under section 482 BNSS"), sec)
    assert r.kind == "section_lookup" and r.family == "procedure" and "boolean" not in r.rules
    q = Query("false evidence")
    fact = AnalyzedQuery(q, tokens=["false", "evid"])
    rt = route(q, fact)
    assert rt.kind == "fact_question" and rt.decompose
    # cue words alone would say "evidence"; the retrieved statutes say penal
    fam, _ = refine_family(rt, q, fact, ["bns:229", "ipc:193", "bns:227"])
    assert fam == "penal"
    stab = Query("my brother was stabbed with a knife")
    fam, _ = refine_family(rt, stab, AnalyzedQuery(stab), ["bnss:435", "bnss:220"],
                           precedent_codes=[["ipc"], ["ipc", "crpc"], ["ipc"]])
    assert fam == "penal"                                   # cue words + what the precedents cite


# --------------------------------------------------------------------------- verifier


class _Bridge:
    statute_terms = {"bns:318": ["sec:bns:318"], "bns:319": ["sec:bns:319"], "bsa:118": ["sec:bsa:118"],
                     "bns:80": ["sec:bns:80"], "bns:3": ["sec:bns:3"], "bns:103": ["sec:bns:103"]}


class _Engine:
    bridge = _Bridge()


TITLES = {"bns:318": "Cheating", "bns:319": "Cheating by personation", "bsa:118": "Presumption as to dowry death",
          "bns:80": "Dowry death", "bns:3": "General explanations", "bns:103": "Punishment for murder"}


def _hits(*ids):
    return [ScoredDoc(d, 1.0 / (i + 1), DocType.STATUTE, rank=i + 1) for i, d in enumerate(ids)]


def test_verifier_family_specificity_general():
    from kanoon_bridge.agent.verify import StatuteVerifier, specificity

    v = StatuteVerifier()
    aq = AnalyzedQuery(Query("x"), code_in_force=Code.BNS)
    out, notes = v.rerank(_hits("bns:319", "bns:318"), Query("cheating"), aq, "penal", _Engine(), TITLES.get)
    assert out[0].doc_id == "bns:318" and notes and "matches the question's words" in notes[0]
    out, notes = v.rerank(_hits("bsa:118", "bns:80"), Query("dowry death"), aq, "penal", _Engine(), TITLES.get)
    assert out[0].doc_id == "bns:80" and "evidence law" in notes[0]
    out, _ = v.rerank(_hits("bns:3", "bns:103"), Query("he stabbed him and he died"), aq, "penal", _Engine(), TITLES.get)
    assert out[0].doc_id == "bns:103"                       # definitions do not answer a fact question
    assert specificity("police arrested me", "When police may arrest without warrant") > 0   # stemmed


# --------------------------------------------------------------------------- issues + memo


def test_issues_and_memo():
    from kanoon_bridge.agent.issues import decompose, find_evidence
    from kanoon_bridge.agent.memo import LawEntry, ResearchMemo

    text = ("(1) Whoever, with intent to cause death, causes bodily injury by a knife commits the offence. "
            "(2) Whoever dishonestly misappropriates movable property of a deceased person commits the offence.")
    analyze = lambda t: [w.lower().strip(".,") for w in t.split() if len(w) > 3]  # noqa: E731
    idf = lambda t: 2.0  # noqa: E731
    issues = decompose(text, analyze, idf)
    assert len(issues) == 2 and all(len(i.terms) >= 2 for i in issues)
    docs = {"P1": Document("P1", DocType.PRECEDENT, title="State v. A",
                           paragraphs=[Paragraph("The knife caused bodily injury with intent to cause death.", "ratio", 0)])}
    cov = find_evidence(issues, ["P1"], docs, analyze)
    assert cov == 0.5 and issues[0].covered and not issues[1].covered
    memo = ResearchMemo("q", "fact_question", "bns", [LawEntry("bns:103", "Punishment for murder", True, "", "death")],
                        issues, cov, 3)
    text = memo.render()
    assert "BNS 103" in text and "[OPEN]" in text and "coverage 50%" in text
    assert memo.to_dict()["open_issues"] == [issues[1].label]


# --------------------------------------------------------------------------- RAG v2


def test_structured_answer_computes_punishment():
    from kanoon_bridge.rag.structured import punishment_by_subsection, structured

    ans = structured("What is the punishment for murder?", CHUNKS, DOCS, "bns")
    assert ans.generator == "structured"
    assert "BNS 103" in ans.text and "death, life imprisonment, fine" in ans.text and "[1]" in ans.text
    assert "the court said" in ans.text and "[2]" in ans.text
    parts = punishment_by_subsection("(1) Whoever ... (2) Whoever, except in the case provided for by sub-section (2) "
                                     "of section 122, shall be punished with imprisonment for a term which may extend "
                                     "to seven years. (3) Whoever ... shall be punished with rigorous imprisonment for "
                                     "a term which shall not be less than ten years.")
    assert parts and "sub-section (2)" in parts[0] and "sub-section (3)" in parts[0] and "sub-section (1)" not in parts[0]


def test_claim_verifier_catches_injected_errors():
    from kanoon_bridge.rag.claim_check import ClaimVerifier, quantities

    assert ("years", 7.0) in quantities("may extend to seven years") and ("rupees", 10000.0) in quantities("₹10,000")
    v = ClaimVerifier(None, 0.2, None, DOCS)
    by_n = {c.n: c for c in CHUNKS}

    def verdict(s):
        return v.check_sentence(s, 1, by_n, "bns").verdict

    assert verdict("It is punishable with death, life imprisonment, fine. [1]") == "supported"
    assert verdict("It is punishable with imprisonment up to 10 years. [1]") == "contradicted"     # quantity
    assert verdict("Murder is not punishable with death. [1]") == "contradicted"                    # negation
    assert verdict("The provision is BNS 104, punishment for murder. [1]") == "unsupported"         # section
    assert verdict("The provision is IPC 302, punishment for murder. [1]") in ("contradicted", "unsupported")
    # a quotation of a judgment is checked for being verbatim, never for its code
    q = v.check_sentence('The court said: "We hold that the offence of murder under Section 302 IPC is made out". [2]',
                         2, by_n, "bns")
    assert q.verdict == "supported"
    bad = v.check_sentence('The court said: "We hold that the offence of murder under Section 304 IPC is made out". [2]',
                           2, by_n, "bns")
    assert bad.verdict == "contradicted" and "not in [2]" in bad.reasons[0]


def test_correct_repairs_and_records():
    from kanoon_bridge.rag.claim_check import ClaimVerifier
    from kanoon_bridge.rag.correct import correct

    class Norm:
        def equivalents(self, ref):
            return {"ipc:302": ["bns:103"]}.get(ref, [])

    ans = parse_answer("It is punishable with imprisonment up to 10 years. [1] Murder is covered by IPC 302 and is "
                       "punishable with death. [1] The moon is made of cheese. [1]", "test")
    ans.chunks = CHUNKS
    v = ClaimVerifier(None, 0.2, Norm(), DOCS)
    v.check(ans, CHUNKS, "bns")
    assert sum(c.verdict != "supported" for c in ans.claims) == 3
    fixed = correct(ans, CHUNKS, "bns", Norm(), v)
    assert "BNS 103" in fixed.text and "IPC 302" not in fixed.text          # counterpart in force
    assert "cheese" not in fixed.text                                         # unsupported removed
    assert "10 years" not in fixed.text and "Whoever commits murder" in fixed.text   # source's own words
    assert len(fixed.corrections) == 3
    v.check(fixed, CHUNKS, "bns")
    assert all(c.verdict == "supported" for c in fixed.claims)


def test_open_eval_scores_by_reference():
    from kanoon_bridge.eval.open_eval import load_rows

    rows = load_rows("e7_temporal", limit=4)
    assert rows and all("expected_code" in r and "offence" in r for r in rows)
    assert date.fromisoformat(rows[0]["incident_date"])
