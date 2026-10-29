"""E14-E16: the agent v2 and RAG v2 layers, on the open corpus (make open; no IL-PCSR, no API key).  [owner: Gaurav — working]

E14  punishment QA, end to end. One question per BNS section whose source gives a one-line
     punishment summary: "What is the punishment for <section heading>?" (incident date
     2025-01-01). Gold: the summary, parsed (bns-study-platform's `punishment` field, written
     independently of the bare-act text the answers come from). Per system:
         statute@1       the answer's law is the right BNS section
         punishment_ok   the punishment the answer states equals the gold on every field
                         (death, life, maximum and minimum term, fine and its limits,
                         community service), over all questions
         punishment_ok|statute   the same, over questions whose statute was right
         flags/answer, corrections/answer
     Systems: core + RAG v1 (extractive quote), core + RAG v2 (structured), agent v2 + RAG v2.
     Caveat: the questions use section headings, which favours heading matching.

E15  verification by error injection. Base answers: the E14 answers of RAG v2 (structured)
     and RAG v1 (extractive) whose statute is right. Each base answer gets one error per type,
     in the sentence that cites the statute:
         years      a term of years changed (7 -> 10)          fine     a fine amount x 10
         negation   "punishable" -> "not punishable" / "shall be" -> "shall not be"
         wrong_code the section named in its old-code form (BNS 103 -> IPC 302) for a 2025 incident
         section    the section number changed to one the source does not name
         unsupported a punishment sentence of a different statute, citing this one
     Detection: the checker raises a new flag on the perturbed answer. Checkers: v1 (cosine
     support + code-in-force), v2 (claim verification + code-in-force). False alarms: flags on
     the clean answers.

E16  the agent on the statute-level sets (eval/open_eval.py): core, agent v1, agent v2 and v2
     without each of its parts (verification, citation votes, family search).
"""

from __future__ import annotations

import copy
import random
import re
from collections import defaultdict

from kanoon_bridge.drift.punishment import extract
from kanoon_bridge.rag.generate import Answer, parse_answer
from kanoon_bridge.schema import Query

FIELDS = ("death", "life", "max_years", "min_years", "fine", "fine_max", "fine_min", "community_service")
_NUM_YEARS = re.compile(r"\b(\d{1,2})\s+years?\b")
_RUPEES = re.compile(r"₹\s*([\d,]+)|\b([\d,]{3,})\s+rupees\b")


def punishment_questions(cfg=None, limit: int | None = None) -> list[dict]:
    from kanoon_bridge.drift import gold
    from kanoon_bridge.drift.provisions import load_bns

    summ = gold.bns_punishment_summaries(cfg)
    bns = load_bns(cfg)
    out = []
    for num, prov in bns.items():
        s = summ.get(num, "")
        if not s or s.lower().startswith("not applicable") or extract(s).empty or not prov.title:
            continue
        out.append({"id": f"P-{num}", "section": f"bns:{num}", "gold": s,
                    "text": f"What is the punishment for {prov.title.rstrip('.').lower()}?", "incident_date": "2025-01-01"})
    return out[:limit] if limit else out


def same_punishment(a, b) -> bool:
    return all((getattr(a, f) or None) == (getattr(b, f) or None) for f in FIELDS)


def answer_law_ref(ans: Answer, docs) -> str:
    """The statute the answer presents as the law: the first statute chunk it cites."""
    by_n = {c.n: c for c in ans.chunks}
    for _, n in ans.sentences:
        c = by_n.get(n)
        if c is not None and c.zone == "statute":
            doc = docs.get(c.doc_id) if hasattr(docs, "get") else None
            return ((getattr(doc, "meta", None) or {}).get("ref") or c.doc_id).lower()
    return ""


def answer_punishment(ans: Answer):
    """The punishment the answer states, read from its own words about the statute."""
    by_n = {c.n: c for c in ans.chunks}
    text = " ".join(s for s, n in ans.sentences if by_n.get(n) is not None and by_n[n].zone == "statute")
    return extract(text)


def punishment_qa(engine, rag, agent, questions: list[dict]) -> tuple[list[dict], dict]:
    from kanoon_bridge.search import SearchOptions

    systems = {"core + RAG v1": (lambda q: engine.search(q, SearchOptions.interactive()), 1),
               "core + RAG v2": (lambda q: engine.search(q, SearchOptions.interactive()), 2),
               "agent v2 + RAG v2": (lambda q: agent.run(q, SearchOptions.interactive()), 2)}
    rows, answers = [], {}
    for name, (search, version) in systems.items():
        acc = defaultdict(list)
        answers[name] = []
        for row in questions:
            q = Query.from_dict(row)
            res = search(q)
            ans = rag.answer(res, generator="extractive", version=version)
            gold = extract(row["gold"])
            ok_statute = answer_law_ref(ans, rag.docs) == row["section"]
            ok_pun = same_punishment(answer_punishment(ans), gold)
            acc["statute@1"].append(ok_statute)
            acc["punishment_ok"].append(ok_pun)
            if ok_statute:
                acc["punishment_ok|statute"].append(ok_pun)
            acc["flags/answer"].append(len(ans.flags))
            acc["corrections/answer"].append(len(getattr(ans, "corrections", [])))
            answers[name].append((row, ans, ok_statute))
        rows.append({"system": name, "n": len(questions),
                     **{k: round(sum(map(float, v)) / len(v), 4) for k, v in acc.items() if v},
                     "n_statute_ok": sum(acc["statute@1"])})
    return rows, answers


# --------------------------------------------------------------------------- E15


def _perturb(ans: Answer, kind: str, row: dict, normalizer, rng: random.Random, other_sentence: str) -> Answer | None:
    """The answer with one injected error in the sentence that cites the statute (None if n/a)."""
    by_n = {c.n: c for c in ans.chunks}
    idx = [i for i, (s, n) in enumerate(ans.sentences) if by_n.get(n) is not None and by_n[n].zone == "statute"]
    if not idx:
        return None
    sents = [s for s, _ in ans.sentences]
    law_n = ans.sentences[idx[0]][1]
    ref = row["section"]
    num = ref.split(":")[1]

    def with_sentence(i, new):
        out = sents[:]
        out[i] = new
        return out

    new_sents = None
    if kind == "years":
        for i in idx:
            m = _NUM_YEARS.search(sents[i]) or re.search(r"\b(one|two|three|five|seven|ten|fourteen|twenty)\s+years?\b", sents[i])
            if m:
                rep = "10 years" if m.group(1) not in ("10", "ten") else "3 years"
                new_sents = with_sentence(i, sents[i][: m.start()] + rep + sents[i][m.end():])
                break
    elif kind == "fine":
        for i in idx:
            m = _RUPEES.search(sents[i])
            if m:
                val = int((m.group(1) or m.group(2)).replace(",", "")) * 10
                new_sents = with_sentence(i, sents[i][: m.start()] + f"₹{val:,}" + sents[i][m.end():])
                break
            m = re.search(r"\b(one|two|five|ten|twenty|fifty) thousand rupees\b", sents[i])
            if m:
                new_sents = with_sentence(i, sents[i][: m.start()] + "one lakh rupees" + sents[i][m.end():])
                break
    elif kind == "negation":
        for i in idx:
            for a, b in (("It is punishable", "It is not punishable"), ("shall be punished", "shall not be punished"),
                         ("shall be liable", "shall not be liable")):
                if a in sents[i]:
                    new_sents = with_sentence(i, sents[i].replace(a, b, 1))
                    break
            if new_sents:
                break
    elif kind == "wrong_code":
        eq = normalizer.equivalents(ref) if normalizer is not None else []
        if eq:
            old_code, _, old_num = eq[0].partition(":")
            new_sents = [s.replace(f"BNS {num.upper()}", f"{old_code.upper()} {old_num.upper()}")
                          .replace(f"BNS Section {num}", f"{old_code.upper()} Section {old_num}") for s in sents]
            if new_sents == sents:
                new_sents = None
    elif kind == "section":
        bad = str(int(re.match(r"\d+", num).group(0)) + 217) if re.match(r"\d+", num) else None
        if bad:
            new_sents = [s.replace(f"BNS {num.upper()}", f"BNS {bad}").replace(f"BNS Section {num}", f"BNS Section {bad}")
                         for s in sents]
            if new_sents == sents:
                new_sents = None
    elif kind == "unsupported":
        new_sents = sents[: idx[-1] + 1] + [f"{other_sentence} [{law_n}]"] + sents[idx[-1] + 1:]
    if new_sents is None:
        return None
    out = parse_answer(" ".join(new_sents), ans.generator)
    out.chunks = ans.chunks
    return out


def _flags_v1(ans: Answer, rag, incident_date) -> list[str]:
    from kanoon_bridge.rag import citation_check, version_check

    a = copy.copy(ans)
    a.flags = []
    citation_check.check(a, a.chunks, rag.idf, threshold=rag.cfg.rag.support_threshold)
    version_check.check(a, incident_date, rag.resolver, rag.normalizer)
    return a.flags


def _flags_v2(ans: Answer, rag, incident_date) -> list[str]:
    from kanoon_bridge.rag import version_check
    from kanoon_bridge.rag.answer import _without_quotes
    from kanoon_bridge.rag.claim_check import ClaimVerifier

    a = copy.copy(ans)
    a.flags = []
    ClaimVerifier(rag.idf, rag.cfg.rag.support_threshold, rag.normalizer, rag.docs).check(a, a.chunks, "bns")
    own = version_check.check(_without_quotes(a), incident_date, rag.resolver, rag.normalizer)
    return a.flags + [f for f in own.flags if f.startswith("wrong code")]


KINDS = ("years", "fine", "negation", "wrong_code", "section", "unsupported")


def error_injection(rag, answers: list[tuple[dict, Answer, bool]], seed: int = 0) -> list[dict]:
    from datetime import date

    from kanoon_bridge.drift.provisions import load_bns

    rng = random.Random(seed)
    other = [p for p in load_bns().values() if extract(p.text).max_years]
    d = date(2025, 1, 1)
    hit = {c: defaultdict(list) for c in ("v1", "v2")}
    clean = {c: [] for c in ("v1", "v2")}
    for row, ans, ok in answers:
        if not ok:
            continue
        base = {"v1": set(_flags_v1(ans, rag, d)), "v2": set(_flags_v2(ans, rag, d))}
        for c in ("v1", "v2"):
            clean[c].append(bool(base[c]))
        o = rng.choice([p for p in other if f"bns:{p.section}" != row["section"]])
        other_sentence = f"It is also punishable with {extract(o.text).describe()}."
        for kind in KINDS:
            bad = _perturb(ans, kind, row, rag.normalizer, rng, other_sentence)
            if bad is None:
                continue
            for c, fn in (("v1", _flags_v1), ("v2", _flags_v2)):
                hit[c][kind].append(bool(set(fn(bad, rag, d)) - base[c]))
    rows = []
    for kind in KINDS:
        n = len(hit["v1"][kind])
        if n:
            rows.append({"error": kind, "n": n, "v1_detected": round(sum(hit["v1"][kind]) / n, 4),
                         "v2_detected": round(sum(hit["v2"][kind]) / n, 4)})
    n = len(clean["v1"])
    rows.append({"error": "none (false alarm on clean answers)", "n": n,
                 "v1_detected": round(sum(clean["v1"]) / n, 4) if n else "",
                 "v2_detected": round(sum(clean["v2"]) / n, 4) if n else ""})
    return rows


__all__ = ["punishment_questions", "punishment_qa", "error_injection", "KINDS", "FIELDS"]
