"""E20: temporal misgrounding of LLM answers at the 1 July 2024 code replacement.  [owner: Gaurav — working]

"Temporal misgrounding" (FiscalQA Pro, arXiv 2608.09393): citing the version of the law in force
today when an earlier one applies, or the reverse. India's 2024 replacement of three codes on one
day, with renumbering and colliding section numbers, is the extreme case: "302" is murder in the
IPC and something else in the BNS.

Questions (deterministic sample, balanced across the two sides of 1 July 2024):
    E7   penal provisions, incident before / after 1 July 2024   -> IPC vs BNS
    E9   procedure and evidence                                  -> CrPC vs BNSS, IEA vs BSA
    E2   bare colliding numbers ("punishment under section 84")   -> the code in force decides
Each is asked as "Which section applies to: <text>?" with the incident date.

Conditions (same LLM, temperature 0):
    closed-book     no retrieval; the prompt gives the date and says the codes changed
    naive RAG       core retrieval, a generic prompt with no rule about the date
    RAG v1          core retrieval, our date-aware prompt (rag/generate.PROMPT) + v1 checks
    RAG v2          agent v2 retrieval, the same prompt, claim verification, one regeneration
                    with the problems listed, deterministic correction (rag/answer._answer_v2)
    structured      agent v2 + the computed answer, no LLM (reference)

Scored from the answer's own words (quotations removed), on the FIRST section it names:
    right@1         the right provision of the code in force (E7/E9: code + provision id;
                    E2: the expected section)
    misgrounded@1   a section of the code NOT in force on the incident date (E2: a colliding
                    wrong section)
    no_section      the answer names no section
    any_wrong_code  any section named, anywhere in the answer, is of the code not in force

The LLM is any of rag/generate's: local (an OpenAI-compatible server: vLLM / Ollama / llama.cpp
on your own GPU), openai, claude.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
from collections import defaultdict

from kanoon_bridge.schema import Query

NAIVE_PROMPT = """Answer the question using the sources below. Name the section that applies.

Question: {question}
Incident date: {incident_date}   User's state: {state}

Sources:
{sources}
"""

OLD_NEW = {"ipc": "bns", "crpc": "bnss", "iea": "bsa"}
NEW_OLD = {v: k for k, v in OLD_NEW.items()}
CONDITIONS = ("closed-book", "naive RAG", "RAG v1", "RAG v2", "structured")
_QUOTED = re.compile(r"[\"“][^\"”]{8,}[\"”]")


def questions(per_set: int = 40) -> list[dict]:
    from kanoon_bridge.eval.open_eval import load_rows

    out = []
    for name in ("e7_temporal", "e9_procedure_temporal", "e2_collision"):
        rows = sorted(load_rows(name), key=lambda r: hashlib.md5(r["id"].encode()).hexdigest())
        early = [r for r in rows if r.get("incident_date", "") < "2024-07-01"]
        late = [r for r in rows if r.get("incident_date", "") >= "2024-07-01"]
        pick = [r for pair in zip(early, late) for r in pair][:per_set]
        for r in pick:
            out.append({**r, "set": name, "question": f"Which section applies to: {r['text']}?"})
    return out


def _sections(text: str):
    from kanoon_bridge.text.tokenize import extract_sections

    clean = _QUOTED.sub(" ", text or "")
    return [m for m in sorted(extract_sections(clean), key=lambda m: m.start) if m.code.value in
            ("ipc", "bns", "crpc", "bnss", "iea", "bsa")]


def score_answer(text: str, row: dict, normalizer) -> dict[str, float]:
    secs = _sections(text)
    date = row.get("incident_date", "")
    new = date >= "2024-07-01"
    if row["set"].startswith("e2"):
        exp = row["expected"].lower()
        exp_code = exp.split(":")[0]
    else:
        exp_code = row["expected_code"]
    family_new = OLD_NEW.get(exp_code, exp_code)
    not_in_force = NEW_OLD[family_new] if new else family_new
    out = {"no_section": float(not secs), "right@1": 0.0, "misgrounded@1": 0.0,
           "any_wrong_code": float(any(m.code.value == not_in_force for m in secs))}
    if not secs:
        return out
    first = secs[0]
    ref = f"{first.code.value}:{first.section.split('(')[0].lower()}"
    if row["set"].startswith("e2"):
        out["right@1"] = float(ref == row["expected"].lower())
        out["misgrounded@1"] = float(ref in [w.lower() for w in row.get("wrong_refs", [])]
                                     or first.code.value == not_in_force)
    else:
        offences = set(normalizer.offences_for(ref)) if normalizer is not None else set()
        out["right@1"] = float(first.code.value == exp_code and row["offence"] in offences)
        out["misgrounded@1"] = float(first.code.value == not_in_force)
    return out


@contextlib.contextmanager
def _prompt(template: str):
    from kanoon_bridge.rag import generate

    old = generate.PROMPT
    generate.PROMPT = template
    try:
        yield
    finally:
        generate.PROMPT = old


def run(engine, rag, agent, rows: list[dict], gen: str, model: str | None = None,
        conditions=CONDITIONS, log=print) -> tuple[list[dict], list[dict]]:
    """Per (set, condition) mean scores, and every answer for reading."""
    from kanoon_bridge.rag import generate
    from kanoon_bridge.search import SearchOptions

    norm = engine.analyzer.text_res.normalizer
    acc = defaultdict(lambda: defaultdict(list))
    dump = []
    for i, row in enumerate(rows):
        def q():
            return Query(row["question"], incident_date=row.get("incident_date"), state=row.get("state"))

        for cond in conditions:
            try:
                if cond == "closed-book":
                    text = generate.call_llm(gen, generate.CLOSED_BOOK_PROMPT.format(
                        question=row["question"], incident_date=row.get("incident_date"), state="unknown"), model)
                    corrections = []
                elif cond == "naive RAG":
                    with _prompt(NAIVE_PROMPT):
                        ans = rag.answer(engine.search(q(), SearchOptions.interactive()), generator=gen, version=1)
                    text, corrections = ans.text, []
                elif cond == "RAG v1":
                    ans = rag.answer(engine.search(q(), SearchOptions.interactive()), generator=gen, version=1)
                    text, corrections = ans.text, []
                elif cond == "RAG v2":
                    ans = rag.answer(agent.run(q(), SearchOptions.interactive()), generator=gen, version=2)
                    text, corrections = ans.text, list(getattr(ans, "corrections", []) or [])
                else:
                    ans = rag.answer(agent.run(q(), SearchOptions.interactive()), generator="extractive", version=2)
                    text, corrections = ans.text, list(getattr(ans, "corrections", []) or [])
            except Exception as err:                       # a server error is recorded, not fatal
                text, corrections = f"[error: {err}]", []
            s = score_answer(text, row, norm)
            for k, v in s.items():
                acc[(row["set"], cond)][k].append(v)
                acc[("all", cond)][k].append(v)
            dump.append({"id": row["id"], "set": row["set"], "condition": cond, "incident_date": row.get("incident_date"),
                         "question": row["question"], "answer": text, "corrections": corrections, **s})
        if log and (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(rows)} questions")
    table = []
    for (name, cond), m in acc.items():
        table.append({"set": name, "condition": cond, "n": len(m["right@1"]),
                      **{k: round(sum(v) / len(v), 4) for k, v in m.items()}})
    order = {c: i for i, c in enumerate(CONDITIONS)}
    table.sort(key=lambda r: (r["set"] != "all", r["set"], order.get(r["condition"], 9)))
    return table, dump


__all__ = ["questions", "score_answer", "run", "CONDITIONS", "NAIVE_PROMPT"]
