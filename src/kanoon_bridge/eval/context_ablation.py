"""E25: context ablation of E20 - is a misgrounded answer copied from the context or recalled?  [owner: Gaurav — working]

E20 counts how often an answer's first section belongs to the code NOT in force on the incident
date. It does not say why. Following the add/remove-document diagnosis of RAGViz (Wang et al.,
EMNLP 2024 demo), each RAG answer is asked again twice, with the same retrieval:

    control   the same context, unchanged                 -> how stable the LLM is at temperature 0
    ablated   every passage of the code not in force removed: its statutes, and the judgments that
              cite it; the next-ranked passages move up, so the context keeps its size

and the first section of each answer is scored as in E20 (eval/misgrounding.score_answer).
A question misgrounded in the control run is then

    context-induced   the ablated answer is no longer misgrounded: the wrong section came from the
                      retrieved text (fix: retrieval / filtering, layers 1-2)
    parametric        still misgrounded with no wrong-code passage in the context: the model
                      recalled the old number (fix: checking the answer, layer 3)

Without any LLM call, `diagnose` also reports for every saved E20 answer whether the context held
wrong-code passages at all and whether the misgrounded section was literally in it.

The context is rebuilt from the corpus, so it is only the E20 context if the corpus is the one E20
ran on; `header_match` (the share of "[n] IPC Section 506"-style mentions in the saved answers
whose chunk n is that statute) measures this.
"""

from __future__ import annotations

import dataclasses
import re
from collections import defaultdict

from kanoon_bridge.eval import misgrounding as M
from kanoon_bridge.schema import Query

CONDITIONS = ("naive RAG", "RAG v1", "RAG v2")
_HEADER = re.compile(r"\[(\d+)\]\s*(IPC|BNS|BNSS|BSA|CrPC|IEA)\s+Section\s+(\d+[A-Za-z]*)", re.I)


def wrong_code(row: dict) -> str:
    """The code NOT in force on the row's incident date, in the row's code family."""
    exp_code = row["expected"].lower().split(":")[0] if row["set"].startswith("e2") else row["expected_code"]
    family_new = M.OLD_NEW.get(exp_code, exp_code)
    return M.NEW_OLD[family_new] if row.get("incident_date", "") >= "2024-07-01" else family_new


def gold_rows() -> dict[str, dict]:
    from kanoon_bridge.eval.open_eval import load_rows

    out = {}
    for name in ("e7_temporal", "e9_procedure_temporal", "e2_collision"):
        for r in load_rows(name):
            out[r["id"]] = {**r, "set": name, "question": f"Which section applies to: {r['text']}?"}
    return out


def _query(row: dict) -> Query:
    return Query(row["question"], incident_date=row.get("incident_date"), state=row.get("state"))


def retrieve(engine, agent, row: dict, cond: str):
    from kanoon_bridge.search import SearchOptions

    if cond == "RAG v2":
        return agent.run(_query(row), SearchOptions.interactive())
    return engine.search(_query(row), SearchOptions.interactive())


def _ref_code(engine, doc_id: str) -> str:
    from kanoon_bridge.eval.open_eval import statute_ref

    return statute_ref(engine, doc_id).split(":", 1)[0]


def ablate(engine, result, bad: str, rag=None, rounds: int = 20):
    """`result` without the statutes of code `bad` and the judgments that cite it. Judgments' cited
    sections (facets.metas) are incomplete, so with `rag` the context is also built and every hit
    whose passage still names a section of `bad` is dropped, until none is left."""
    metas = engine.facets.metas

    def cites_bad(doc_id):
        m = metas.get(doc_id)
        return any(r.split(":", 1)[0] == bad for r in (getattr(m, "statutes_cited", None) or []))

    out = dataclasses.replace(result,
                              statutes=[h for h in result.statutes if _ref_code(engine, h.doc_id) != bad],
                              precedents=[h for h in result.precedents if not cites_bad(h.doc_id)])
    for _ in range(rounds if rag is not None else 0):
        drop = {c.doc_id for c in chunks_of(rag, out) if any(r.split(":")[0] == bad for r in _chunk_refs(engine, c))}
        if not drop:
            break
        out = dataclasses.replace(out, statutes=[h for h in out.statutes if h.doc_id not in drop],
                                  precedents=[h for h in out.precedents if h.doc_id not in drop])
    return out


def chunks_of(rag, result):
    from kanoon_bridge.rag import chunker

    rcfg = rag.cfg.rag
    q = getattr(result, "analyzed", None) or result.query
    qterms = list(q.weighted_terms()) if hasattr(q, "weighted_terms") else None
    return chunker.make_chunks(result, rag.docs, max_chunks=rcfg.max_chunks, max_chars=rcfg.max_chunk_chars,
                               query_terms=qterms)


def _chunk_refs(engine, chunk) -> set[str]:
    """Sections a chunk carries: the statute itself, and every section its text names."""
    from kanoon_bridge.eval.open_eval import statute_ref

    refs = {f"{m.code.value}:{m.section.split('(')[0].lower()}" for m in M._sections(chunk.text, strip_quotes=False)}
    if chunk.zone == "statute":
        refs.add(statute_ref(engine, chunk.doc_id).split("(")[0].lower())
    return refs


def _right_number(first, row: dict, normalizer) -> bool:
    """The first section's NUMBER is the right provision of the code in force (only its code is wrong).
    Not for E2: there the question gives the number, and the code is what is tested."""
    if not first or row["set"].startswith("e2"):
        return False
    num = first.split(":", 1)[1]
    offences = set(normalizer.offences_for(f"{row['expected_code']}:{num}")) if normalizer is not None else set()
    return row.get("offence") in offences


def diagnose(engine, chunks, answer: str, row: dict, bad: str, normalizer=None) -> dict:
    """No LLM: does the context hold wrong-code text, and is the answer's first section in it?
    `mislabel`: the first section has the right number of the code in force under the other code's
    name ("Section 118 of the Indian Penal Code (BNS)")."""
    refs = [_chunk_refs(engine, c) for c in chunks]
    wrong = sum(any(r.split(":")[0] == bad for r in rs) for rs in refs)
    secs = M._sections(answer)
    first = f"{secs[0].code.value}:{secs[0].section.split('(')[0].lower()}" if secs else ""
    heads = _HEADER.findall(answer or "")
    match = [int(n) <= len(chunks) and _chunk_refs(engine, chunks[int(n) - 1]) & {f"{c.lower()}:{s.lower()}"}
             for n, c, s in heads]
    return {"n_chunks": len(chunks), "wrong_code_chunks": wrong,
            "mislabel": float(bool(secs) and secs[0].code.value == bad and _right_number(first, row, normalizer)),
            "first_section_in_context": float(bool(first) and any(first in rs for rs in refs)),
            "headers": len(heads), "headers_matching": sum(bool(m) for m in match)}


def _answer(rag, result, cond: str, gen: str) -> str:
    try:
        if cond == "naive RAG":
            with M._prompt(M.NAIVE_PROMPT):
                return rag.answer(result, generator=gen, version=1).text
        return rag.answer(result, generator=gen, version=2 if cond == "RAG v2" else 1).text
    except Exception as err:                               # a server error is recorded, not fatal
        return f"[error: {err}]"


def classify(control_mis: float, ablated_mis: float, ablated_wrong_chunks: int, mislabel: float = 0.0) -> str:
    if not control_mis:
        return "not misgrounded"
    if mislabel:
        return "mislabel"
    if not ablated_mis:
        return "context-induced"
    return "parametric" if ablated_wrong_chunks == 0 else "unresolved (wrong-code text left)"


def run(engine, rag, agent, saved: list[dict], gen: str | None, conditions=CONDITIONS,
        only_misgrounded: bool = False, log=print) -> tuple[list[dict], list[dict]]:
    """`saved`: E20 answers (rescored). `gen` None = diagnostics only, no LLM calls.
    Returns (summary table, one row per answer)."""
    gold = gold_rows()
    norm = engine.analyzer.text_res.normalizer
    todo = [s for s in saved if s["condition"] in conditions and s["id"] in gold
            and (not only_misgrounded or s["misgrounded@1"])]
    rows = []
    for i, s in enumerate(todo):
        row, cond = gold[s["id"]], s["condition"]
        bad = wrong_code(row)
        result = retrieve(engine, agent, row, cond)
        abl = ablate(engine, result, bad, rag)
        ctx, ctx_abl = chunks_of(rag, result), chunks_of(rag, abl)
        out = {"id": s["id"], "set": row["set"], "condition": cond, "incident_date": row.get("incident_date"),
               "wrong_code": bad, "e20_misgrounded@1": s["misgrounded@1"], "e20_right@1": s["right@1"],
               **diagnose(engine, ctx, s["answer"], row, bad, norm),
               "ablated_wrong_code_chunks": diagnose(engine, ctx_abl, "", row, bad)["wrong_code_chunks"],
               "ablated_n_chunks": len(ctx_abl)}
        if gen:
            for name, res in (("control", result), ("ablated", abl)):
                text = _answer(rag, res, cond, gen)
                sc = M.score_answer(text, row, norm)
                d = diagnose(engine, ctx if name == "control" else ctx_abl, text, row, bad, norm)
                out.update({f"{name}_mislabel": d["mislabel"], f"{name}_answer": text, f"{name}_misgrounded@1": sc["misgrounded@1"],
                            f"{name}_right@1": sc["right@1"], f"{name}_no_section": sc["no_section"]})
            out["cause"] = classify(out["control_misgrounded@1"], out["ablated_misgrounded@1"],
                                    out["ablated_wrong_code_chunks"], out["control_mislabel"])
        rows.append(out)
        if log and (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(todo)} answers")
    return summarize(rows), rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["condition"], "all")].append(r)
        groups[(r["condition"], r["set"])].append(r)
    table = []
    for (cond, name), rs in groups.items():
        n = len(rs)
        mis = [r for r in rs if r["e20_misgrounded@1"]]
        t = {"condition": cond, "set": name, "n": n,
             "e20_misgrounded@1": round(len(mis) / n, 4),
             "context_has_wrong_code": round(sum(r["wrong_code_chunks"] > 0 for r in rs) / n, 4),
             "ablated_has_wrong_code": round(sum(r["ablated_wrong_code_chunks"] > 0 for r in rs) / n, 4),
             "misgrounded_n": len(mis),
             "misgrounded_section_in_context": round(sum(r["first_section_in_context"] for r in mis) / len(mis), 4) if mis else "",
             "misgrounded_mislabel": round(sum(r["mislabel"] for r in mis) / len(mis), 4) if mis else "",
             "misgrounded_copied": round(sum(r["first_section_in_context"] and not r["mislabel"] for r in mis) / len(mis), 4) if mis else "",
             "misgrounded_recalled": round(sum(not r["first_section_in_context"] and not r["mislabel"] for r in mis) / len(mis), 4) if mis else "",
             "chunks": round(sum(r["n_chunks"] for r in rs) / n, 2),
             "ablated_chunks": round(sum(r.get("ablated_n_chunks", r["n_chunks"]) for r in rs) / n, 2),
             "misgrounded_no_wrong_code_in_context": round(sum(r["wrong_code_chunks"] == 0 for r in mis) / len(mis), 4) if mis else "",
             "header_match": round(sum(r["headers_matching"] for r in rs) / max(1, sum(r["headers"] for r in rs)), 4)}
        if "cause" in rs[0]:
            for k in ("control_misgrounded@1", "ablated_misgrounded@1", "control_right@1", "ablated_right@1",
                      "ablated_no_section"):
                t[k] = round(sum(r[k] for r in rs) / n, 4)
            ctl = [r for r in rs if r["control_misgrounded@1"]]
            t["control_misgrounded_n"] = len(ctl)
            t["unstable_vs_e20"] = round(sum(r["control_misgrounded@1"] != r["e20_misgrounded@1"] for r in rs) / n, 4)
            for cause in ("mislabel", "context-induced", "parametric", "unresolved (wrong-code text left)"):
                t[cause] = round(sum(r["cause"] == cause for r in ctl) / len(ctl), 4) if ctl else ""
        table.append(t)
    order = {c: i for i, c in enumerate(CONDITIONS)}
    table.sort(key=lambda t: (order.get(t["condition"], 9), t["set"] != "all", t["set"]))
    return table


def significance(rows: list[dict]) -> list[dict]:
    """Exact McNemar, control vs ablated, on misgrounded@1 and right@1, per condition (Holm)."""
    from kanoon_bridge.eval.significance import holm, mcnemar_exact

    out = []
    for cond in CONDITIONS:
        rs = [r for r in rows if r["condition"] == cond and "cause" in r]
        if not rs:
            continue
        for metric in ("misgrounded@1", "right@1"):
            b = sum(1 for r in rs if r[f"control_{metric}"] and not r[f"ablated_{metric}"])
            c = sum(1 for r in rs if not r[f"control_{metric}"] and r[f"ablated_{metric}"])
            out.append({"condition": cond, "metric": metric, "n": len(rs), "control_only": b, "ablated_only": c,
                        "p": mcnemar_exact(b, c)})
    for r, p in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = p
    return out


__all__ = ["run", "summarize", "significance", "diagnose", "ablate", "wrong_code", "classify", "CONDITIONS"]
