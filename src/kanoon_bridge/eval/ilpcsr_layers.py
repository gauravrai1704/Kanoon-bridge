"""E17-E18: agent v2 and RAG v2 on the IL-PCSR corpus (make build).  [owner: Gaurav — working]

E14-E16 (eval/layers_eval.py) ran on the open corpus, whose eight precedents are synthetic.
Here the same layers run on the full corpus: IL-PCSR's 3,183 precedents and 936 statutes, plus
the BNS, BNSS and BSA.

E17  retrieval, every judged set, same SearchOptions.full() underneath every system:
         core          one search (the v1 engine)
         agent v1      rule planner + RRF + reflection (agent.version 1)
         agent v2      routing, family search, citation votes, verifier, issue rounds
         v2 - verification, v2 - family search, v2 - issue rounds    (ablations)
     metrics: MAP, MRR, nDCG@10, F1@k (k chosen on val where the set has a val split), plus the
     set's own extras (E2 wrong-offence, E6 binding share, E7 code@1); searches per question.
     Significance: paired randomization test on per-query AP (and RR), agent v2 vs core and
     vs agent v1, Holm-corrected over all comparisons (eval/significance.py).

E18  answers.
     a. data/queries/rag_questions.jsonl (20 hand-written questions, some unanswerable): core +
        RAG v1 vs agent v2 + RAG v2. Per answer: abstained, sentences, the v1 citation-support
        rate, the share of claims the v2 verifier accepts (applied to both systems' final
        answers; v2's are verified by construction, so this column measures v1), wrong-code
        flags, corrections. Answers dumped for reading.
     b. E14 punishment QA and E15 error injection (eval/layers_eval.py) on this corpus: the
        retrieved precedents are now real judgments.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict

from kanoon_bridge.config import Config, load_config, project_path

SETS = ("e1_ilpcsr", "e1q_ilpcsr_short", "e1s_ilpcsr_statutes", "e2_collision", "e4_multilingual",
        "e6_jurisdiction", "e7_temporal", "e9_procedure_temporal")
ABLATIONS = {"v2 - verification": {"verify_family": 0, "verify_force": 0, "verify_heading": 0, "verify_general": 0},
             "v2 - family search": {"family_weight": 0},
             "v2 - issue rounds": {"issue_rounds": 0}}
KEEP = ("MAP", "MRR", "nDCG@10", "P@1", "F1@k_val", "k_val", "wrong_hit@10", "binding_share@5", "code@1",
        "n_queries")


class _Counting:
    """Wraps an agent to count the searches it makes per question."""

    def __init__(self, agent):
        self.agent, self.searches = agent, []

    def run(self, query, options=None):
        res = self.agent.run(query, options)
        self.searches.append(getattr(res, "n_searches", 1))
        return res


def systems(engine, ablations: bool = True) -> dict:
    from kanoon_bridge.agent.research import ResearchAgent, ResearchAgentV2

    out = {"core": None, "agent v1": ResearchAgent.load(engine), "agent v2": ResearchAgentV2.load(engine)}
    if ablations:
        for name, over in ABLATIONS.items():
            cfg = copy.deepcopy(engine.cfg)
            for k, v in over.items():
                cfg.agent[k] = v
            out[name] = ResearchAgentV2.load(engine, cfg=cfg)
    return out


def retrieval(engine, sets=SETS, limit: int | None = None, ev: Config | None = None, ablations: bool = True,
              ablation_sets=None, only_systems=None) -> tuple[list[dict], list[dict]]:
    """E17: score rows and significance rows. `only_systems`: run just these; the others needed
    for the comparisons (core, agent v1) are read from their run files of an earlier run."""
    from kanoon_bridge.eval import significance as sig
    from kanoon_bridge.eval.run_eval import evaluate_set, load_test_set, read_run
    from kanoon_bridge.search import SearchOptions

    ev = ev or load_config("eval.yaml")
    out_dir = project_path(ev.outputs.runs) / "layers"
    systems_all = systems(engine, ablations)
    rows, comps = [], []
    for name in sets:
        try:
            ts = load_test_set(name, ev=ev, limit=limit)
        except FileNotFoundError as err:
            print(f"  {name}: skipped ({err})")
            continue
        if not ts.judged:
            print(f"  {name}: no judgments - skipped")
            continue
        runs = {}
        for system, agent in systems_all.items():
            if system.startswith("v2 -") and ablation_sets is not None and name not in ablation_sets:
                continue
            if only_systems and system not in only_systems:
                path = out_dir / f"{name}.{system.replace(' ', '_')}.run"
                if system in ("core", "agent v1") and path.exists():
                    runs[system] = read_run(path)
                continue
            counted = _Counting(agent) if agent is not None else None
            scores = evaluate_set(engine, ts, SearchOptions.full(), system.replace(" ", "_"), ev, out_dir, agent=counted)
            row = {"set": name, "system": system, **{k: round(v, 4) for k, v in scores.items() if k in KEEP}}
            row["searches_per_q"] = round(sum(counted.searches) / len(counted.searches), 2) if counted and counted.searches else 1.0
            rows.append(row)
            runs[system] = read_run(out_dir / f"{name}.{system.replace(' ', '_')}.run")
            print(f"  {name:22s} {system:20s} MAP {row.get('MAP', 0):.4f}  MRR {row.get('MRR', 0):.4f}  "
                  f"searches {row['searches_per_q']}")
        qrels = ts.qrels
        if name.startswith("e6"):
            from kanoon_bridge.eval.qrels import jurisdiction_grades

            qrels = jurisdiction_grades(ts.qrels, engine.facets.metas, {q.query_id: q.state for q in ts.queries})
        per = {s: sig.per_query(r, qrels) for s, r in runs.items()}
        for a, b in (("core", "agent v2"), ("agent v1", "agent v2")):
            for m in ("AP", "RR"):
                comps.append({"set": name, "a": a, "b": b, "metric": m, **sig.compare(per[a][m], per[b][m])})
    adj = sig.holm([c["p"] for c in comps])
    for c, p in zip(comps, adj):
        c["p_holm"] = round(p, 5)
    return rows, comps


def merge(tables_dir) -> tuple[list[dict], list[dict]]:
    """Concatenate the tagged tables of parallel runs (ilpcsr_layers_retrieval_<tag>.csv) in SETS
    order and redo the Holm correction over every comparison."""
    import csv

    from kanoon_bridge.eval.significance import holm

    def read(prefix):
        out = []
        for path in sorted(tables_dir.glob(f"{prefix}_*.csv")):
            with open(path, encoding="utf-8") as f:
                out += list(csv.DictReader(f))
        return sorted(out, key=lambda r: SETS.index(r["set"]) if r["set"] in SETS else len(SETS))

    # a later file (sorted by tag) replaces an earlier one's rows for the same set and system
    rows = list({(r["set"], r["system"]): r for r in read("ilpcsr_layers_retrieval")}.values())
    comps = list({(r["set"], r["a"], r["b"], r["metric"]): r for r in read("ilpcsr_layers_significance")}.values())
    rank = {s: i for i, s in enumerate(SETS)}
    rows.sort(key=lambda r: rank.get(r["set"], len(SETS)))
    comps.sort(key=lambda r: rank.get(r["set"], len(SETS)))
    for c, p in zip(comps, holm([float(c["p"]) for c in comps])):
        c["p_holm"] = round(p, 5)
    return rows, comps


# --------------------------------------------------------------------------- E18a


def rag_questions(engine, ev: Config | None = None, limit: int | None = None) -> tuple[list[dict], list[dict]]:
    from kanoon_bridge.agent.research import ResearchAgentV2
    from kanoon_bridge.eval.agent_eval import load_questions
    from kanoon_bridge.eval.layers_eval import _flags_v2
    from kanoon_bridge.rag.answer import RagPipeline
    from kanoon_bridge.rag.claim_check import ClaimVerifier
    from kanoon_bridge.schema import Query
    from kanoon_bridge.search import SearchOptions

    ev = ev or load_config("eval.yaml")
    questions = load_questions(ev.rag.questions)[:limit]
    rag = RagPipeline.load(engine)
    agent = ResearchAgentV2.load(engine, docs=rag.docs)
    verifier = ClaimVerifier(rag.idf, rag.cfg.rag.support_threshold, rag.normalizer, rag.docs)
    systems_ = {"core + RAG v1": (lambda q: engine.search(q, SearchOptions(top_k=10)), 1),
                "agent v2 + RAG v2": (lambda q: agent.run(q, SearchOptions(top_k=10)), 2)}
    rows, dump = [], []
    for row in questions:
        q = Query.from_dict(row)
        for system, (search, version) in systems_.items():
            res = search(copy.deepcopy(q))
            ans = rag.answer(res, generator="extractive", version=version)
            a = copy.copy(ans)
            a.flags = []
            cif = _code_in_force(q)
            verifier.check(a, a.chunks, cif)
            claims = getattr(a, "claims", []) or []
            from datetime import date

            d = date.fromisoformat(row["incident_date"]) if row.get("incident_date") else None
            wrong = [f for f in _flags_v2(ans, rag, d) if f.startswith("wrong code")] if d and ans.sentences else []
            rows.append({"id": row.get("id"), "system": system, "answerable": row.get("answerable", True),
                         "abstained": ans.abstained, "sentences": len(ans.sentences),
                         "v1_support_rate": round(ans.supported_rate, 3) if ans.sentences and version == 1 else "",
                         "claims_verified": round(sum(c.verdict == "supported" for c in claims) / len(claims), 3)
                         if claims else "",
                         "wrong_code_flags": len(wrong), "corrections": len(getattr(ans, "corrections", []) or [])})
            dump.append({"id": row.get("id"), "system": system, "question": q.text,
                         "incident_date": row.get("incident_date"), "answer": ans.text,
                         "notes": getattr(ans, "notes", []), "corrections": getattr(ans, "corrections", []),
                         "flags": ans.flags, "sources": [c.header() for c in ans.chunks],
                         "memo": res.memo.render() if getattr(res, "memo", None) else ""})
    return rows + summarise(rows), dump


def _code_in_force(q) -> str | None:
    from datetime import date

    if not q.incident_date:
        return None
    d = q.incident_date if isinstance(q.incident_date, date) else date.fromisoformat(str(q.incident_date))
    return "bns" if d >= date(2024, 7, 1) else "ipc"


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    by = defaultdict(list)
    for r in rows:
        by[r["system"]].append(r)
    for system, rs in by.items():
        answered = [r for r in rs if not r["abstained"]]
        s = {"id": "MEAN", "system": system, "abstained": sum(r["abstained"] for r in rs)}
        for key in ("sentences", "v1_support_rate", "claims_verified", "wrong_code_flags", "corrections"):
            vals = [r[key] for r in answered if r[key] != ""]
            if vals:
                s[key] = round(sum(vals) / len(vals), 3)
        unans = [r for r in rs if r["answerable"] is False]
        abst = [r for r in rs if r["abstained"]]
        if abst:
            s["abstain_precision"] = round(sum(r["answerable"] is False for r in abst) / len(abst), 3)
        if unans:
            s["abstain_recall"] = round(sum(r["abstained"] for r in unans) / len(unans), 3)
        out.append(s)
    return out


def write_jsonl(rows: list[dict], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


__all__ = ["retrieval", "merge", "rag_questions", "systems", "SETS", "ABLATIONS"]
