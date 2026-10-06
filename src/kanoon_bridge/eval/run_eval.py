"""Load test sets, run systems on them, score them.  [owner: Gaurav — working]

Every run goes through search.SearchEngine (rule 4), so results equal what the demo shows.

Test sets (configs/eval.yaml):
    e1_ilpcsr            IL-PCSR test judgments as queries -> cited precedents (k chosen on val)
    e1s_ilpcsr_statutes  same queries -> cited statutes
    e2_collision         hand-built; extra metric wrong_hit@10 from each row's "wrong_refs"
    e3_cross_version     generated: "cases under section 103 BNS"; gold = precedents that applied IPC 302
    e4_multilingual      hand-built; P@5 also reported per language
    e6_jurisdiction      hand-built; grades: binding = 2, persuasive = 1; binding_share@5
    e7_temporal          hand-built; code_accuracy@1 (top statute in the code in force)

TREC run format (one line per result):  query_id  Q0  doc_id  rank  score  system

    python -m kanoon_bridge.eval.run_eval --set e2_collision --system full
    python -m kanoon_bridge.eval.run_eval --set e1_ilpcsr --system baseline --limit 50
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from kanoon_bridge.config import Config, load_config, project_path
from kanoon_bridge.schema import Document, Query

Run = dict[str, list[tuple[str, float]]]
Qrels = dict[str, dict[str, int]]


# --------------------------------------------------------------------------- run files


def write_run(run: Run, path: str | Path, system: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q, ranked in run.items():
            for i, (d, s) in enumerate(ranked, start=1):
                f.write(f"{q}\tQ0\t{d}\t{i}\t{s:.6f}\t{system}\n")


def read_run(path: str | Path) -> dict[str, list[str]]:
    out: dict[str, list[tuple[int, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            q, _, d, rank, _score, _sys = line.rstrip("\n").split("\t")
            out.setdefault(q, []).append((int(rank), d))
    return {q: [d for _, d in sorted(v)] for q, v in out.items()}


def ranked_ids(run: Run) -> dict[str, list[str]]:
    return {q: [d for d, _ in r] for q, r in run.items()}


# --------------------------------------------------------------------------- test sets


@dataclass
class TestSet:
    name: str
    queries: list[Query]
    qrels: Qrels
    target: str                                   # "precedent" | "statute"
    rows: dict[str, dict] = field(default_factory=dict)       # raw jsonl row per query id (extra fields)
    val: "TestSet | None" = None                  # for choosing k (E1)
    max_query_terms: int | None = None

    @property
    def judged(self) -> int:
        return sum(1 for q in self.queries if any(g > 0 for g in self.qrels.get(q.query_id, {}).values()))


def query_from_judgment(doc: Document) -> Query:
    """An IL-PCSR query judgment as a search query: masked text, its court's state, its date."""
    states = [s for s in doc.states if s != "*"]
    return Query(text=doc.text, query_id=doc.doc_id, state=states[0] if len(states) == 1 else None,
                 incident_date=doc.decision_date)


def _ilpcsr_queries(split: str, cfg: Config) -> list[Query]:
    from kanoon_bridge.ingest import load_ilpcsr, metadata

    table = metadata.CourtTable.load(cfg)
    out = []
    for doc in load_ilpcsr.load_queries(split, cfg):
        metadata.enrich(doc, table)
        out.append(query_from_judgment(doc))
    return out


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_test_set(name: str, cfg: Config | None = None, ev: Config | None = None, limit: int | None = None) -> TestSet:
    """Queries, qrels and target for a set named in configs/eval.yaml."""
    from kanoon_bridge.eval.qrels import load_tsv
    from kanoon_bridge.ingest import load_ilpcsr

    cfg = cfg or load_config()
    ev = ev or load_config("eval.yaml")
    spec = ev.test_sets[name]
    target = spec.target

    if spec.queries.startswith("ilpcsr_"):
        split = spec.queries.removeprefix("ilpcsr_")
        queries = _ilpcsr_queries(split, cfg)[:limit]
        qrels = _restrict(load_ilpcsr.load_qrels(split, target, cfg), queries)
        val = None
        if spec.get("val") and ev.get("ilpcsr_protocol", True):
            vsplit = spec.val.removeprefix("ilpcsr_")
            vq = _ilpcsr_queries(vsplit, cfg)[:limit]
            val = TestSet(f"{name}:val", vq, _restrict(load_ilpcsr.load_qrels(vsplit, target, cfg), vq),
                          target, max_query_terms=ev.get("e1_max_query_terms"))
        return TestSet(name, queries, qrels, target, val=val, max_query_terms=ev.get("e1_max_query_terms"))

    rows = _read_rows(project_path(spec.queries))[:limit]
    queries = [Query.from_dict(r) for r in rows]
    for q, r in zip(queries, rows):
        q.query_id = q.query_id or r.get("id")
    by_id = {q.query_id: r for q, r in zip(queries, rows)}
    if spec.qrels.startswith("ilpcsr_"):                     # E3: gold comes from the source query
        source = load_ilpcsr.load_qrels(spec.qrels.removeprefix("ilpcsr_"), target, cfg)
        qrels = {q.query_id: source.get(by_id[q.query_id].get("source_query_id", ""), {}) for q in queries}
    else:
        path = project_path(spec.qrels)
        qrels = load_tsv(path) if path.exists() else {}
    return TestSet(name, queries, _restrict(qrels, queries), target, rows=by_id)


def _restrict(qrels: Qrels, queries: list[Query]) -> Qrels:
    """Only the loaded queries count (so --limit does not score missing queries as zeros)."""
    ids = {q.query_id for q in queries}
    return {q: r for q, r in qrels.items() if q in ids}


# --------------------------------------------------------------------------- running


def run_queries(engine, queries: list[Query], options, target: str = "precedent", k: int = 100,
                max_query_terms: int | None = None, agent=None) -> Run:
    """Search every query (or run the agent); keep the `target` list, `k` deep."""
    import copy

    opt = copy.deepcopy(options)
    opt.top_k = k
    if max_query_terms:
        opt.max_query_terms = max_query_terms
    run: Run = {}
    for q in queries:
        q2 = copy.deepcopy(q)
        res = agent.run(q2, opt) if agent is not None else engine.search(q2, opt)
        hits = res.precedents if target == "precedent" else res.statutes
        run[q.query_id or q.text] = [(h.doc_id, h.score) for h in hits]
    return run


# --------------------------------------------------------------------------- set-specific metrics


def _statute_ref(engine, doc_id: str) -> str:
    terms = getattr(getattr(engine, "bridge", None), "statute_terms", {}) or {}
    sec = next((t for t in terms.get(doc_id, []) if t.startswith("sec:")), None)
    return sec[4:] if sec else doc_id


def extra_metrics(ts: TestSet, run: Run, engine) -> dict[str, float]:
    """Metrics that need more than qrels (see module docstring)."""
    from kanoon_bridge.eval.metrics import precision_at_k
    from kanoon_bridge.rank.authority import binding_status

    out: dict[str, float] = {}
    ids = ranked_ids(run)
    if ts.name.startswith("e2"):
        rows = [(q, ts.rows[q.query_id].get("wrong_refs") or []) for q in ts.queries]
        rows = [(q, w) for q, w in rows if w]
        if rows:
            hits = sum(any(_statute_ref(engine, d) in w for d in ids.get(q.query_id, [])[:10]) for q, w in rows)
            out["wrong_hit@10"] = hits / len(rows)
    if ts.name.startswith("e4"):
        for lang in ("en", "hi", "hinglish"):
            qs = [q for q in ts.queries if (ts.rows[q.query_id].get("lang") or q.lang) == lang
                  and any(g > 0 for g in ts.qrels.get(q.query_id, {}).values())]
            if qs:
                out[f"P@5_{lang}"] = sum(precision_at_k(ids.get(q.query_id, []), ts.qrels[q.query_id], 5) for q in qs) / len(qs)
    if ts.name.startswith("e6"):
        metas = engine.facets.metas
        shares = []
        for q in ts.queries:
            top = ids.get(q.query_id, [])[:5]
            if top and q.state:
                shares.append(sum(binding_status(metas[d], q.state) == "binding" for d in top if d in metas) / len(top))
        if shares:
            out["binding_share@5"] = sum(shares) / len(shares)
    if ts.name.startswith("e7"):
        from kanoon_bridge.ingest.metadata import code_in_force

        checks = []
        for q in ts.queries:
            top = ids.get(q.query_id, [])[:1]
            if top and q.incident_date:
                checks.append(_statute_ref(engine, top[0]).startswith(code_in_force(q.incident_date).value + ":"))
        if checks:
            out["code_accuracy@1"] = sum(checks) / len(checks)
    return out


def evaluate_set(engine, ts: TestSet, options, system: str, ev: Config, out_dir: Path | None = None,
                 agent=None) -> dict[str, float]:
    """Run + score one system on one set. With a val split, F1@k uses k chosen on val."""
    from kanoon_bridge.eval.metrics import best_k, evaluate, macro_f1_at_k

    qrels = ts.qrels
    if ts.name.startswith("e6"):
        from kanoon_bridge.eval.qrels import jurisdiction_grades

        qrels = jurisdiction_grades(ts.qrels, engine.facets.metas, {q.query_id: q.state for q in ts.queries})
    depth = ev.get("depth", 100)
    run = run_queries(engine, ts.queries, options, ts.target, depth, ts.max_query_terms, agent)
    if out_dir is not None:
        write_run(run, out_dir / f"{ts.name}.{system}.run", system)
    scores = evaluate(ranked_ids(run), qrels, list(ev.k_values))
    if ts.val is not None and ts.val.queries:
        vrun = run_queries(engine, ts.val.queries, options, ts.target, depth, ts.max_query_terms, agent)
        k = best_k(ranked_ids(vrun), ts.val.qrels, range(1, 11))
        scores["k_val"] = float(k)
        scores["F1@k_val"] = macro_f1_at_k(ranked_ids(run), qrels, k)
    scores.update(extra_metrics(ts, run, engine))
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--system", default="full", help="'full', 'baseline', or an ablation step name")
    ap.add_argument("--limit", type=int, help="only the first N queries (quick runs)")
    args = ap.parse_args()

    from kanoon_bridge.search import SearchEngine

    ev = load_config("eval.yaml")
    engine = SearchEngine.load()
    ts = load_test_set(args.set, ev=ev, limit=args.limit)
    print(f"{ts.name}: {len(ts.queries)} queries, {ts.judged} judged")
    scores = evaluate_set(engine, ts, options_for(args.system, ev), args.system, ev, project_path(ev.outputs.runs))
    print(json.dumps({k: round(v, 4) for k, v in scores.items()}, indent=2))


def options_for(system: str, ev: Config | None = None):
    """SearchOptions for 'full', 'baseline' or a ladder step name."""
    from kanoon_bridge.search import SearchOptions

    if system == "baseline":
        return SearchOptions.baseline()
    if system == "full":
        return SearchOptions.full()
    ev = ev or load_config("eval.yaml")
    step = next(s for s in ev.ablation_ladder if s["name"] == system)
    return SearchOptions.from_dict(step)


if __name__ == "__main__":
    main()
