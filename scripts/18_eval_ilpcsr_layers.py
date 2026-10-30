"""Run E17-E18 (eval/ilpcsr_layers.py): agent v2 and RAG v2 on the IL-PCSR corpus.  [owner: Gaurav — working]

Needs the full build (make build: IL-PCSR in data/raw/ilpcsr, indexes, graph, LTR) and the
bns-study-platform checkout. No API key: answers use the offline generators.

    results/tables/ilpcsr_layers_retrieval.csv      E17 scores
    results/tables/ilpcsr_layers_significance.csv   E17 agent v2 vs core / agent v1 (Holm)
    results/tables/ilpcsr_layers_rag.csv            E18a
    results/tables/ilpcsr_layers_rag_answers.jsonl  E18a answers, for reading
    results/tables/ilpcsr_layers_punishment_qa.csv  E18b (E14 on this corpus)
    results/tables/ilpcsr_layers_error_injection.csv E18b (E15 on this corpus)

    python scripts/18_eval_ilpcsr_layers.py
    python scripts/18_eval_ilpcsr_layers.py --limit 50 --no-ablations     # quick run
    # in parallel (about 2 hours on 4 cores instead of 6 on one), then merge:
    python scripts/18_eval_ilpcsr_layers.py --skip-answers --sets e1_ilpcsr --tag a &
    python scripts/18_eval_ilpcsr_layers.py --skip-answers --sets e1q_ilpcsr_short --tag b &
    python scripts/18_eval_ilpcsr_layers.py --skip-answers --sets e1s_ilpcsr_statutes --tag c &
    python scripts/18_eval_ilpcsr_layers.py --skip-answers --sets e2_collision e4_multilingual \
        e6_jurisdiction e7_temporal e9_procedure_temporal --tag d &
    wait; python scripts/18_eval_ilpcsr_layers.py --merge
"""

from __future__ import annotations

import argparse
import csv
import time

from kanoon_bridge.agent.research import ResearchAgentV2
from kanoon_bridge.config import project_path
from kanoon_bridge.eval import ilpcsr_layers as IL
from kanoon_bridge.eval import layers_eval as L
from kanoon_bridge.rag.answer import RagPipeline
from kanoon_bridge.search import SearchEngine


def write(rows: list[dict], name: str) -> None:
    path = project_path("results/tables") / f"{name}.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="queries per set")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--sets", nargs="*", default=list(IL.SETS))
    ap.add_argument("--ablation-sets", nargs="*", default=[s for s in IL.SETS if not s.startswith("e1")],
                    help="sets the ablations run on (default: all but the three E1 sets, for time)")
    ap.add_argument("--systems", nargs="*", default=None, help="run only these systems (others' runs are reused)")
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--skip-answers", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the retrieval tables, to run sets in parallel processes")
    ap.add_argument("--merge", action="store_true", help="merge the tagged retrieval tables; Holm over all")
    args = ap.parse_args()
    t0 = time.perf_counter()
    if args.merge:
        rows, comps = IL.merge(project_path("results/tables"))
        write(rows, "ilpcsr_layers_retrieval")
        write(comps, "ilpcsr_layers_significance")
        print(f"merged {len(rows)} score rows, {len(comps)} comparisons")
        return
    engine = SearchEngine.load()

    if not args.skip_answers:
        rows, dump = IL.rag_questions(engine)
        write(rows, "ilpcsr_layers_rag")
        IL.write_jsonl(dump, project_path("results/tables") / "ilpcsr_layers_rag_answers.jsonl")
        for r in rows:
            if r["id"] == "MEAN":
                print("E18a", r)
        rag = RagPipeline.load(engine)
        qa, answers = L.punishment_qa(engine, rag, ResearchAgentV2.load(engine, docs=rag.docs),
                                      L.punishment_questions(limit=args.limit))
        write(qa, "ilpcsr_layers_punishment_qa")
        inj = []
        for base, system in (("RAG v2", "core + RAG v2"), ("RAG v1", "core + RAG v1")):
            inj += [{"base": base, **r} for r in L.error_injection(rag, answers[system])]
        write(inj, "ilpcsr_layers_error_injection")
        for r in qa:
            print(f"E18b {r['system']:20s} statute@1 {r['statute@1']:.3f}  punishment_ok {r['punishment_ok']:.3f}")
        for r in inj:
            print(f"E18b [{r['base']}] {r['error']:38s} v1 {r['v1_detected']}  v2 {r['v2_detected']}  (n={r['n']})")
        print(f"answers done in {time.perf_counter() - t0:.0f}s")

    if not args.skip_retrieval:
        rows, comps = IL.retrieval(engine, args.sets, args.limit, ablations=not args.no_ablations,
                                    ablation_sets=args.ablation_sets, only_systems=args.systems)
        tag = f"_{args.tag}" if args.tag else ""
        write(rows, f"ilpcsr_layers_retrieval{tag}")
        write(comps, f"ilpcsr_layers_significance{tag}")
        for c in comps:
            print(f"E17 {c['set']:22s} {c['a']:9s} -> {c['b']}  {c['metric']:3s} {c['mean_a']:.4f} -> {c['mean_b']:.4f}  "
                  f"diff {c['diff']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  p_holm {c['p_holm']}")
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
