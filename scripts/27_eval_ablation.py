"""E25: context ablation of the E20 RAG answers (eval/ablation.py).  [owner: Gaurav — working]

Needs the corpus E20 ran on and, for the LLM part, the same model through Ollama:

    python scripts/27_eval_ablation.py --diagnose-only                      # no LLM, seconds per answer
    OLLAMA_CONTEXT_LENGTH=8192 ollama serve
    python scripts/27_eval_ablation.py --model llama3.2:3b                  # control + ablated, 2 calls per answer
    python scripts/27_eval_ablation.py --model llama3.2:3b --only-misgrounded   # quick: the misgrounded ones only

    --tag llama3_2_3b reads results/tables/misgrounding_<tag>_answers.jsonl (E20) and writes
    results/tables/ablation_<tag>.csv               per condition and set
    results/tables/ablation_<tag>_significance.csv  control vs ablated, exact McNemar + Holm
    results/tables/ablation_<tag>_rows.jsonl        every answer (diagnose-only: ablation_<tag>_context.*)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

from kanoon_bridge.agent.research import ResearchAgentV2
from kanoon_bridge.config import project_path
from kanoon_bridge.eval import ablation as A
from kanoon_bridge.eval.misgrounding import rescore
from kanoon_bridge.rag.answer import RagPipeline
from kanoon_bridge.search import SearchEngine


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="llama3_2_3b", help="the E20 run to ablate")
    ap.add_argument("--generator", default="ollama", help="ollama | local | openai | claude")
    ap.add_argument("--model", default=None, help="the model E20 used (default: the generator's default)")
    ap.add_argument("--conditions", nargs="*", default=list(A.CONDITIONS))
    ap.add_argument("--only-misgrounded", action="store_true", help="ask again only the misgrounded E20 answers")
    ap.add_argument("--diagnose-only", action="store_true", help="context statistics only, no LLM call")
    args = ap.parse_args()
    tables = project_path("results/tables")
    src = tables / f"misgrounding_{args.tag}_answers.jsonl"
    if not src.exists():
        raise SystemExit(f"{src} not found - run E20 first (scripts/run_e20_ollama.sh)")
    gen = None if args.diagnose_only else args.generator
    if args.model:
        os.environ[{"claude": "KB_RAG_MODEL", "openai": "OPENAI_RAG_MODEL"}.get(args.generator, "KB_LLM_MODEL")] = args.model
    if gen:
        from kanoon_bridge.rag.generate import call_llm

        try:
            call_llm(gen, "Reply with the single word OK.", args.model, max_tokens=5)
        except Exception as err:                                  # noqa: BLE001
            raise SystemExit(f"{gen} is not answering ({err}); start it with "
                             "`OLLAMA_CONTEXT_LENGTH=8192 ollama serve` and `ollama pull <model>`") from None
    t0 = time.perf_counter()
    engine = SearchEngine.load()
    rag = RagPipeline.load(engine)
    agent = ResearchAgentV2.load(engine, docs=rag.docs)
    _, saved = rescore(src, engine.analyzer.text_res.normalizer)          # the current E20 scorer
    table, rows = A.run(engine, rag, agent, saved, gen, args.conditions, args.only_misgrounded)
    name = f"ablation_{args.tag}" + ("_context" if not gen else "")
    write_csv(table, tables / f"{name}.csv")
    with open(tables / f"{name}_rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if gen:
        write_csv(A.significance(rows), tables / f"{name}_significance.csv")
    for t in table:
        if t["set"] != "all":
            continue
        line = (f"E25 {t['condition']:10s} n={t['n']:4d} E20 misgrounded {t['e20_misgrounded@1']:.3f}  "
                f"context has wrong code {t['context_has_wrong_code']:.2f}  "
                f"misgrounded section in context {t['misgrounded_section_in_context'] or 0:.2f}  "
                f"header match {t['header_match']:.2f}")
        if gen:
            line += (f"\n    control misgrounded {t['control_misgrounded@1']:.3f} -> ablated {t['ablated_misgrounded@1']:.3f}"
                     f"  right {t['control_right@1']:.3f} -> {t['ablated_right@1']:.3f}  "
                     f"cause: context {t['context-induced'] or 0:.2f}, parametric {t['parametric'] or 0:.2f}")
        print(line)
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
