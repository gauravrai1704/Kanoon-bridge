"""Run E20 (eval/misgrounding.py): do LLM answers cite the code in force on the incident date?  [owner: Gaurav — working]

Needs a built corpus (make build, or make open) and an LLM. On your own GPU, serve any open model
with an OpenAI-compatible API, e.g.

    vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192            # http://localhost:8000/v1
    ollama serve & ollama pull qwen2.5:7b                                # http://localhost:11434/v1
    python -m llama_cpp.server --model qwen2.5-7b-instruct-q4_k_m.gguf   # http://localhost:8000/v1

then

    KB_LLM_BASE_URL=http://localhost:8000/v1 KB_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \\
        python scripts/19_eval_misgrounding.py --generator local --per-set 100 --tag qwen7b

(or --generator openai / claude with an API key in .env).

    results/tables/misgrounding_<tag>.csv           per set and condition
    results/tables/misgrounding_<tag>_answers.jsonl every answer, for reading
"""

from __future__ import annotations

import argparse
import csv
import json
import time

from kanoon_bridge.agent.research import ResearchAgentV2
from kanoon_bridge.config import project_path
from kanoon_bridge.eval import misgrounding as M
from kanoon_bridge.rag.answer import RagPipeline
from kanoon_bridge.search import SearchEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="local", help="local | openai | claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--per-set", type=int, default=40, help="questions per set (E7, E9, E2)")
    ap.add_argument("--conditions", nargs="*", default=list(M.CONDITIONS))
    ap.add_argument("--tag", default="local")
    args = ap.parse_args()
    t0 = time.perf_counter()
    engine = SearchEngine.load()
    rag = RagPipeline.load(engine)
    agent = ResearchAgentV2.load(engine, docs=rag.docs)
    rows = M.questions(args.per_set)
    print(f"{len(rows)} questions x {len(args.conditions)} conditions, generator {args.generator}")
    table, dump = M.run(engine, rag, agent, rows, args.generator, args.model, args.conditions)
    out = project_path("results/tables")
    with open(out / f"misgrounding_{args.tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in table for k in r)))
        w.writeheader()
        w.writerows(table)
    with open(out / f"misgrounding_{args.tag}_answers.jsonl", "w", encoding="utf-8") as f:
        for r in dump:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    for r in table:
        print(f"E20 {r['set']:22s} {r['condition']:12s} right@1 {r['right@1']:.3f}  misgrounded@1 {r['misgrounded@1']:.3f}  "
              f"any_wrong_code {r['any_wrong_code']:.3f}  no_section {r['no_section']:.3f}  (n={r['n']})")
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
