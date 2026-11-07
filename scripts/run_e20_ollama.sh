#!/usr/bin/env bash
# E20 (LLM temporal misgrounding) with local models through Ollama.
# Needs a built corpus (make build, or make open) and Ollama running:
#     OLLAMA_CONTEXT_LENGTH=8192 ollama serve
# Usage: bash scripts/run_e20_ollama.sh [per-set] [model ...]
#     bash scripts/run_e20_ollama.sh                       # llama3.2:3b, 100 questions per set
#     bash scripts/run_e20_ollama.sh 100 llama3.2:3b llama3.1:8b
# Writes results/tables/misgrounding_<model>.csv and _answers.jsonl for each model.
set -euo pipefail
PER_SET="${1:-100}"
shift || true
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("llama3.2:3b")
for m in "${MODELS[@]}"; do
    ollama pull "$m"
    tag="$(echo "$m" | tr ':.' '__')"
    python scripts/19_eval_misgrounding.py --generator ollama --model "$m" --per-set "$PER_SET" --tag "$tag"
done
