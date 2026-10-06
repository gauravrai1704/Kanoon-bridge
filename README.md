# Kanoon-Bridge

Legal search for Indian criminal law across the July 2024 IPC → BNS change.
One query — in BNS or IPC terms, in English, Hindi or Hinglish — finds the right statute
in both codes and the precedents that apply it, ranked by jurisdiction and date.

CSD358 (Information Retrieval) IR Hackathon · Track T6 (vertical search: law) with a T5 (Hinglish) layer.

> Not legal advice. This is a course research prototype.

---

## Status

| Component | Owner | Status |
| --- | --- | --- |
| Shared schema, config, storage, search entry point | all | working |
| Legal tokeniser (section tokens) | A | working baseline |
| Positional / zone / facet indexes, query parser | A | stub |
| Crosswalk parsing, version normalisation, collision resolver | B | stub |
| Transliteration, phonetic matching, stemming, dense channel | C | stub |
| tf-idf, BM25F, statute bridge, authority, QPP, tiers | D | stub |
| Evaluation metrics | D | working |
| RAG layer (additional work) | first free member | stub |

Update this table as components land. Stubs raise `NotImplementedError` with a TODO saying what to build.

---

## Setup

```bash
git clone <repo-url> kanoon-bridge
cd kanoon-bridge
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core + pytest
pip install -e ".[dense]"          # optional: sentence-transformers for the dense channel
pip install -e ".[app]"            # optional: streamlit demo
make test                          # should pass on a fresh clone
```

Python 3.10+.

## Data

Downloads go in `data/raw/` (gitignored). Run `python scripts/00_check_access.py` first — it reports what is missing.

| Data | Where to get it | Put it in |
| --- | --- | --- |
| IL-PCSR (IIT Kharagpur + IIT Kanpur) | https://huggingface.co/datasets/Exploration-Lab/IL-PCSR | `data/raw/ilpcsr/` |
| BNS bare-act text | India Code (https://www.indiacode.nic.in) | `data/raw/bns/` |
| IPC↔BNS comparison table (government copy) | https://www.keralaprisons.gov.in/userfiles/act-and-rules/comparison_summary_BNS_to_IPC.pdf | `data/raw/crosswalk/` |
| PoliceDrishti (gated, optional) | https://huggingface.co/datasets/happyman11/PoliceDristi | `data/raw/policedrishti/` |
| NLLB-E5 weights (optional) | https://github.com/ArkadeepAcharya/NLLB-E5 | downloaded by `scripts/04_encode_dense.py` |

No crawling. If any is added later: obey robots.txt, rate-limit, collect no personal data.

## How to run

```bash
make data      # scripts/01_build_corpus.py  → data/processed/docs.jsonl
make index     # scripts/02_build_index.py   → data/processed/index/
make graph     # scripts/03_build_graph.py   → citation graph + authority scores
make dense     # scripts/04_encode_dense.py  → paragraph embeddings (GPU recommended)
make eval      # scripts/05_run_all_evals.py → results/tables, results/figures
make app       # streamlit demo
python app/cli.py "mere bhai ko chaku maara" --state delhi --date 2025-03-01 --debug
```

## Repo layout

See `PROJECT_STRUCTURE.md` for every file and what it does.

## Team rules

1. Everything passes `schema.py` objects (`Document`, `Query`, `ScoredDoc`).
2. Documents and queries go through the same `text/` pipeline.
3. No test leakage: graph building and tuning use train/val only.
4. The app and the evaluator both call `kanoon_bridge.search.search()`.
5. Log AI use in `docs/ai_use.md` as you go.

## Credits

IL-PCSR (Paul et al., 2025), Hindi-BEIR / NLLB-E5 (Acharya et al., NAACL 2025), CaseLink (Tang et al., SIGIR 2024),
QPP for agentic RAG (Tian et al., IR-RAG@SIGIR 2025), PoliceDrishti dataset. Spatio-temporal facets inspired by
the work of Dr. Sonia Khetarpaul (SNU).
