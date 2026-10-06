# Setup and run: from a fresh machine to every table, figure and answer

This guide takes you from an empty folder to:

- every dataset downloaded,
- the crosswalk cross-checked against the government PDF,
- the indexes built,
- all evaluations run (E1–E7, ablations, efficiency, agent, RAG),
- the figures drawn,
- and the CLI, the demo app and the RAG layer answering questions.

Commands assume Linux or macOS with `make`. On Windows, either use WSL or run the `python ...` command shown next to each `make` target.

Time on a laptop, roughly:

| Step | Time |
| --- | --- |
| Downloads | 10–20 min (IL-PCSR is about 1 GB of text) |
| Corpus + index build | 5–15 min |
| Full evaluation (627 test + 627 val queries per system) | 30–90 min |
| `--limit 50` quick pass | a few minutes |

---

## 0. Prerequisites

| Need | Why |
| --- | --- |
| Python 3.10+ and `git` | everything |
| A Hugging Face account | IL-PCSR is a gated dataset. You must accept its terms (CC-BY-NC-SA 4.0, non-commercial). |
| An Anthropic API key (optional) | Layer 3 with Claude. Without one, the RAG layer uses an offline extractive generator. |
| About 3 GB of free disk | raw data, processed corpus, indexes |

## 1. Install

```bash
git clone <repo-url> kanoon-bridge && cd kanoon-bridge
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,data,app,rag]"                      # core + tests + HF datasets + streamlit + Claude SDK
# optional, Kashvi's dense channel (large download, GPU recommended):
# pip install -e ".[dense]"
make test                                                 # expect: all passed, some skipped (s = teammate TODOs)
```

Skipped tests are spec tests for code a teammate has not written yet. They run for real once that code lands.

## 2. Download IL-PCSR (Hugging Face, gated)

1. Log in at https://huggingface.co and open https://huggingface.co/datasets/Exploration-Lab/IL-PCSR.
2. Click **Agree and access repository**.
3. Create a **read** token at https://huggingface.co/settings/tokens.
4. Log in from the terminal and fetch:

```bash
huggingface-cli login            # newer versions: hf auth login   (paste the read token)
make fetch                       # = python scripts/00_fetch_data.py (IL-PCSR + BNS source)
#   python scripts/00_fetch_data.py --only ilpcsr   # just one of them
```

This writes `data/raw/ilpcsr/{queries,statutes,precedents}/<split>.parquet`:

| Config | Split | Rows |
| --- | --- | --- |
| queries | train_queries | 5,017 |
| queries | dev_queries | 627 |
| queries | test_queries | 627 |
| statutes | statute_candidates | 936 |
| precedents | precedent_candidates | 3,183 |

Every later step reads these local files, so nothing after this needs the network.

Check the schema and look at one example:

```bash
make inspect                     # = python -m kanoon_bridge.ingest.load_ilpcsr
```

## 3. Download the BNS source

`make fetch` already ran a `git clone` of https://github.com/PSKprem/bns-study-platform into `data/raw/bns-study-platform/`. That gives 358 BNS sections, each with its bare-act text and the IPC section(s) it replaces.

We use only the bare-act text (government material) and the section correspondences. We do not use any of that repository's commentary.

```bash
python scripts/00_fetch_data.py --only bns     # if you skipped it
```

## 4. The government comparison PDF, and the cross-check

The crosswalk (`data/crosswalk/ipc_bns.csv`, 496 IPC↔BNS pairs) is generated from the BNS JSON. The government comparison table is a second, independent source. Download it **by hand in a browser**, because some government sites block scripts:

- Kerala Prisons copy: https://www.keralaprisons.gov.in/userfiles/act-and-rules/comparison_summary_BNS_to_IPC.pdf
- The same "Comparison Summary BNS to IPC" table is also published by BPR&D / MHA. Any copy with a BNS-section column and an IPC-section column works.

Save it exactly here:

```
data/raw/crosswalk/comparison_summary_BNS_to_IPC.pdf
```

Then run:

```bash
make crosswalk                   # regenerate ipc_bns.csv + offence_ids.csv from the BNS JSON
make compare                     # = python -m kanoon_bridge.ingest.parse_crosswalk --compare
# a PDF saved elsewhere / see what the parser reads:
python -m kanoon_bridge.ingest.parse_crosswalk --compare --pdf path/to/file.pdf --debug
```

The comparison output looks like this:

```
PDF pairs: 4xx  agree: 4xx/494 JSON pairs  only in JSON: N  only in PDF: M
disagreements written to data/processed/crosswalk_compare.csv
```

How the PDF parser works (`extract_rows_from_pdf` in `ingest/parse_crosswalk.py`):

- It uses pdfplumber tables.
- The header row decides which column is BNS and which is IPC. It recognises "BNS", "Bharatiya Nyaya Sanhita", "IPC" and "Indian Penal Code".
- The columns carry over to continuation pages that have no header.
- "New section" rows are skipped.
- Ranges like "342-344" are expanded.
- If the PDF has no ruled tables, it falls back to a line-by-line text mode.

If it reads 0 rows, run with `--debug`. That prints the first table rows pdfplumber sees, so you can adjust the header keywords at the top of that function.

What to do with the disagreements:

1. Open `crosswalk_compare.csv` and check each pair against the bare acts. Prioritise the sections in the E2/E3/E7 sets: 302/103, 304B/80, 498A/85-86, 506/351, 420/318.
2. If the PDF is right, fix the source (or add an override) and rerun `make crosswalk`.
3. Report the agreement rate in the report. It is a nice data-quality number.

## 5. Build the corpus, indexes and citation graph

```bash
make data      # scripts/01_build_corpus.py → data/processed/docs.jsonl (statutes, precedents, query cases)
make index     # scripts/02_build_index.py  → data/processed/index/ (zone + facet indexes, statute terms)
make graph     # scripts/03_build_graph.py  → citation graph (train qrels only), authority, tiers
# optional, Kashvi: make dense  (then set dense.enabled: true in configs/default.yaml)
```

> **Until Sharanya's index code is merged**, `make index` stops with `NotImplementedError: TODO(A)`.
> To run everything anyway, use the temporary reference shim. It patches only the index methods that still raise NotImplementedError, and switches itself off method by method as her code lands:
>
> ```bash
> export KB_DEV_SHIM=1              # Windows PowerShell: $env:KB_DEV_SHIM="1"
> ```
>
> Keep it exported for every later command too: index, graph, eval, the CLI and the app. The pickled index objects need the same methods at load time. Each run prints a `[KB_DEV_SHIM] ...` line, so you always know it is on. Delete `src/kanoon_bridge/dev/` once her tests pass. **The graded index implementation is hers.**

Sanity check:

```bash
python app/cli.py "BNS 103" --date 2025-01-03 --debug
python app/cli.py "punishment under section 302" --date 2023-05-10 --debug      # bare 302 → IPC by date
python app/cli.py "bail in dowry death" --state delhi --debug                  # jurisdiction-aware authority
```

## 6. Run the evaluations

```bash
make eval-quick           # first 50 queries per set: checks the pipeline in a few minutes
make eval                 # everything (= python scripts/05_run_all_evals.py)
```

Useful flags:

```bash
python scripts/05_run_all_evals.py --sets e1_ilpcsr e1s_ilpcsr_statutes
python scripts/05_run_all_evals.py --skip-ablation --skip-efficiency --skip-agent --skip-rag
python scripts/05_run_all_evals.py --rag-generator extractive
make plots                # redraw figures from existing tables, no retrieval
```

What runs:

| Output | What it is |
| --- | --- |
| `results/tables/main_results.csv` | BM25 baseline vs full system on every set that has queries and judgments |
| `results/tables/ablation_e1_ilpcsr.csv` (and e2) | ladder bm25 → +zones → +code_filter → +bridge → +authority → +dense → +qpp → +jurisdiction |
| `results/tables/language_e4.csv` | E4 analyzer steps switched on one at a time (needs E4 qrels) |
| `results/tables/efficiency.csv` | exhaustive vs tiered vs champion lists: median/p95 latency and Recall@20 vs exhaustive (200 E1 queries) |
| `results/tables/agent_e1_ilpcsr.csv` | core single query vs research agent (RRF); searches per question |
| `results/tables/rag.csv`, `rag_answers.jsonl` | layer 3: supported-sentence rate, version flags, abstention precision/recall |
| `results/runs/*.run` | TREC run files: `query_id Q0 doc_id rank score system` |
| `results/figures/*.png` | ablation, baseline-vs-full, language, efficiency, RAG charts |

How E1 is run:

- Each IL-PCSR test judgment is a query.
- Its masked text is capped to its **100 highest-idf terms** (`e1_max_query_terms` in `configs/eval.yaml`). Whole judgments run to thousands of terms and BM25F cost grows with query length.
- The incident date is the judgment's decision date, and the state is its court's state.
- `F1@k_val` follows the IL-PCSR protocol: k is chosen on the val (dev) split, then reported on test.
- MAP, MRR and nDCG are reported as well.

Hand-built sets (E2–E7) live in `data/queries/*.jsonl` and `data/queries/qrels/*.tsv`. They currently hold only `EXAMPLE` rows. Fill them with `scripts/judge_queries.py` (two judges per query), following the field notes in `configs/eval.yaml`:

| Set | What its rows need |
| --- | --- |
| E2 | `wrong_refs` |
| E3 | `source_query_id` (an IL-PCSR test query id) |
| E4 | `lang` |
| E6 | `state` |
| E7 | `incident_date` |

A set with no judgments is skipped, except E2 (wrong_hit@10) and E7 (code_accuracy@1), whose headline metrics need no qrels.

Rules that keep the numbers honest:

- `train` is for building (the citation graph uses train qrels only).
- `val` is for tuning.
- `test` is only for reporting.

## 7. Layer 3: RAG answers

```bash
cp .env.example .env              # then put your key in .env:  ANTHROPIC_API_KEY=sk-ant-...
python app/cli.py "What is the punishment for murder?" --date 2025-01-03 --state delhi --answer
python app/cli.py "What is the punishment for murder?" --date 2025-01-03 --answer --generator extractive   # no key needed
python app/cli.py "..." --agent --answer                                                                    # on top of layer 2
```

The pipeline (`src/kanoon_bridge/rag/`):

1. **abstain**: QPP. No specific query term, nothing retrieved, or a flat score head means no LLM call.
2. **chunker**: top statutes, labelled "BNS Section 103 - Punishment for murder", then the best ratio/decision paragraph of each top precedent.
3. **generate**: Claude at temperature 0, 3–5 sentences, each ending in one `[n]` citation. The default model is `claude-haiku-4-5-20251001`. Change it with `rag.model` in `configs/default.yaml` or `KB_RAG_MODEL` in `.env`.
4. **citation_check**: tf-idf cosine of each sentence vs the chunk it cites. Below 0.2 it is flagged unsupported.
5. **version_check**: flags an IPC section cited for an incident on or after 1 July 2024 (or BNS before), with the equivalent section. A bare "302" gets both readings with confidences.

The RAG evaluation uses `data/queries/rag_questions.jsonl`: 16 answerable questions and 4 out-of-scope ones, each marked with `"answerable"`. Edit or extend it freely.

With a key, `make eval` also asks Claude **closed-book** (no sources) and checks those sentences against the same retrieved chunks. That gives the "supported-sentence rate, RAG vs no retrieval" comparison. With the extractive generator, the support rate is about 1 by construction, so report the Claude numbers. The API key stays in `.env`, which is git-ignored. Declare Claude use in `docs/ai_use.md`.

## 8. Demo app

```bash
make app                          # streamlit run app/streamlit_app.py
```

The app has a query box, a state picker and an incident date. It shows statutes and precedents with score breakdowns and the query-analysis trace. A **Grounded answer (layer 3)** checkbox adds the RAG answer with its flags.

## 9. No Hugging Face access yet? Run everything on the synthetic sample

```bash
export KB_DEV_SHIM=1              # until the index code lands
make sample                       # synthetic IL-PCSR-shaped data → corpus → index → graph
make eval-sample                  # all evals + figures on the sample (extractive RAG)
```

The sample cases are fictional (`tests/data/ilpcsr_sample`). Use it only to check that everything runs, and never report its numbers. Run `make data && make index && make graph` again afterwards to go back to the real data.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `401` / `GatedRepoError` in `make fetch` | Accept the dataset terms on its HF page, then `huggingface-cli login` again with a **read** token. |
| `pip install -e ".[data]" first` | `pip install -e ".[data]"` (needs `datasets`). |
| `NotImplementedError: TODO(A) ...` in index/eval | Index code not merged yet. `export KB_DEV_SHIM=1` (see step 5). |
| `NotImplementedError: TODO(C) ...` | Kashvi's step. The analyzer skips it and notes this in the `--debug` trace. Dense stays off until `dense.enabled: true`. |
| `FileNotFoundError: ... index/...pkl` | Run `make data index graph` (in that order). |
| `PDF not found` in `make compare` | Save the PDF at `data/raw/crosswalk/comparison_summary_BNS_to_IPC.pdf` or pass `--pdf`. |
| `make compare` reads 0 PDF rows | Run with `--debug`. Check the header keywords in `extract_rows_from_pdf`. A scanned (image) PDF needs OCR first. |
| `ANTHROPIC_API_KEY is not set` | Put it in `.env`, or use `--generator extractive`. |
| Claude model not found | Set `KB_RAG_MODEL` in `.env` to a model your key can use. |
| E1 is slow | `--limit 50` while developing. Lower `e1_max_query_terms` only if you report it. |
| A test set is "skipped (0 judged)" | Fill its qrels TSV (step 6). |
