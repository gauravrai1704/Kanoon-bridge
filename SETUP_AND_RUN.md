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
make ltr       # scripts/06_train_ltr.py    → learning-to-rank model trained on IL-PCSR val (prints 5-fold CV MAP)
# optional, Kashvi: make dense  (then set dense.enabled: true in configs/default.yaml)
```

`make index` keeps memory down: each judgment-section index stores word counts only, and only the whole-judgment index keeps word positions, packed 4 bytes each. On a synthetic benchmark of 400 judgments of 7,500 words, memory fell from about 400 MB to 140 MB, so expect roughly 1–1.5 GB for all of IL-PCSR. To check on your machine:

```bash
/usr/bin/time -v make index 2>&1 | grep -E "Maximum resident|Elapsed"     # Linux (macOS: /usr/bin/time -l)
```

It also writes the corpus spellings (for "did you mean") and the near-duplicate groups (MinHash). It prints how many it found. `make ltr` runs about 627 searches and takes a few minutes. Rerun it whenever the indexes, the dense setting or the features change; an outdated model is refused with a warning.

### Optional: the dense channel (multilingual embeddings, GPU recommended)

```bash
pip install -e ".[dense]"            # sentence-transformers + torch
make dense                           # = python scripts/04_encode_dense.py → data/processed/embeddings/
# then in configs/default.yaml:  dense.enabled: true
```

It encodes every zone paragraph once and scores a document by its best paragraph (MaxP). The model is `dense.model`, NLLB-E5. Check the exact Hugging Face id before a long run. If the model can't be loaded, the encoder falls back to `dense.fallback_model` (multilingual-e5-base) and says so. On a CPU, expect hours for ~3k long judgments; a free Colab GPU takes well under an hour. If `dense.enabled` is true but the embeddings are missing, search warns and runs without the dense channel.

Sanity check:

```bash
python app/cli.py "BNS 103" --date 2025-01-03 --debug
python app/cli.py "punishment under section 302" --date 2023-05-10 --debug      # bare 302 → IPC by date
python app/cli.py "bail in dowry death" --state delhi --debug                  # jurisdiction-aware authority
python app/cli.py "mere bhai ko chaku maara" --state delhi --date 2025-03-01 --debug   # Hinglish → lexicon
python app/cli.py "मेरे भाई को चाकू मारा" --state delhi --date 2025-03-01 --debug      # Devanagari, same result
python app/cli.py '"dowry death" AND NOT acquittal' --debug                      # Boolean: phrase, AND, NOT
python app/cli.py 'bail /5 parity state:delhi' --debug                           # proximity + facet filter
python app/cli.py 'BNS 103 AND knife' --debug                                    # section = offence in either code
```

The query syntax is:

| Syntax | Meaning |
| --- | --- |
| `AND` `OR` `NOT` | Boolean operators; they must be uppercase. Adjacent words mean AND. |
| `"..."` | phrase |
| `t1 /k t2` | the two terms within k words of each other |
| `( )` | grouping |
| `extort*` `*bail` | wildcard (works inside Boolean queries too) |
| `state:` `date:` `code:` `court:` | filters |
| `after:2015` `before:2020-06-30` | precedents decided in a date range |

Matching documents are ranked on the query's positive words. Section mentions such as "BNS 103" or "Section 302 IPC" stay one term and match the same offence under either code.

### Search features to try

```bash
python app/cli.py "punishmnt for murdr with a knfe" --date 2025-01-03     # Did you mean: punishment for murder with a knife
python app/cli.py "extort* threat" --debug                                # wildcard expansion in the trace
python app/cli.py "dowry death" --state delhi --snippets 5                # snippets with **highlights** + "why" lines
python app/cli.py "knife attack" --feedback P01,P06                       # Rocchio relevance feedback (use real ids)
python app/cli.py "" --like <precedent_id>                                # similar cases (text + coupling + co-citation)
python app/cli.py "dowry death" --similar                                 # results, then cases like the top one
python app/cli.py "dowry death" --no-ltr                                  # hand-set weights instead of the learned ranker
```

Every precedent shows a query-biased snippet and a "why" line. The line names the offence matched across codes, where the match is, binding or persuasive for your state, and old-code relevance. Statutes show their version note (`BNS 103 <- IPC 302 (same offence)`). Near-duplicate judgments are collapsed to one. `--no-snippets` prints ids and scores only.

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
| `results/tables/typos.csv`, `typo_queries.jsonl` | typo robustness: known-item statute search with simulated misspellings; spelling correction off vs on, plus the clean titles (MRR@10, Success@1/10) |
| `results/tables/agent_e1_ilpcsr.csv` (and e6 once judged) | core vs agent (RRF) vs agent (CombSUM); searches per question, sub-query kinds, share reformulated and share where reformulation helped |
| `results/tables/rag.csv`, `rag_answers.jsonl` | layer 3: supported-sentence rate, version flags, abstention precision/recall |
| `results/tables/significance.csv` | paired randomization tests + 95% CIs, Holm-corrected, for every comparison above |
| `results/runs/*.run` | TREC run files: `query_id Q0 doc_id rank score system` |
| `results/figures/*.png` | ablation, baseline-vs-full, agent, language, efficiency, RAG charts |

How E1 is run:

- Each IL-PCSR test judgment is a query.
- Its masked text is capped to its **100 highest-idf terms** (`e1_max_query_terms` in `configs/eval.yaml`). Whole judgments run to thousands of terms and BM25F cost grows with query length.
- The incident date is the judgment's decision date, and the state is its court's state.
- `F1@k_val` follows the IL-PCSR protocol: k is chosen on the val (dev) split, then reported on test.
- MAP, MRR and nDCG are reported as well.

### Fill the test sets

E2–E7 ship with `EXAMPLE` rows only.

**Generated sets (E2, E3, E7).** Run this after `make data`:

```bash
make testsets             # = python scripts/07_make_test_sets.py   (or: --sets e3 e7, --limit 200)
```

| Set | Queries | Correct answer |
| --- | --- | --- |
| `e3_cross_version` + `e3_control` | Each IL-PCSR test judgment's opening facts plus the sections it applied, named in BNS (`; charged under BNS 103`) or in IPC (control) | The IL-PCSR citations, unchanged |
| `e7_temporal` | Each offence with sections in both codes, asked on 2024-03-01 and on 2024-09-01 | The IPC section(s), then the BNS section(s) |
| `e2_collision` | Section numbers that mean different offences in the two codes, asked bare and with context, before and after 1 July 2024 | The reading in force; the other one goes in `wrong_refs` |

Every generated row is marked `"generated": true`. Their answers are only as good as the crosswalk, so check it first with `make compare`.

**Hand-built sets (E4 Hinglish, E6 jurisdiction).** `scripts/08_make_judged_sets.py` (run by
`make testsets`) writes both: E4's 20 needs × 3 languages with the gold sections chosen for each
need, and E6's 10 topics × 4 states with citation-based topical relevance. The E4 needs and
sections were drafted with AI help: review them (edit the `NEEDS` list in the script) and,
for a second judgment and Cohen's kappa, follow the steps below.

1. Write the queries in `data/queries/e4_multilingual.jsonl` (the same need in `en` / `hi` / `hinglish`, sharing a `need_id`) and `data/queries/e6_jurisdiction.jsonl` (the same text, once per state). Delete the `EXAMPLE` rows.
2. Produce results to judge:
   ```bash
   for s in e4_multilingual e6_jurisdiction; do
     python -m kanoon_bridge.eval.run_eval --set $s --system baseline
     python -m kanoon_bridge.eval.run_eval --set $s --system full
   done
   ```
3. Two judges per set. Each judge's file is created automatically, and both systems' top 10 are pooled and mixed, so a judge can't tell which system found a document:
   ```bash
   python scripts/judge_queries.py --set e4_multilingual --judge gaurav      # -> qrels/e4_multilingual.gaurav.tsv
   python scripts/judge_queries.py --set e4_multilingual --judge kashvi
   ```
   Grades are 0, 1 or 2 (`s` skips, `q` quits; you can resume any time). For E6, grade topical relevance only; binding or persuasive is added automatically.
4. Agreement, disagreements and the merged file:
   ```bash
   python scripts/merge_judgments.py --set e4_multilingual
   ```
   This prints percent agreement and Cohen's kappa (report it; above 0.6 is good). It lists the pairs graded 2 apart in `results/tables/e4_multilingual_disagreements.tsv`; settle those together and edit the merged `data/queries/qrels/e4_multilingual.tsv`. The merged file is what `make eval` uses.

### Significance

`make eval` ends with paired randomization tests: 10,000 permutations, a 95% bootstrap confidence interval, and Holm correction across all comparisons (Smucker, Allan & Carterette, CIKM 2007). It covers:

- baseline vs full on every set,
- each ablation step vs the previous one,
- E3's IPC wording vs BNS wording.

The output is `results/tables/significance.csv`; † marks p_holm < 0.05. To run it on its own:

```bash
make significance                                                       # everything, from the existing run files
python -m kanoon_bridge.eval.significance --set e1_ilpcsr --a baseline --b full
python -m kanoon_bridge.eval.significance --set e1_ilpcsr --ladder
python -m kanoon_bridge.eval.significance --e3
```

In the report, call a difference an improvement only when it is significant after Holm correction, and give its interval.

Rules that keep the numbers honest:

- `train` is for building (the citation graph uses train qrels only).
- `val` is for tuning.
- `test` is only for reporting.

## 7. Layer 2: the research agent

```bash
python app/cli.py "BNS 103 knife stabbing" --state delhi --date 2025-01-03 --agent --debug
python app/cli.py "..." --agent --fusion combsum            # CombSUM instead of reciprocal rank fusion
python app/cli.py "..." --agent --planner hybrid            # + Claude-proposed sub-queries (needs the key, step 8)
```

`--debug` prints the plan and each sub-query's hit count. Each precedent shows which sub-queries found it. One question becomes these sub-queries, each a normal layer-1 search:

1. **original**: the question as asked.
2. **cross_code**: BNS 103 rewritten as IPC Section 302, which brings back judgments decided under the old code.
3. **boolean**: `(murder offence OR its IPC/BNS sections) AND <most salient term>`, run as a postings filter and then ranked.
4. **facet**: only Supreme Court and own-High-Court precedents (when a state is given).
5. **statutes**: a statute search with the offence names spelled out.

The lists are fused. If fewer than 10 precedents come back, or QPP confidence is low, a second round adds pseudo-relevance-feedback terms from the top 5 documents.

All of this is tunable in the `agent:` block of `configs/default.yaml`: planner, fusion, rounds, PRF terms and thresholds. On E1 the agent runs about 5–6 searches per query, so expect `make eval` to take several times longer for the agent rows. Use `--skip-agent` while iterating.

## 8. Layer 3: RAG answers

```bash
cp .env.example .env              # then put your key in .env:  ANTHROPIC_API_KEY=sk-ant-...
python app/cli.py "What is the punishment for murder?" --date 2025-01-03 --state delhi --answer
python app/cli.py "What is the punishment for murder?" --date 2025-01-03 --answer --generator extractive   # no key needed
python app/cli.py "..." --agent --answer                                                                    # on top of layer 2
```

The pipeline (`src/kanoon_bridge/rag/`):

1. **abstain**: no LLM call when QPP finds no specific query term, nothing is retrieved, or the score head is flat. After chunking, it also abstains when the sources cover under 50% of the question's idf-weighted terms. Terms no index has ever seen ("GST") count fully against coverage.
2. **chunker**: top statutes, labelled "BNS Section 103 - Punishment for murder", then the best ratio/decision paragraph of each top precedent.
3. **generate**: Claude at temperature 0, 3–5 sentences, each ending in one `[n]` citation. The default model is `claude-haiku-4-5-20251001`. Change it with `rag.model` in `configs/default.yaml` or `KB_RAG_MODEL` in `.env`.
4. **citation_check**: tf-idf cosine of each sentence vs the chunk it cites. Below 0.2 it is flagged unsupported.
5. **version_check**: flags an IPC section cited for an incident on or after 1 July 2024 (or BNS before), with the equivalent section. A bare "302" gets both readings with confidences.

The RAG evaluation uses `data/queries/rag_questions.jsonl`: 16 answerable questions and 4 out-of-scope ones, each marked with `"answerable"`. Edit or extend it freely.

With a key, `make eval` also asks Claude **closed-book** (no sources) and checks those sentences against the same retrieved chunks. That gives the "supported-sentence rate, RAG vs no retrieval" comparison. With the extractive generator, the support rate is about 1 by construction, so report the Claude numbers. The API key stays in `.env`, which is git-ignored. Declare Claude use in `docs/ai_use.md`.

## 9. The web app

```bash
make web                          # = python app/web_server.py, then open http://localhost:8000
python app/web_server.py --port 9000 --host 0.0.0.0     # another port, or reachable from your network
```

It needs no extra packages (Python's standard library serves it). The fonts are bundled, so it also works offline.

- **Search on the bridge.** Type a question in English, Hindi or Hinglish, or a section (`BNS 103`), Boolean syntax, wildcards, or filters (`after:2015`). Pick a state and an incident date. Two switches turn on the advocate's written answer (layer 3) and the research agent (layer 2).
- **Your advocate.** Adv. Meera or Adv. Kabir (choose in the sidebar) delivers the answer in a speech bubble. Each sentence links to its source, and the checks appear as red-tape notes (weak support, wrong code for the date, an ambiguous bare number). If the system declines to answer, the advocate says why and gives a short brief built from the results.
- **What was understood.** Chips show the language, the Hinglish reading, the code in force, crossings like `BNS 103 ⌒ IPC 302`, the Boolean query, and "Did you mean" for corrected spellings.
- **Judgments.** Each one has a highlighted extract, a binding or persuasive badge for your state, "why this result" reasons, similar cases, score details, and "Mark relevant" (then **Refine with feedback**). Click a title to read the judgment by zone.
- **The law.** Statute cards are marked "in force on your date", with their IPC↔BNS version notes.
- **How this was found** (gold button). A step-by-step replay of the run: understanding the question, finding the law, finding cases, the agent's sub-searches and fusion, and writing and checking the answer. It shows what each step actually did and its time. **Show raw response** shows the JSON.
- **History.** The sidebar keeps your questions in this browser. Click one to run it again. Every search also has a shareable URL (`#q=...&state=...&date=...`).

The older Streamlit page is still there (`make app`).

## 10. No Hugging Face access yet? Run everything on the synthetic sample

```bash
make sample                       # synthetic IL-PCSR-shaped data → corpus → index → graph
make eval-sample                  # all evals + figures on the sample (extractive RAG)
```

The sample cases are fictional (`tests/data/ilpcsr_sample`), and sample results go to `results/sample/` (git-ignored). Use it only to check that everything runs, and never report its numbers. Run `make data && make index && make graph` again afterwards to go back to the real data.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `401` / `GatedRepoError` in `make fetch` | Accept the dataset terms on its HF page, then `huggingface-cli login` again with a **read** token. |
| `pip install -e ".[data]" first` | `pip install -e ".[data]"` (needs `datasets`). |
| `dense channel off: ... not found` warning | `dense.enabled` is true but `make dense` has not been run (step 5). |
| `could not parse Boolean query ...` warning | Unbalanced parentheses or quotes; the query was ranked as plain text instead. |
| Old index files fail to load after an update | Rebuild: `make index graph` (indexes are pickles of the current classes). |
| `FileNotFoundError: ... index/...pkl` | Run `make data index graph` (in that order). |
| `PDF not found` in `make compare` | Save the PDF at `data/raw/crosswalk/comparison_summary_BNS_to_IPC.pdf` or pass `--pdf`. |
| `make compare` reads 0 PDF rows | Run with `--debug`. Check the header keywords in `extract_rows_from_pdf`. A scanned (image) PDF needs OCR first. |
| `ANTHROPIC_API_KEY is not set` | Put it in `.env`, or use `--generator extractive`. |
| Claude model not found | Set `KB_RAG_MODEL` in `.env` to a model your key can use. |
| E1 is slow | `--limit 50` while developing. Lower `e1_max_query_terms` only if you report it. |
| A test set is "skipped (0 judged)" | Fill its qrels TSV (step 6). |
