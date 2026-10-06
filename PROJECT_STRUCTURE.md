# Kanoon-Bridge — Project Structure

Read `ARCHITECTURE.md` first for how the pieces fit. This file is the map: every file, who owns
it, whether it works yet, what it is for, and **what it must contain** (the functions and classes
other code calls — keep these names and signatures, or tell the team).

**Owners:** **A** indexing & ingest · **B** version normalisation · **C** Hindi/Hinglish + dense ·
**D** ranking + evaluation · **AG** research agent (layer 2, suggest B) · **R** RAG (layer 3, first free member).

**Status:** ✅ working · 🔧 wiring works, parts are stubs · ⬜ stub (`raise NotImplementedError("TODO(owner): …")`).
Run `python scripts/00_check_access.py` to see how many TODOs each owner has left.

```
kanoon-bridge/
├── README.md                     setup, data, how to run, status
├── ARCHITECTURE.md               end-to-end design (read first)
├── PROJECT_STRUCTURE.md          this file
├── pyproject.toml  requirements.txt  Makefile  .gitignore
├── configs/
│   ├── default.yaml              every tunable value
│   └── eval.yaml                 test sets, ablation ladder
├── data/
│   ├── raw/                      downloads (gitignored)
│   ├── processed/                built artefacts (gitignored)
│   ├── crosswalk/                README.md  ipc_bns.csv  offence_ids.csv  crpc_bnss.csv
│   ├── lexicons/                 hinglish_legal_terms.csv  hindi_stopwords.txt  legal_stopwords.txt  court_to_states.csv
│   └── queries/                  e2…e7 *.jsonl  +  qrels/*.tsv
├── src/kanoon_bridge/
│   ├── schema.py  config.py  search.py
│   ├── ingest/                   load_ilpcsr  load_statutes  parse_crosswalk  segment  metadata
│   ├── text/                     pipeline  tokenize  stem  collision  version_norm  transliterate  phonetic
│   ├── index/                    positional  zones  facets  tiers  store  docstore
│   ├── query/                    parser  boolean  proximity  analyzer
│   ├── rank/                     vsm  bm25f  dense  statute_bridge  citation_graph  authority  qpp  fusion  topk
│   ├── agent/        (layer 2)   plan  executor  fuse  reflect  research
│   ├── rag/          (layer 3)   answer  chunker  generate  citation_check  version_check  abstain
│   └── eval/                     metrics  qrels  run_eval  ablation  efficiency  agreement  agent_eval  plots
├── scripts/                      00_check_access … 05_run_all_evals, judge_queries
├── app/                          cli.py  streamlit_app.py
├── notebooks/                    01_explore_ilpcsr  02_postings_demo  03_results
├── tests/                        conftest + one test file per area
├── results/                      runs/ tables/ (gitignored)  figures/ (committed)
└── docs/                         architecture.svg/.png  make_architecture.py  ai_use.md  report/
```

---

## Root files

| File | Status | Purpose | Must contain |
| --- | --- | --- | --- |
| `README.md` | ✅ | What graders read first: setup, data sources + credits, how to run, what works / what's planned | Keep the status table current; the exact commands that reproduce the demo |
| `ARCHITECTURE.md` | ✅ | End-to-end design, data usage, worked example, evaluation map, gates | Update if a layer's interface changes |
| `pyproject.toml` | ✅ | Makes `kanoon_bridge` installable (`pip install -e ".[dev]"`) | Optional extras: `dev`, `data`, `dense`, `app`, `rag` |
| `requirements.txt` | ✅ | Same deps, for people who prefer `pip install -r` | Optional deps commented |
| `Makefile` | ✅ | One-word commands | `check data index graph dense eval app test all` |
| `.gitignore` | ✅ | Keeps data, artefacts, secrets out of git | `data/raw`, `data/processed`, `results/runs`, `.env`, model weights |

## `configs/`

| File | Status | Purpose | Must contain |
| --- | --- | --- | --- |
| `default.yaml` | ✅ | Every tunable number — no magic numbers in code | `paths`, `legal.bns_in_force`, `text`, `zones`, `bm25`, `statute_bridge`, `authority`, `dense`, `fusion`, `tiers`, `search`, `agent`, `rag` |
| `eval.yaml` | ✅ | What `scripts/05` runs | `k_values`, `test_sets` (queries, qrels, target), `ablation_ladder`, `language_ablation`, `outputs` |

## `data/` (data files, not code)

| Path | Owner | Status | Holds | Format |
| --- | --- | --- | --- | --- |
| `raw/` | A | — | Downloads, never edited: IL-PCSR, BNS text, crosswalk PDF | as downloaded |
| `processed/` | — | — | `docs.jsonl`, `index/*.pkl|json`, `citation_graph.json`, `embeddings/` | built by scripts |
| `crosswalk/ipc_bns.csv` | B | seed rows only | One row per IPC↔BNS pair | `ipc_section, bns_section, relation, note` |
| `crosswalk/offence_ids.csv` | B | seed rows only | Canonical offences | `offence_id, label, ipc_sections, bns_sections, keywords` (`;`-separated lists) |
| `crosswalk/README.md` | B | ✅ | Format notes and the verification warning | — |
| `lexicons/hinglish_legal_terms.csv` | C | seed rows | Roman/Devanagari term → English legal term | `roman, devanagari, english, weight` |
| `lexicons/*_stopwords.txt` | A, C | seed lists | Stop words | one per line, `#` comments |
| `lexicons/court_to_states.csv` | A | seed rows | Court → states it binds (`*` = everywhere) | `court_name, name_pattern, level, states` |
| `queries/e*.jsonl` | all | examples only | Hand-built test queries | one JSON per line: `id, text, lang, state, incident_date, notes` |
| `queries/qrels/*.tsv` | all | headers only | Judgments | `query_id  doc_id  grade  judge` (tab-separated) |

---

## `src/kanoon_bridge/` — shared core

| File | Owner | Status | Purpose | Must contain |
| --- | --- | --- | --- | --- |
| `schema.py` | all | ✅ | The data classes every module passes. Agree in hour 1; add fields with defaults, never ad-hoc dicts | `Document` (doc_id, doc_type, paragraphs, court, states, decision_date, code, statutes_cited, …; `.text`, `.zone_text()`), `Paragraph`, `Query`, `AnalyzedQuery` (`.weighted_terms()`), `Posting`, `ScoredDoc`, `SearchResult`, enums `DocType`/`Code`/`Court`, `ZONES`, `section_ref()`, `read_documents()`/`write_documents()`/`read_queries()` |
| `config.py` | D | ✅ | Loads YAML into attribute-style objects | `load_config(name, overrides)`, `project_path()`, `PROJECT_ROOT` |
| `search.py` | D | ✅ wiring | **Layer 1 entry point.** The CLI, app, agent and evaluator all call it (rule 4) | `SearchOptions` (one switch per component; `.baseline()`, `.from_dict()`), `SearchEngine.load()`, `SearchEngine.search(query, options) -> SearchResult`, module-level `search()` |

### `ingest/` — raw data → `Document`s (A; crosswalk B)

| File | Owner | Status | Purpose | Must contain |
| --- | --- | --- | --- | --- |
| `load_ilpcsr.py` | A | ⬜ | Read IL-PCSR queries, precedents, statutes, qrels. First job in hour 0: `inspect()` the real files | `inspect()`, `load_queries(split)`, `load_precedents()`, `load_statute_candidates()`, `load_qrels(split, target) -> {qid: {doc_id: 1}}` |
| `load_statutes.py` | A | ⬜ | BNS bare act → one `Document` per section | `parse_bns(path) -> list[Document]` (doc_id like `bns:103`, code BNS) |
| `parse_crosswalk.py` | B | ⬜ | Government PDF → the two crosswalk CSVs | `extract_rows(pdf)`, `write_crosswalk(rows, out)`, `build_offence_ids(rows, out)`; runnable as a script |
| `segment.py` | A | 🔧 | Judgments → zone paragraphs ("what is a document") | `ROLE_TO_ZONE`, `CUE_PHRASES`, ✅`split_paragraphs()`, ✅`zone_from_cues()`, ⬜`segment_judgment()`, ⬜`segment_statute()` |
| `metadata.py` | A | 🔧 | Court, states, decision date, code in force — the spatio-temporal facets | ✅`CourtTable.load()/lookup()`, ✅`code_in_force()`, ⬜`find_date()`, ⬜`enrich(doc, table)` |

### `text/` — analysis and normalisation (A, B, C)

| File | Owner | Status | Purpose | Must contain |
| --- | --- | --- | --- | --- |
| `pipeline.py` | all | ✅ | **The one text pipeline** for documents and queries (rule 2). Skips unfinished steps with a warning | `TextOptions`, `TextResources.load()`, `analyze_text(text, resources, lang, date, options) -> list[str]` |
| `tokenize.py` | A | ✅ | Section-aware tokeniser: "u/s 498A IPC" → `sec:ipc:498a`; bare numbers → `sec:?:302` | `tokenize()`, `extract_sections() -> list[SectionMention]`, `is_section_token()`, `SECTION_PREFIX` |
| `stem.py` | C | 🔧 | Porter (✅) and Hindi light stemmer (⬜); never stems `sec:`/`off:` tokens | `stem_tokens(tokens, lang, enabled)`, `stem_english()`, `stem_hindi()` |
| `version_norm.py` | B | 🔧 | **Core novelty.** Section tokens → canonical offence IDs, both codes | ✅`VersionNormalizer.load()`, ⬜`to_offences(ref) -> [(off_id, weight)]`, ⬜`equivalents(ref)`, ⬜`normalize_tokens(tokens)` |
| `collision.py` | B | 🔧 | Resolve bare/colliding numbers from date + context. **Hardest part** | `Reading`, ✅`CollisionResolver.load()/code_for_date()`, ⬜`resolve(section, context, date)`, ⬜`resolve_tokens(tokens, date)` |
| `transliterate.py` | C | 🔧 | Language detection, Devanagari↔Roman, Hinglish spelling, legal lexicon | ✅`detect_lang()`, ⬜`devanagari_to_roman()`, ⬜`normalize_roman()`, ✅`LegalLexicon.load()`, ⬜`.lookup(word)` |
| `phonetic.py` | C | 🔧 | Soundex for Roman Hindi (`hatya = hathya = hattya`) | ✅`soundex()` (baseline), ⬜`hindi_soundex()`, ⬜`PhoneticIndex.build()/matches()` |

### `index/` — data structures (A; tiers D)

| File | Owner | Status | Purpose | Must contain |
| --- | --- | --- | --- | --- |
| `positional.py` | A | 🔧 | Inverted index with positions | `PositionalIndex`: ⬜`add(doc_id, tokens)`, ⬜`phrase(terms)`; ✅ stats `df/idf/tf/postings_for/docs_with/n_docs/avg_doc_len/vocabulary` |
| `zones.py` | A | 🔧 | One positional index per zone + whole-doc index; feeds BM25F | `ZoneIndex`: ⬜`build(docs, analyze)`; ✅`tf(term, doc, zone)`, `zone_len()`, `avg_zone_len()`, `df()`, `.whole` |
| `facets.py` | A | ⬜ | Parametric index: type, court, states, date, code, sections cited | `DocMeta`, `FacetIndex.build(docs)`, `.filter(doc_type, court, states, date_from, date_to, code, cites_any) -> set`, `.meta(doc_id)`, `.by_section` |
| `tiers.py` | D | ⬜ | Tiered index + champion lists (efficiency experiment) | `TieredIndex.build(zidx, authority, quantile, champion_size)`, `.candidates(terms, min_results, use_champions)` |
| `store.py` | A | ✅ | Save/load artefacts under `data/processed/index/` | `save(obj, name, fmt)`, `load(name, fmt)`, `exists()`, `index_dir()` |
| `docstore.py` | all | ✅ | doc_id → `Document` text (agent, RAG, judging tool, app) | `DocStore.load()`, mapping access, `.snippet()` |

### `query/` — understanding input (A; analyzer C)

| File | Owner | Status | Purpose | Must contain |
| --- | --- | --- | --- | --- |
| `parser.py` | A | 🔧 | Query syntax: phrases, AND/OR/NOT, `/k`, filters `state: date: code: court:` | ✅`extract_filters()`, ✅`has_operators()`, `parse() -> ParsedQuery`, `QueryNode`/`Op`; ⬜`_parse_tree()` |
| `boolean.py` | A | ⬜ | Boolean retrieval; intersection in increasing-df order | `intersect(p1, p2)` (two-pointer), `intersect_many(lists)`, `evaluate(node, index, analyze, universe)` |
| `proximity.py` | A | ⬜ | `t1 /k t2` with positions | `within(index, t1, t2, k) -> set` |
| `analyzer.py` | C (+B) | ✅ wiring | Raw query → `AnalyzedQuery`, recording every rewrite in `.trace` | `QueryAnalyzer.load(cfg, vocabulary)`, `.analyze(query)`; steps: filters → language → translit → lexicon/phonetic → pipeline → code in force → cross-code |

### `rank/` — scoring (D; dense + fusion C)

| File | Owner | Status | Purpose | Must contain |
| --- | --- | --- | --- | --- |
| `vsm.py` | D | ⬜ | tf-idf, SMART lnc.ltc, cosine — classic baseline | `TfidfScorer(index).prepare()`, `.score(terms, candidates) -> {doc: score}` |
| `bm25f.py` | D | ⬜ | **Main lexical scorer**, zone-weighted BM25 | `BM25F.from_config(zidx, cfg, use_zones)`, `.idf(term)`, `.score(terms, candidates)` |
| `dense.py` | C | ⬜ | NLLB-E5 paragraph embeddings, MaxP doc score | `DenseRetriever.load(cfg)`, `.encode_corpus(docs, name)`, `.score(text, candidates)` |
| `statute_bridge.py` | D | ⬜ | Top statutes → expand + boost precedents (replaces IL-PCSR's LLM step) | `BridgeOutput(expansion, boosts, statutes_used)`, `StatuteBridge.run(statute_hits)` |
| `citation_graph.py` | D | 🔧 | Citation graph from **train** qrels + precedent citations | ⬜`build_graph()`, ⬜`jurisdiction_subgraphs()`, ✅`save_graph()/load_graph()` |
| `authority.py` | D | 🔧 | g(d) and g(d \| state) | ✅`binding_status(meta, state)`, ⬜`Authority.compute()`, ⬜`.score(doc, state, jurisdiction)`, ✅`to_dict/from_dict` |
| `qpp.py` | D | ⬜ | Query performance prediction (no labels) | `QPPFeatures`, `pre_retrieval()`, `post_retrieval()`, `alpha_from_qpp()`, `should_abstain()` |
| `fusion.py` | C | ✅ | Score normalisation + lexical/dense fusion | `minmax()`, `zscore()`, `fuse(lexical, dense, alpha, method)` |
| `topk.py` | D | ✅ | Heap top-K and net score | `top_k(scores, k)`, `net_score()`, `to_scored()` |

### `agent/` — layer 2 research agent (AG)

| File | Status | Purpose | Must contain |
| --- | --- | --- | --- |
| `plan.py` | 🔧 | One question → sub-queries; `original` always present, other rules stubbed | `SubQuery`, `Plan.add()`, `RulePlanner.plan(query, aq, idf)` with ⬜ rules `_cross_code`, `_boolean`, `_facet`, `_statutes`; ⬜ optional `LLMPlanner` |
| `executor.py` | ✅ | Run each sub-query through `SearchEngine.search` | `SubResult(.ranking(), .scores())`, `run_plan(engine, plan, base_options, depth)` |
| `fuse.py` | 🔧 | Merge ranked lists | ✅`reciprocal_rank_fusion(rankings, weights, k)`, ✅`fuse_subresults()`, ⬜`combsum()` |
| `reflect.py` | 🔧 | Retry decision + pseudo-relevance feedback | ✅ round guard, ⬜ QPP test in `should_reformulate()`, ⬜`reformulate()` |
| `research.py` | ✅ | The loop: plan → execute → fuse → reflect | `ResearchAgent.load(engine)`, `.run(query, options) -> AgentResult`, `AgentResult(statutes, precedents, plans, subresults, trace, n_searches)` |

### `rag/` — layer 3 grounded answer (R)

| File | Status | Purpose | Must contain |
| --- | --- | --- | --- |
| `answer.py` | ✅ wiring | Order of steps: abstain → chunk → generate → citation check → version check | `answer(result, docs, cfg, idf, resolver, normalizer) -> Answer` |
| `chunker.py` | ⬜ | Top statutes + ratio/decision paragraphs → numbered chunks | `Chunk`, `make_chunks(result, docs, max_chunks, max_chars)` |
| `generate.py` | ⬜ | LLM call with the citation-per-sentence prompt | `PROMPT`, `Answer(text, sentences, flags, abstained)`, `generate(question, chunks, incident_date, state)` |
| `citation_check.py` | ⬜ | tf-idf cosine of each sentence vs its cited chunk | `check(answer, chunks, idf, threshold)` |
| `version_check.py` | ⬜ | Wrong code for the incident date; bare colliding numbers | `check(answer, incident_date, resolver, normalizer)` |
| `abstain.py` | ⬜ | Refuse before generating when QPP says weak | `decide(result, idf) -> (bool, reason)` |

### `eval/` — measuring everything (D; agent_eval AG + R)

| File | Status | Purpose | Must contain |
| --- | --- | --- | --- |
| `metrics.py` | ✅ | P@k, R@k, F1@k, AP, MRR, nDCG; IL-PCSR macro-F1@k protocol | `evaluate(run, qrels, ks)`, `best_k()` (on **val**), per-query metric functions |
| `qrels.py` | ✅ | Load/save judgments; E6 binding/persuasive grades | `load_tsv()`, `load_by_judge()`, `save_tsv()`, `merge_judges()`, `jurisdiction_grades()` |
| `run_eval.py` | 🔧 | Run a system on a test set; TREC run files | ✅`write_run()/read_run()/run_queries()/main()`, ⬜`load_test_set(name)` |
| `ablation.py` | ⬜ | Ablation ladder + language ablation → CSV | `run_ladder(set)`, `run_language_ablation()` |
| `efficiency.py` | ⬜ | Exhaustive vs tiered vs champion lists | `compare_modes(n_queries)` |
| `agreement.py` | ✅ | Inter-judge agreement | `percent_agreement()`, `cohens_kappa()` |
| `agent_eval.py` | ⬜ | Layer 2 vs core; layer 3 checks | `compare_agent(set)`, `evaluate_rag(path)` |
| `plots.py` | ⬜ | Report/video charts | `bar_chart()`, `make_all()` |

---

## `scripts/` — run in order

| File | Owner | Status | Does |
| --- | --- | --- | --- |
| `00_check_access.py` | all | ✅ | PASS/MISSING for config, data, packages; TODOs left per owner |
| `01_build_corpus.py` | A | 🔧 | IL-PCSR + BNS → zones + metadata → `docs.jsonl`; prints missing court/date counts |
| `02_build_index.py` | A | 🔧 | Same text pipeline as queries → zone indexes, facets, statute terms |
| `03_build_graph.py` | D | 🔧 | Train qrels → graph → authority + tiers |
| `04_encode_dense.py` | C | 🔧 | Paragraph embeddings once (GPU) |
| `05_run_all_evals.py` | D | 🔧 | All test sets (baseline vs full), ablations, efficiency, figures |
| `judge_queries.py` | all | ✅ | Terminal judging tool, resumable, per-judge TSV |

## `app/` — demo (frontend is not graded)

| File | Status | Does |
| --- | --- | --- |
| `cli.py` | ✅ | `python app/cli.py "<query>" --state delhi --date 2025-03-01 [--agent] [--answer] [--baseline] [--debug]` — `--debug` prints the full trace and score breakdowns for the video |
| `streamlit_app.py` | ✅ thin | Query box, state, date, results, trace; RAG panel to add after the gate |

## `notebooks/`, `tests/`, `results/`, `docs/`

| Path | Does |
| --- | --- |
| `notebooks/01_explore_ilpcsr.ipynb` | Corpus statistics for the report |
| `notebooks/02_postings_demo.ipynb` | Postings, weights and "IPC 302 = BNS 103" live, for the video |
| `notebooks/03_results.ipynb` | Results walkthrough |
| `tests/conftest.py` | `@todo` marks a test as a spec: skipped while the code raises `NotImplementedError`, runs for real once implemented; `toy_docs` fixture |
| `tests/test_*.py` | One file per area: tokenize, version_norm, collision, phonetic, positional, boolean, bm25f, authority, metrics, agent, schema_and_pipeline |
| `results/` | `runs/` (TREC files), `tables/` (CSV), `figures/` (committed, used in report) |
| `docs/architecture.svg/.png` | The architecture diagram; regenerate with `python docs/make_architecture.py` |
| `docs/ai_use.md` | Running AI-use log → report declaration |
| `docs/report/` | Report source and PDF (≤ 8 pages) |

---

## Workload (TODO stubs left at scaffold time)

| Owner | Stubs | Main pieces |
| --- | --- | --- |
| A | 20 | IL-PCSR loader, segmentation, metadata, positional/zone/facet indexes, Boolean parser |
| B | 8 | crosswalk parsing, version normalisation, collision resolver |
| C | 10 | transliteration, phonetic, Hindi stemmer, dense channel |
| D | 21 | BM25F, tf-idf, statute bridge, graph, authority, QPP, tiers, evaluation |
| AG | 8 | planner rules, CombSUM, reflection |
| R | 7 | RAG steps + agent/RAG evaluation |

Suggested rebalancing (team decision): move `rank/qpp.py` to C (fusion uses it) and `index/tiers.py`
to B; B also owns the agent after finishing normalisation.

## Integration rules

1. **Agree on `schema.py` in hour 1.** Pass `Document`, `Query`, `ScoredDoc`, nothing ad hoc.
2. **One text pipeline** (`text/pipeline.analyze_text`) for documents and queries.
3. **No test leakage.** Graph and tuning use train/val only.
4. **Everything through `search.py`.** App, agent and evaluator call the same function.
5. **Keep the "Must contain" names.** Change a signature only after telling the team.
6. **Log AI use** in `docs/ai_use.md` as you go.
