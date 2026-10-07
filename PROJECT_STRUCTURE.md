# Kanoon-Bridge — Project Structure

Read `ARCHITECTURE.md` first for how the pieces fit. This file is the map: every file, who owns
it, whether it works yet, what it is for, and **what it must contain** (the functions and classes
other code calls — keep these names and signatures, or tell the team).

**Owners:** **Gaurav** ingest, version normalisation (was B), ranking + evaluation (was D) ·
**Sharanya** (A) positional/zone/facet indexes, query parser, tokenizer · **Kashvi** (C) Hindi/Hinglish + dense ·
**Shaurya** (AG) research agent, layer 2 · **Gaurav** RAG, layer 3.

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
│   ├── schema.py  config.py  search.py  present.py
│   ├── ingest/                   load_ilpcsr  load_statutes  parse_crosswalk  segment  metadata
│   ├── text/                     pipeline  tokenize  stem  collision  version_norm  transliterate  phonetic  spell
│   ├── index/                    positional  zones  facets  tiers  store  docstore  dedup
│   ├── query/                    parser  boolean  proximity  analyzer
│   ├── rank/                     vsm  bm25f  dense  statute_bridge  citation_graph  authority  qpp  fusion  topk  ltr  similar  feedback
│   ├── agent/        (layer 2)   plan  executor  fuse  reflect  research
│   ├── rag/          (layer 3)   answer  chunker  generate  citation_check  version_check  abstain
│   └── eval/                     metrics  qrels  run_eval  ablation  efficiency  agreement  agent_eval  typos  plots
├── scripts/                      00_check_access … 05_run_all_evals, 06_train_ltr, judge_queries
├── app/                          web_server.py  web/ (index.html styles.css app.js art.js fonts/)  cli.py  streamlit_app.py
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
| `search.py` | D | ✅ | **Layer 1 entry point.** The CLI, app, agent and evaluator all call it (rule 4) | `SearchOptions` (one switch per component; `.full()`, `.interactive()`, `.baseline()`, `.from_dict()`; agent hooks `restrict_states/require_terms/extra_terms/drop_terms`; `ltr`, `collapse_duplicates`), `SearchEngine.load()/from_components()`, `.search(query, options) -> SearchResult`, `.boolean_candidates()`, module-level `search()` |
| `present.py` | Gaurav | ✅ | Query-biased snippets + highlighting, "why this result", statute version notes, facet counts (CLI/app only; never affects ranking) | `snippet(doc, aq, res)`, `highlight()`, `explain(hit, aq, engine)`, `version_note(ref, normalizer)`, `facet_counts()`, `title_of()` |

### `ingest/` — raw data → `Document`s (A; crosswalk B)

| File | Owner | Status | Purpose | Must contain |
| --- | --- | --- | --- | --- |
| `load_ilpcsr.py` | Gaurav | ✅ | Read IL-PCSR queries, precedents, statutes, qrels. First job in hour 0: `inspect()` the real files | `parse_provision()`, `rows()`, `inspect()`, `load_queries(split)`, `load_precedents()`, `load_statute_candidates()`, `load_qrels(split, target) -> {qid: {doc_id: 1}}`, `iter_all()` — reads the parquet/jsonl export, falls back to the HF hub |
| `load_statutes.py` | Gaurav | ✅ | BNS bare act → one `Document` per section | `parse_bns(path)` (JSON dir or plain text), `parse_bns_json()`, `parse_bns_text()`, `iter_section_records()` (doc_id like `bns:103`) |
| `parse_crosswalk.py` | Gaurav | ✅ | BNS sections' IPC correspondences → the two crosswalk CSVs (government PDF as optional cross-check) | `parse_ipc_reference()`, `extract_rows_from_bns_json()`, `extract_rows_from_pdf()` (best-effort), `compare_sources()`, `write_crosswalk()`, `build_offences()`/`build_offence_ids()`; `python -m kanoon_bridge.ingest.parse_crosswalk [--compare]` |
| `segment.py` | Gaurav | ✅ | Judgments → zone paragraphs ("what is a document") | `ROLE_TO_ZONE` (real IL-PCSR labels), `CUE_PHRASES`, `split_paragraphs()`, `zone_from_cues()`, `segment_judgment(text_or_paragraphs, roles)`, `segment_statute()` |
| `metadata.py` | Gaurav | ✅ | Court, states, decision date, code in force — the spatio-temporal facets | `CourtTable.load()/lookup()` (all 25 High Courts), `find_dates()`, `find_date()`, `code_in_force()`, `enrich(doc, table)` |

### `text/` — analysis and normalisation (A, B, C)

| File | Owner | Status | Purpose | Must contain |
| --- | --- | --- | --- | --- |
| `pipeline.py` | all | ✅ | **The one text pipeline** for documents and queries (rule 2). Skips unfinished steps with a warning | `TextOptions`, `TextResources.load()`, `analyze_text(text, resources, lang, date, options) -> list[str]` |
| `tokenize.py` | A | ✅ | Section-aware tokeniser: "u/s 498A IPC" → `sec:ipc:498a`; bare numbers → `sec:?:302` | `tokenize()`, `extract_sections() -> list[SectionMention]`, `is_section_token()`, `SECTION_PREFIX` |
| `stem.py` | C | ✅ | Porter + Hindi light stemmer (Ramanathan & Rao suffix list, min stem 2); never stems `sec:`/`off:` tokens | `stem_tokens(tokens, lang, enabled)`, `stem_english()`, `stem_hindi()` |
| `version_norm.py` | Gaurav | ✅ | **Core novelty.** Section tokens → canonical offence IDs, both codes | `VersionNormalizer.load()`, `to_offences(ref) -> [(off_id, weight)]`, `offences_for()`, `sections_of()`, `equivalents(ref)`, `normalize_tokens(tokens)`, `label()` |
| `collision.py` | Gaurav | ✅ | Resolve bare/colliding numbers from date + context. **Hardest part** | `Reading`, `CollisionResolver.load()`, `code_for_date()`, `resolve(section, context, date)`, `resolve_tokens(tokens, date)`, `last_readings` (trace) |
| `transliterate.py` | C | ✅ | Language detection, Devanagari→Roman (own character table, schwa deletion), Hinglish spelling rules, legal lexicon (exact → normalised → phonetic) | `detect_lang()`, `devanagari_to_roman()`, `normalize_roman()` + `NORMALISATION_RULES`, `LegalLexicon.load()`, `.lookup(word, use_phonetic)`, `FUNCTION_WORDS` |
| `phonetic.py` | C | ✅ | Soundex for Roman Hindi (`hatya = hathya = hattya`) | `soundex()` (baseline), `hindi_soundex()`, `PhoneticIndex.build()/matches()` |
| `spell.py` | Gaurav | ✅ | Tolerant retrieval: k-gram index, Damerau–Levenshtein spelling correction, did-you-mean, wildcards (IIR ch. 3) | `KGramIndex(.candidates, .wildcard)`, `damerau_levenshtein()`, `Speller.build(df, surface).check(word, term) -> Correction`, `.expand_wildcard()`, `surface_forms(docs)` |

### `index/` — data structures (A; tiers D)

| File | Owner | Status | Purpose | Must contain |
| --- | --- | --- | --- | --- |
| `positional.py` | A | ✅ | Inverted index with positions (compact `array('I')`), or counts only (`positions=False`, used for zones) | `PositionalIndex`: `add(doc_id, tokens)`, `phrase(terms)` (rarest-first positional intersection); stats `df/idf/tf/postings_for/docs_with/n_docs/avg_doc_len/vocabulary` |
| `zones.py` | A | ✅ | One positional index per zone + whole-doc index; feeds BM25F | `ZoneIndex`: `build(docs, analyze)`, `tf(term, doc, zone)`, `zone_len()`, `avg_zone_len()`, `df()`, `.whole` |
| `facets.py` | A | ✅ | Parametric index: type, court, states, date, code, sections cited | `DocMeta`, `FacetIndex.build(docs)`, `.filter(doc_type, court, states, date_from, date_to, code, cites_any) -> set` (smallest set first), `.date_range()` (bisect), `.meta(doc_id)`, `.by_section` |
| `tiers.py` | Gaurav | ✅ | Tiered index + champion lists (efficiency experiment) | `TieredIndex.build(zidx, authority, quantile, champion_size)`, `.candidates(terms, min_results, use_champions, index)` |
| `store.py` | A | ✅ | Save/load artefacts under `data/processed/index/` | `save(obj, name, fmt)`, `load(name, fmt)`, `exists()`, `index_dir()` |
| `docstore.py` | all | ✅ | doc_id → `Document` text (agent, RAG, judging tool, app) | `DocStore.load()`, mapping access, `.snippet()` |
| `dedup.py` | Gaurav | ✅ | Near-duplicate judgments: 5-word shingles, MinHash, LSH bands, union-find groups | `shingles()`, `minhash()`, `near_duplicate_groups(docs) -> {doc: group}` |

### `query/` — understanding input (A; analyzer C)

| File | Owner | Status | Purpose | Must contain |
| --- | --- | --- | --- | --- |
| `parser.py` | A | ✅ | Query syntax: phrases, AND/OR/NOT, `/k`, filters `state: date: code: court:`; section mentions stay one word | `extract_filters()`, `has_operators()`, `parse() -> ParsedQuery(.tree, .text_for_ranking)`, `QueryNode`/`Op`, recursive-descent `_parse_tree()`, `ranking_text()`, `show()` |
| `boolean.py` | A | ✅ | Boolean retrieval; intersection in increasing-df order; section terms match their offence in either code | `intersect(p1, p2)` (two-pointer), `intersect_many(lists)`, `evaluate(node, index, analyze, universe)` |
| `proximity.py` | A | ✅ | `t1 /k t2` with positions (two-pointer) | `within(index, t1, t2, k) -> set`, `within_positions()`, `positions_within()` |
| `analyzer.py` | C (+B) | ✅ | Raw query → `AnalyzedQuery`, recording every rewrite in `.trace` | `QueryAnalyzer.load(cfg, vocabulary)`, `.analyze(query)`; steps: filters (incl. `after:`/`before:`) → wildcards → language → translit → lexicon/phonetic → pipeline → spelling → code in force → cross-code; `.steps` switches for ablations |

### `rank/` — scoring (D; dense + fusion C)

| File | Owner | Status | Purpose | Must contain |
| --- | --- | --- | --- | --- |
| `vsm.py` | Gaurav | ✅ | tf-idf, SMART lnc.ltc, cosine — classic baseline | `TfidfScorer(index).prepare()`, `.score(terms, candidates) -> {doc: score}` |
| `bm25f.py` | Gaurav | ✅ | **Main lexical scorer**, zone-weighted BM25 | `BM25F.from_config(zidx, cfg, use_zones)`, `.idf(term)`, `.score(terms, candidates)` |
| `dense.py` | C | ✅ | NLLB-E5 paragraph embeddings (fallback multilingual-e5-base), E5 prefixes, MaxP doc score | `DenseRetriever.load(cfg, name, model)`, `.encode_corpus(docs, name)`, `.score(text, candidates)` |
| `statute_bridge.py` | Gaurav | ✅ | Top statutes → expand + boost precedents (replaces IL-PCSR's LLM step) | `BridgeOutput(expansion, boosts, statutes_used)`, `StatuteBridge(facets, statute_terms, top_n, boost, normalizer).run(statute_hits)` — reaches IPC-citing precedents from BNS statutes via offence ids |
| `citation_graph.py` | Gaurav | ✅ | Citation graph from **train** qrels + precedent citations | `build_graph()` (leakage guard), `jurisdiction_subgraphs()`, `stats()`, `save_graph()/load_graph()` |
| `authority.py` | Gaurav | ✅ | g(d) and g(d \| state) | `binding_status(meta, state)`, `pagerank()` (own power iteration), `Authority.compute()`, `.score(doc, state, jurisdiction)`, `.top()`, `to_dict/from_dict` |
| `qpp.py` | Gaurav | ✅ | Query performance prediction (no labels) | `QPPFeatures`, `pre_retrieval()`, `post_retrieval()`, `confidence()`, `alpha_from_qpp()`, `should_abstain()` |
| `fusion.py` | C | ✅ | Score normalisation + lexical/dense fusion | `minmax()`, `zscore()`, `fuse(lexical, dense, alpha, method)` |
| `topk.py` | D | ✅ | Heap top-K and net score | `top_k(scores, k)`, `net_score()`, `to_scored()` |
| `ltr.py` | Gaurav | ✅ | Learning to rank: 13 features, linear model, coordinate ascent on MAP (Metzler & Croft 2007), trained on val | `FEATURES`, `feature_rows(hits, aq, engine)`, `coordinate_ascent()`, `cross_validate()`, `LinearRanker` |
| `similar.py` | Gaurav | ✅ | "More like this": text profile + version-aware bibliographic coupling + co-citation | `SimilarCases.load(engine, docs).find(doc_id, k)` |
| `feedback.py` | Gaurav | ✅ | Explicit relevance feedback (Rocchio) → `SearchOptions.extra_terms/drop_terms` | `rocchio_options(engine, aq, docs, relevant, non_relevant)` |

### `agent/` — layer 2 research agent (AG: Shaurya)

| File | Status | Purpose | Must contain |
| --- | --- | --- | --- |
| `plan.py` | ✅ | One question → sub-queries: original, cross-code, Boolean (CNF over postings), facet (binding courts), statutes-first; optional LLM / hybrid planners | `SubQuery`, `Plan.add()`, `RulePlanner.plan(query, aq, idf)`, `LLMPlanner`, `HybridPlanner`, `make_planner(cfg, normalizer, df)`, `parse_llm_queries()` |
| `executor.py` | ✅ | Run each sub-query through `SearchEngine.search` with its `SearchOptions` overrides | `SubResult(.ranking(), .scores())`, `run_plan(engine, plan, base_options, depth)` |
| `fuse.py` | ✅ | Merge ranked lists | `reciprocal_rank_fusion()`, `fuse_subresults()`, `combsum()`, `fuse(subresults, target, method, k)` |
| `reflect.py` | ✅ | QPP retry decision + Rocchio-style pseudo-relevance feedback | `should_reformulate(fused, aq, idf, round, max_rounds, core_scores)`, `reformulate(aq, fused, docs, plan, ..., idf, df, analyze)`, `feedback_terms()` |
| `research.py` | ✅ | The loop: plan → execute → fuse → reflect → (PRF round) → fuse | `ResearchAgent.load(engine, docs, fusion, planner)`, `.run(query, options) -> AgentResult`; `AgentResult(statutes, precedents, query, analyzed, plans, subresults, trace, timings_ms, n_searches, found_by())` |

Layer-1 hooks the agent uses (`SearchOptions`): `restrict_states` (facet), `require_terms`
(Boolean CNF, executed by `SearchEngine.boolean_candidates`), `extra_terms` / `drop_terms` (PRF).
All default to off, so plain searches are unchanged.

### `rag/` — layer 3 grounded answer (Gaurav)

| File | Status | Purpose | Must contain |
| --- | --- | --- | --- |
| `answer.py` | ✅ | Order of steps: abstain → chunk → generate → citation check → version check | `answer(result, docs, cfg, idf, resolver, normalizer, generator, closed_book) -> Answer`, `RagPipeline.load(engine).answer(result)`, `render(ans)` |
| `chunker.py` | ✅ | Top statutes ("BNS Section 103 - ...") + best ratio/decision paragraph per top precedent → numbered chunks | `Chunk`, `make_chunks(result, docs, max_chunks, max_chars, query_terms)`, `truncate()` |
| `generate.py` | ✅ | Claude (temperature 0, one `[n]` per sentence) or offline extractive; closed-book baseline | `PROMPT`, `Answer`, `generate(question, chunks, incident_date, state, generator, model, closed_book)`, `load_env()` |
| `citation_check.py` | ✅ | tf-idf cosine of each sentence vs its cited chunk (any chunk for closed-book) | `check(answer, chunks, idf, threshold, any_chunk)` |
| `version_check.py` | ✅ | Wrong code for the incident date (+ equivalent); bare colliding numbers (+ readings) | `check(answer, incident_date, resolver, normalizer)` |
| `abstain.py` | ✅ | Refuse before generating when QPP says weak | `decide(result, idf, query_terms) -> (bool, reason)` |
| `_util.py` | ✅ | Sentence split (abbreviation-safe), citations, tf-idf/cosine helpers | |

### `eval/` — measuring everything (Gaurav; agent internals Shaurya)

| File | Status | Purpose | Must contain |
| --- | --- | --- | --- |
| `metrics.py` | ✅ | P@k, R@k, F1@k, AP, MRR, nDCG; IL-PCSR macro-F1@k protocol | `evaluate(run, qrels, ks)`, `best_k()` (on **val**), per-query metric functions |
| `qrels.py` | ✅ | Load/save judgments; E6 binding/persuasive grades | `load_tsv()`, `load_by_judge()`, `save_tsv()`, `merge_judges()`, `jurisdiction_grades()` |
| `run_eval.py` | ✅ | Load test sets (E1 judgments as queries, 100-term cap, k from val), run, score, set-specific metrics | `load_test_set(name) -> TestSet`, `evaluate_set()`, `extra_metrics()` (wrong_hit@10, P@5 per language, binding_share@5, code_accuracy@1), TREC run files |
| `ablation.py` | ✅ | Ablation ladder + language ablation → CSV | `run_ladder(set)`, `run_language_ablation()`, `write_table()` |
| `efficiency.py` | ✅ | Exhaustive vs tiered vs champion lists: latency + Recall@20 vs exhaustive | `compare_modes(n_queries)` |
| `agreement.py` | ✅ | Inter-judge agreement | `percent_agreement()`, `cohens_kappa()` |
| `agent_eval.py` | ✅ | Layer 2 vs core; layer 3 support rate, version flags, abstention | `compare_agent(set)`, `evaluate_rag(path)` |
| `significance.py` | ✅ | Paired randomization test, bootstrap CI, Holm correction over run files (Smucker et al. CIKM 2007) | `per_query()`, `randomization_test()`, `bootstrap_ci()`, `holm()`, `compare()`, `run_all()` |
| `typos.py` | ✅ | Typo robustness: simulated known-item statute queries with Damerau errors; spelling off vs on | `corrupt()`, `run()` → `typos.csv`, `typo_queries.jsonl` |
| `plots.py` | ✅ | Report/video charts from results/tables | `bar_chart()`, `make_all()` |

---

## `scripts/` — run in order

| File | Owner | Status | Does |
| --- | --- | --- | --- |
| `00_fetch_data.py` | Gaurav | ✅ | Clones the BNS source; exports IL-PCSR from Hugging Face to parquet (needs `huggingface-cli login`) |
| `00_check_access.py` | all | ✅ | PASS/MISSING for config, data, packages; TODOs left per owner |
| `01_build_corpus.py` | Gaurav | ✅ | IL-PCSR + BNS → zones + metadata → `docs.jsonl`; prints missing court/date counts |
| `02_build_index.py` | A | ✅ | Same text pipeline as queries → zone indexes, facets, statute terms |
| `03_build_graph.py` | Gaurav | ✅ | Train qrels → graph → authority + tiers |
| `04_encode_dense.py` | C | ✅ | Paragraph embeddings once (GPU) |
| `05_run_all_evals.py` | D | ✅ | All test sets (baseline vs full), ablations, efficiency, typos, agent, RAG, figures |
| `07_make_test_sets.py` | Gaurav | ✅ | Generate E2 (collisions), E3 (+ control; cross-version), E7 (temporal) with gold answers from the crosswalk and IL-PCSR |
| `merge_judgments.py` | all | ✅ | Agreement + kappa between judges, disagreement list, merged qrels |
| `06_train_ltr.py` | Gaurav | ✅ | Train the learning-to-rank model on IL-PCSR val (5-fold CV report) → `index/ltr.json` |
| `judge_queries.py` | all | ✅ | Terminal judging tool, resumable, per-judge TSV |

## `app/` — demo (frontend is not graded)

| File | Status | Does |
| --- | --- | --- |
| `cli.py` | ✅ | `python app/cli.py "<query>" --state delhi --date 2025-03-01 [--agent [--fusion rrf|combsum] [--planner rules|llm|hybrid]] [--answer [--generator ...]] [--baseline] [--debug]` — `--debug` prints the full trace and score breakdowns for the video |
| `web_server.py` | ✅ | **Main UI.** Standard-library JSON API (`/api/search`, `/api/similar`, `/api/doc`, `/api/health`) + static files; builds the step-by-step pipeline view from the traces | `Backend(engine, docs).search(body)`, `Handler`, `main()` |
| `web/` | ✅ | Single-page app: bridge hero, advocate caricatures (`art.js`), history sidebar, results, "How this was found" drawer, reader; bundled OFL fonts | `index.html`, `styles.css`, `app.js`, `art.js`, `fonts/` |
| `streamlit_app.py` | ✅ thin | Query box, state, date, results, trace; RAG panel to add after the gate |

## `notebooks/`, `tests/`, `results/`, `docs/`

| Path | Does |
| --- | --- |
| `notebooks/01_explore_ilpcsr.ipynb` | Corpus statistics for the report |
| `notebooks/02_postings_demo.ipynb` | Postings, weights and "IPC 302 = BNS 103" live, for the video |
| `notebooks/03_results.ipynb` | Results walkthrough |
| `tests/data/make_ilpcsr_sample.py` | Writes the synthetic IL-PCSR-schema sample used by `make sample` and the ingest tests (fictional cases) |
| `tests/conftest.py` | `@todo` marks a test as a spec: skipped while the code raises `NotImplementedError`, runs for real once implemented; `toy_docs` fixture |
| `tests/test_*.py` | One file per area: tokenize, version_norm, collision, phonetic, stem, dense, positional, boolean, bm25f, authority, metrics, agent, rag, eval, crosswalk_pdf, schema_and_pipeline; `test_layers_integration.py` runs layers 1-3 + the evaluator together on an in-memory corpus |
| `results/` | `runs/` (TREC files), `tables/` (CSV), `figures/` (committed, used in report) |
| `docs/architecture.svg/.png` | The architecture diagram; regenerate with `python docs/make_architecture.py` |
| `docs/ai_use.md` | Running AI-use log → report declaration |
| `docs/report/` | Report source and PDF (≤ 8 pages) |

---

## Workload (TODO stubs left)

| Owner | Left | Main pieces |
| --- | --- | --- |
| Gaurav | 6 (evaluation only) | run_eval test-set loading, ablations, efficiency, plots — ingest, normalisation and all Layer 1 ranking are done |
| Sharanya | 10 | positional/zone/facet index build + filter, phrase, Boolean parser, intersection, proximity |
| Kashvi | 10 | transliteration, Roman-Hindi normalisation, lexicon lookup, Hindi soundex, Hindi stemmer, dense channel |
| Shaurya | 8 | planner rules, CombSUM, reflection (QPP retry + pseudo-relevance feedback) |
| RAG | 7 | RAG steps + agent/RAG evaluation |

**Sharanya's index build is now the critical path**: `scripts/02_build_index.py` (and so search) needs
`PositionalIndex.add`, `ZoneIndex.build` and `FacetIndex.build/filter`. Everything downstream of it is built.

## Integration rules

1. **Agree on `schema.py` in hour 1.** Pass `Document`, `Query`, `ScoredDoc`, nothing ad hoc.
2. **One text pipeline** (`text/pipeline.analyze_text`) for documents and queries.
3. **No test leakage.** Graph and tuning use train/val only.
4. **Everything through `search.py`.** App, agent and evaluator call the same function.
5. **Keep the "Must contain" names.** Change a signature only after telling the team.
6. **Log AI use** in `docs/ai_use.md` as you go.
