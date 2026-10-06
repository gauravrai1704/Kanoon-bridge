# AI-use log

Add a row whenever an AI tool writes or substantially edits something. This becomes the
report's AI-use declaration (no deduction for AI use; it only has to be declared).

| Date | Member | Tool | Files / part | What it did |
| --- | --- | --- | --- | --- |
| 2026-10-06 | Gaurav | Claude | proposal, PROJECT_STRUCTURE.md, repo scaffold (all files' structure, docstrings and stubs; working: schema, config, store, tokenizer, text pipeline, analyzer wiring, search orchestration, metrics, qrels, agreement, scripts, app wrappers, tests) | Brainstormed the idea, drafted the proposal, generated the scaffold |
| 2026-10-06 | Gaurav | Claude | agent/ (layer 2), rag/answer.py, index/docstore.py, eval/agent_eval.py, ARCHITECTURE.md, docs/architecture.svg + generator, PROJECT_STRUCTURE.md rewrite | Added the research-agent layer scaffold and the architecture/structure guides |
| 2026-10-06 | Gaurav | Claude | ingest/ (IL-PCSR + BNS loaders, segmentation, metadata), ingest/parse_crosswalk.py, text/version_norm.py, text/collision.py, rank/ (vsm, bm25f, statute_bridge, citation_graph, authority, qpp), index/tiers.py, scripts 00_fetch_data/01/03, tokenizer code-first fix, tests, synthetic sample | Implemented Gaurav's offline-build and Layer 1 components |
| 2026-10-06 | Gaurav | Claude | eval/ (run_eval test-set loading, ablation, efficiency, plots, agent_eval), rag/ (abstain, chunker, generate, citation_check, version_check, answer), PDF cross-check parser hardening, scripts/05_run_all_evals.py, SETUP_AND_RUN.md, tests | Implemented the evaluation harness and the RAG layer |
| (runtime) | — | Claude API (claude-haiku-4-5-20251001 by default) | rag/generate.py | Generates the layer-3 answers at run time from retrieved sources; declare in the report |
| 2026-10-06 | Gaurav | Claude | agent/ (planner rules, CombSUM, reflection + pseudo-relevance feedback, LLM/hybrid planners, AgentResult), SearchOptions agent hooks + SearchEngine.from_components, RAG coverage abstention, English stop-word list, app/CLI agent wiring, test_layers_integration.py | Completed layer 2 and checked the hand-offs between all layers |
| 2026-10-06 | Gaurav | Claude | Sharanya's part: index/positional.py, index/zones.py, index/facets.py, query/parser.py (recursive descent), query/boolean.py, query/proximity.py, Boolean wiring in search.py; Kashvi's part: text/transliterate.py (Devanagari table, normalisation rules, lexicon lookup), text/phonetic.py, text/stem.py (Hindi), rank/dense.py, lexicon expanded to ~165 rows, analyzer phonetic step; tests | Implemented the remaining stubs at Gaurav's request; owners to review |
