# AI-use log

Add a row whenever an AI tool writes or substantially edits something. This becomes the
report's AI-use declaration (no deduction for AI use; it only has to be declared).

| Date | Member | Tool | Files / part | What it did |
| --- | --- | --- | --- | --- |
| 2026-10-06 | Gaurav | Claude | proposal, PROJECT_STRUCTURE.md, repo scaffold (all files' structure, docstrings and stubs; working: schema, config, store, tokenizer, text pipeline, analyzer wiring, search orchestration, metrics, qrels, agreement, scripts, app wrappers, tests) | Brainstormed the idea, drafted the proposal, generated the scaffold |
| 2026-10-06 | Gaurav | Claude | agent/ (layer 2), rag/answer.py, index/docstore.py, eval/agent_eval.py, ARCHITECTURE.md, docs/architecture.svg + generator, PROJECT_STRUCTURE.md rewrite | Added the research-agent layer scaffold and the architecture/structure guides |
| 2026-10-06 | Gaurav | Claude | ingest/ (IL-PCSR + BNS loaders, segmentation, metadata), ingest/parse_crosswalk.py, text/version_norm.py, text/collision.py, rank/ (vsm, bm25f, statute_bridge, citation_graph, authority, qpp), index/tiers.py, scripts 00_fetch_data/01/03, tokenizer code-first fix, tests, synthetic sample | Implemented Gaurav's offline-build and Layer 1 components |
