# Kanoon-Bridge — Architecture

![Kanoon-Bridge architecture](docs/architecture.svg)

*Regenerate the diagram with `python docs/make_architecture.py` (edit the content lists in that file).*

Kanoon-Bridge has **one core retriever and two optional layers on top of it**:

| Layer | Package | Entry point | Required? | What it adds |
| --- | --- | --- | --- | --- |
| 1 Core retriever | `search.py` + `text/ index/ query/ rank/` | `SearchEngine.search(query)` | **Yes** — the graded system | Version-aware, Hinglish-aware, jurisdiction-aware ranking of statutes and precedents |
| 2 Research agent | `agent/` | `ResearchAgent.run(query)` | Stretch, from the hour-26 gate | Several IR sub-queries per question, fused with RRF, one reformulation round if retrieval is weak |
| 3 Grounded answer | `rag/` | `rag.answer.answer(result, docs)` | Stretch, only if layer 2 is done by ~hour 30 | A short cited answer, checked sentence by sentence; abstains when grounding is weak |

Each layer only talks to the one below it through a public function, so layers 2 and 3 can be
switched off without touching layer 1 — and the core is a complete submission on its own.

---

## 1. Data sources and how each one is used

| Data | What it is | Lands in | Used by | Rule |
| --- | --- | --- | --- | --- |
| **IL-PCSR queries** (IIT Kgp + IIT Kanpur, 2025) | 6,271 Indian judgments; citations masked | `docs.jsonl` as `QUERY_CASE` | queries for E1/E3; train qrels build the graph | split 8:1:1, see below |
| **IL-PCSR precedents** | 3,183 SC + HC judgments (unmasked) | `docs.jsonl` as `PRECEDENT` → precedent zone index | layer 1 ranking, citation graph, RAG chunks | indexed |
| **IL-PCSR statutes** | 936 IPC-era sections | `docs.jsonl` as `STATUTE` → statute index | layer 1 statute ranking, statute bridge | indexed |
| **IL-PCSR qrels** | which precedents/statutes each query cited | loaded by `ingest/load_ilpcsr.load_qrels` | train → graph; val → tuning; test → E1/E3 | never tune on test |
| **BNS bare act** (via bns-study-platform, verified against the Gazette) | 358 sections, in force from 1 July 2024 | `docs.jsonl` as `STATUTE` (code BNS) | BNS-era statute retrieval | indexed |
| **IPC↔BNS crosswalk** (each BNS section's IPC correspondence; government table as cross-check) | 496 section pairs → ~350 offences | `data/crosswalk/ipc_bns.csv`, `offence_ids.csv` | version normalisation, collision resolver, statute bridge, RAG version check | spot-check key sections |
| **Lexicons** (ours) | Hinglish legal terms, stop words, court→states | `data/lexicons/` | analyzer expansion, text pipeline, metadata | committed |
| **Hand-built query sets** (ours) | E2, E3, E4, E6, E7 queries + 2-judge qrels | `data/queries/` | evaluation only | written **before** rules are final |
| PoliceDrishti (optional, gated) | 190 case summaries → BNS charges | `data/raw/policedrishti/` | E5 facts → sections | eval only |
| NLLB-E5 weights (optional) | zero-shot multilingual encoder | downloaded by `scripts/04_encode_dense.py` | dense channel | — |

### How the IL-PCSR splits are used (team rule 3: no test leakage)

| Split | Size | Used for | Never used for |
| --- | --- | --- | --- |
| train | 5,017 queries | citation-graph edges → authority g(d) | evaluation |
| val | 627 queries | tuning λ (authority), α (fusion), zone weights, choosing k for F1@k | reported results |
| test | 627 queries | E1 and E3 results in the report | tuning anything, graph edges |
| hand-built sets | ~30–100 each | E2, E4, E6, E7 | rule design (write them first) |

---

## 2. Offline build (run once, in order)

| Step | Script | Input | Output in `data/processed/` | Owner |
| --- | --- | --- | --- | --- |
| 01 build corpus | `scripts/01_build_corpus.py` | IL-PCSR, BNS text, court table | `docs.jsonl` — every document in one schema, with zones, court, states, decision date, code in force | A |
| 02 build index | `scripts/02_build_index.py` | `docs.jsonl` | `index/statutes_zone.pkl`, `index/precedents_zone.pkl`, `index/facets.pkl`, `index/statute_terms.json` | A |
| 03 build graph | `scripts/03_build_graph.py` | `docs.jsonl`, **train** qrels, facets | `citation_graph.json`, `index/authority.json`, `index/tiers.pkl` | D |
| 04 encode dense | `scripts/04_encode_dense.py` | `docs.jsonl` | `embeddings/*.npy` + ids (optional, GPU) | C |

Step 02 runs every document through `text/pipeline.analyze_text` — **the same function queries use**:

```
"convicted u/s 302 IPC for stabbing"
  tokenize       -> convicted  sec:ipc:302  for  stabbing
  stop words     -> convicted  sec:ipc:302  stabbing
  stem           -> convict    sec:ipc:302  stab
  collision      -> (bare numbers like sec:?:302 resolved to ipc/bns here)
  version norm   -> convict    sec:ipc:302  off:murder  stab
```

Because `off:murder` is indexed, a later query that says "BNS 103" (which also normalises to
`off:murder`) matches this 1990s judgment.

---

## 3. Online: one question through all three layers

Worked example (values are illustrative):
**"mere bhai ko chaku maara"**, state = Delhi, incident date = 2025-03-01.

### Layer 1 — core retriever (`SearchEngine.search`)

| Step | Module | What happens in the example |
| --- | --- | --- |
| Analyze | `query/analyzer.py` | language = Hinglish; spelling normalised; lexicon adds *knife, hurt*; code in force = **BNS** (date ≥ 1 Jul 2024); trace recorded |
| Rank statutes | `rank/bm25f.py` on the statute index, filtered by `index/facets.py` to code = BNS | e.g. BNS sections on hurt with a dangerous weapon, BNS 109 (attempt to murder) |
| Statute bridge | `rank/statute_bridge.py` | adds `sec:bns:109`, `off:attempt_to_murder` to the precedent query; boosts precedents citing those offences (in either code, via offence IDs) |
| Rank precedents | `rank/bm25f.py` (+ `rank/dense.py`, `rank/fusion.py` with α from `rank/qpp.py`) | IPC-era stabbing cases now match through `off:*` tokens |
| Authority + top-K | `rank/authority.py`, `rank/topk.py` | net = relevance + λ·g(d \| delhi): SC and Delhi HC cases get full weight, other HCs 0.4 |
| Output | `schema.SearchResult` | statutes, precedents, per-document score breakdown, trace, timings |

### Layer 2 — research agent (`ResearchAgent.run`)

| Step | Module | What happens |
| --- | --- | --- |
| Plan | `agent/plan.py` (`RulePlanner`) | q0 original · q1 cross-code ("IPC 307 knife injury") · q2 Boolean ("(knife OR chaku) AND (injury OR hurt)") · q3 facet (binding courts for Delhi only) · q4 statutes-first |
| Execute | `agent/executor.py` | each sub-query = one layer-1 search (same engine, rule 4) |
| Fuse | `agent/fuse.py` | reciprocal rank fusion: Σ w / (60 + rank) |
| Reflect | `agent/reflect.py` | QPP on the fused list; if weak, pseudo-relevance feedback adds top-document terms and runs one more round (max 2) |
| Output | `agent.research.AgentResult` | fused statutes + precedents, every sub-query and its results in the trace |

With only the `original` sub-query the agent returns exactly the layer-1 ranking — a built-in sanity check.

### Layer 3 — grounded answer (`rag.answer.answer`)

| Step | Module | What happens |
| --- | --- | --- |
| Abstain? | `rag/abstain.py` (+ `rank/qpp.py`) | weak retrieval → "not enough grounding", no LLM call |
| Chunk | `rag/chunker.py` | top statutes + ratio/decision paragraphs of top precedents, numbered [1]..[8] |
| Generate | `rag/generate.py` | LLM, temperature 0, 3–5 sentences, one [n] citation each |
| Citation check | `rag/citation_check.py` | tf-idf cosine of each sentence vs its chunk; < 0.2 → flagged |
| Version check | `rag/version_check.py` | flags IPC cited for a 2025 incident, or a bare "302" |
| Output | `rag.generate.Answer` | text + flags + sources; "not legal advice" |

### What each layer passes to the next

```
Query ──► Layer 1 ──► SearchResult(statutes, precedents, query.trace)
Query ──► Layer 2 ──(many Queries)──► Layer 1 ──► AgentResult(statutes, precedents, trace)
SearchResult | AgentResult ──► Layer 3 ──► Answer(text, flags, abstained)
```

Both `SearchResult` and `AgentResult` expose `.statutes` and `.precedents`, so layer 3 works on either.

---

## 4. Evaluation (owner D, `scripts/05_run_all_evals.py`)

| Test | Data | Main metric | Claim |
| --- | --- | --- | --- |
| E1 | IL-PCSR test (627) | macro-F1@k (k chosen on val), MAP, MRR | backend matches published baselines; each component adds (ablation ladder) |
| E2 | ~50 colliding-number queries | wrong-offence rate in top-10 | version normalisation removes wrong-offence hits |
| E3 | ~100 IL-PCSR queries rewritten in BNS terms | Recall@20, MAP | BNS queries reach IPC-era precedent |
| E4 | 30 needs × en / hi / Hinglish | P@5 per language | Hinglish and Hindi approach English quality |
| E5 (optional) | PoliceDrishti | P@5, R@10 | layperson facts find the right sections |
| E6 | ~30 queries × several states | binding share in top-5, nDCG@10 | g(d \| state) lifts binding precedent |
| E7 | ~30 incidents dated either side of 1 Jul 2024 | code accuracy of the top statute | the date facet picks the code in force |
| Layer 2 | E1, E6 | MAP, nDCG; searches per question | agent beats the single query (RRF vs CombSUM) |
| Layer 3 | ~20 questions | supported-sentence rate, version errors per answer | retrieval + checks reduce unsupported and wrong-code claims |
| Efficiency | E1 | latency, Recall@20 | tiers / champion lists trade little recall for speed |

---

## 5. Gates

| Hour | Gate | If not met |
| --- | --- | --- |
| 0 | data access works (`scripts/00_check_access.py`) | switch to IL-PCR / hand-entered crosswalk (proposal risk table) |
| 24 | core build complete | cut dense channel first, then tiers |
| 26 | E1–E4 have results and charts → start layer 2 | skip layers 2 and 3; polish evaluation and video |
| ~30 | layer 2 works → start layer 3 | ship layer 2 only (it scores better under the rubric) |
| 34 | code freeze | report + video only |
