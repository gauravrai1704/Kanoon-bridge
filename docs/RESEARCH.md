# Kanoon-Bridge: features and the research behind them

This file maps every feature to the IR technique it uses, the work it builds on, the file that
implements it, and the experiment that measures it. Use it for the report's "Related work" and
"Novelty" sections and for the viva.

**IIR** below means Manning, Raghavan & Schütze, *Introduction to Information Retrieval* (CUP,
2008), the course textbook. **Ours** marks a contribution we have not seen in prior legal IR
systems.

## 1. Query understanding

| Feature | Technique and source | File | Measured by |
| --- | --- | --- | --- |
| Section tokens ("Section 302 IPC", "u/s 498A", "302 r/w 34", "323-325") | Domain-specific tokenisation (IIR §2.2) | `text/tokenize.py` | tests; E2 |
| **IPC↔BNS version normalisation** (sections → offence ids, both codes) | Crosswalk built from the BNS bare act. Equivalence classes act as index terms, as in thesaurus-based normalisation (IIR §2.2.3). **Ours**: applied across a statute *replacement* | `text/version_norm.py`, `ingest/parse_crosswalk.py` | E3 (cross-version), ablation "+bridge" |
| **Bare-number collision resolver** ("302" = IPC murder before 1 Jul 2024, BNS religious-feelings after) | Date prior × offence-keyword context, a word-sense-disambiguation-style decision. **Ours** | `text/collision.py` | E2 wrong_hit@10 |
| Hinglish / Devanagari queries | Own transliteration table and normalisation rules; legal lexicon of ~165 entries | `text/transliterate.py` | E4 language ablation |
| Phonetic matching for Roman Hindi | Soundex (Russell & Odell, 1918; IIR §3.4) adapted to Hindi aspirates and vowel length | `text/phonetic.py` | E4 "+phonetic" |
| Hindi stemming | Light suffix stripping: Ramanathan & Rao, "A lightweight stemmer for Hindi", 2003 | `text/stem.py` | tests |
| **Spelling correction + "did you mean"** | Bigram k-gram index for candidates (Zobel & Dart, *Software: Practice & Experience* 1995; IIR §3.3); Damerau–Levenshtein distance (Damerau, CACM 1964; Kukich, ACM Computing Surveys 1992); ranked by document frequency | `text/spell.py`, analyzer step 5b | **Typo experiment** (`eval/typos.py`) |
| **Wildcard terms** (`extort*`, `*bail`), also inside Boolean queries | k-gram wildcard index with post-filtering (IIR §3.2.2) | `text/spell.py`, `query/boolean.py` | tests |
| Boolean, phrase and proximity syntax | Recursive-descent parser. Two-pointer postings intersection in increasing-df order; positional intersect for phrases and `/k` (IIR ch. 1–2). **Ours**: a section term matches its offence in either code | `query/parser.py`, `query/boolean.py`, `query/proximity.py` | tests; agent "boolean" sub-query |
| Facet filters (`state:`, `code:`, `court:`, `after:`, `before:`) | Parametric/zone indexes (IIR §6.1). Spatio-temporal querying follows Dr. Sonia Khetarpaul's line of work | `index/facets.py` | E6, E7 |

## 2. Ranking

| Feature | Technique and source | File | Measured by |
| --- | --- | --- | --- |
| BM25F over judgment zones (facts / arguments / reasoning / decision) | Robertson, Zaragoza & Taylor, "Simple BM25 extension to multiple weighted fields", CIKM 2004 (Microsoft Research). Zone value in legal case retrieval: SAILER (Li et al., SIGIR 2023, Tsinghua) | `rank/bm25f.py`, `ingest/segment.py` | ablation "+zones" |
| tf-idf (lnc.ltc) baseline | IIR ch. 6 | `rank/vsm.py` | baseline rows |
| Statute bridge (statutes ranked first, then used to find precedents) | Two-stage statute → precedent retrieval; the statute network as a bridge, as in Hier-SPCNet (Bhattacharya et al., SIGIR 2020, IIT Kharagpur) | `rank/statute_bridge.py` | ablation "+bridge" |
| Citation authority | PageRank (Brin & Page, WWW 1998, Stanford) on the citation graph. Legal importance of precedents: Fowler et al., *Political Analysis* 2007 | `rank/citation_graph.py`, `rank/authority.py` | ablation "+authority" |
| **Jurisdiction-aware authority** g(d \| state): binding (SC + own HC) vs persuasive | PageRank per jurisdiction subgraph, in the spirit of topic-sensitive PageRank (Haveliwala, WWW 2002, Stanford). **Ours**: Indian *stare decisis* | `rank/authority.py` | E6 binding_share@5, ablation "+jurisdiction" |
| Dense channel (paragraph embeddings, best paragraph per document) | E5 embeddings (Wang et al., 2022, Microsoft); NLLB-E5 multilingual model (Hindi-BEIR, NAACL 2025); best-paragraph scoring (Dai & Callan, SIGIR 2019, CMU) | `rank/dense.py`, `rank/fusion.py` | ablation "+dense" |
| QPP-gated lexical/dense mix | Pre-retrieval (He & Ounis, SPIRE 2004) and post-retrieval score-gap predictors. QPP for agentic RAG: Tian et al., IR-RAG @ SIGIR 2025 | `rank/qpp.py` | ablation "+qpp" |
| **Learning to rank** | Linear model trained by coordinate ascent directly on MAP: Metzler & Croft, *Information Retrieval* 2007 (UMass Amherst). Pairwise alternative: Joachims, KDD 2002 (Cornell). The offence-match feature reflects LeCaRD's "key element" relevance (Ma et al., SIGIR 2021, Tsinghua) | `rank/ltr.py`, `scripts/06_train_ltr.py` | 5-fold CV on val; ablation "+ltr" |
| Tiered index and champion lists | Static quality ordering (IIR §7.1) | `index/tiers.py` | efficiency (latency vs Recall@20) |

## 3. Results and interaction

| Feature | Technique and source | File | Measured by |
| --- | --- | --- | --- |
| **Query-biased snippets with highlighting** | Tombros & Sanderson, SIGIR 1998 ("Advantages of query biased summaries in IR"); Turpin et al., SIGIR 2007 ("Fast generation of result snippets in web search") | `present.py` | demo |
| **"Why this result"** (offence match across codes, zone, binding/persuasive, old-code relevance) | Explanation from the score breakdown. **Ours** (legal-specific) | `present.py` | demo |
| Statute version notes ("BNS 103 ← IPC 302, same offence"; "new in BNS") | From the crosswalk. **Ours** | `present.py` | demo |
| **Near-duplicate collapse** | Shingling + MinHash (Broder, SEQUENCES 1997); LSH banding (IIR §19.6); multiply-shift hashing (Dietzfelbinger et al., *J. Algorithms* 1997). Web-scale variant: Manku, Jain & Das Sarma, WWW 2007 (Google) | `index/dedup.py` | build report (groups found) |
| **Similar cases ("more like this")** | Text profile used as a query, plus bibliographic coupling (Kessler, 1963, MIT) and co-citation (Small, JASIS 1973). Text + citation similarity for Indian judgments: Kumar et al., 2011 (IIT Kharagpur). **Ours**: coupling at the *offence* level, so IPC-era and BNS-era cases couple | `rank/similar.py` | demo; tests |
| **Explicit relevance feedback** | Rocchio (SMART, 1971; IIR ch. 9) | `rank/feedback.py`, app checkbox, CLI `--feedback` | demo |
| Facet counts (court / decade / code) | Faceted navigation | `present.py`, app sidebar | demo |

## 4. Layer 2: research agent

| Feature | Technique and source | File |
| --- | --- | --- |
| Rule planner: cross-code, Boolean, binding-court facet and statutes-first sub-queries | Query reformulation into structured IR queries | `agent/plan.py` |
| Fusion | Reciprocal rank fusion (Cormack, Clarke & Büttcher, SIGIR 2009, Waterloo); CombSUM (Fox & Shaw, TREC-2 1994) | `agent/fuse.py` |
| Reflection + pseudo-relevance feedback round | QPP-triggered PRF (relevance models: Lavrenko & Croft, SIGIR 2001, UMass) | `agent/reflect.py` |
| Optional LLM planner | The LLM only *proposes* sub-query text; ranking stays IR | `agent/plan.py` |

## 5. Layer 3: grounded answers

| Feature | Technique and source | File |
| --- | --- | --- |
| Abstention before generation | QPP signals + idf-weighted question coverage by the retrieved sources | `rag/abstain.py`, `rag/answer.py` |
| Citation-support check | tf-idf cosine of each sentence vs its cited chunk. Citation evaluation for LLM answers: ALCE (Gao et al., EMNLP 2023, Princeton) | `rag/citation_check.py` |
| **Version-validity check** (wrong code for the incident date; bare colliding numbers) | Reuses the crosswalk and the collision resolver. **Ours** | `rag/version_check.py` |

## 6. Evaluation

| Experiment | Method and source | File |
| --- | --- | --- |
| E1 precedent / statute retrieval | IL-PCSR protocol: macro-F1@k with k chosen on val, plus MAP and MRR | `eval/run_eval.py` |
| Ablation ladder (bm25 → … → +ltr) | One component added per step | `eval/ablation.py` |
| **Significance** for every comparison | Paired randomization test + bootstrap CI (Smucker, Allan & Carterette, CIKM 2007); Holm–Bonferroni correction | `eval/significance.py` |
| Generated E2 / E3 / E7 sets | Gold derived from the crosswalk and IL-PCSR citations; E3 pairs each query with an IPC-worded control to isolate the version effect | `scripts/07_make_test_sets.py` |
| **Typo robustness** | Simulated known-item queries (Azzopardi, de Rijke & Balog, SIGIR 2007); Damerau error model | `eval/typos.py` |
| Efficiency | Latency and Recall@20 vs exhaustive scoring | `eval/efficiency.py` |
| Agent vs core; RAG support / version errors / abstention | | `eval/agent_eval.py` |
| Inter-judge agreement for hand-built sets | Cohen's κ | `eval/agreement.py` |

## What is new here

These are the claims to make in the report. Each is backed by an experiment above.

1. **Version-aware legal IR across a statute replacement.** IPC and BNS sections map to shared offence ids everywhere in the system: indexing, Boolean search, the statute bridge, similar-case coupling, LTR features and RAG version checks.
2. **Collision resolution for bare section numbers** that changed meaning on 1 July 2024.
3. **Jurisdiction-aware authority**: binding vs persuasive precedent for the user's state, inside the ranking itself.
4. **Typo-robust, Hinglish-robust legal search**, measured with known-item simulation and the E4 language ablation.
5. **Legal explanations** for every result: which offence matched across codes, and whether the case binds the user.
6. **QPP-gated legal RAG** that abstains, checks that each sentence is supported by its cited source, and flags out-of-date sections.
