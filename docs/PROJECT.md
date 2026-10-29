# Kanoon-Bridge Drift: change-aware legal retrieval across a statute replacement

> Course research project built on Kanoon-Bridge (CSD358). Not legal advice.

## 1. The problem

On 1 July 2024 the Bharatiya Nyaya Sanhita (BNS) replaced the Indian Penal Code (IPC). Almost all
case law that Indian courts cite was decided under the IPC. Kanoon-Bridge v1 bridges the two codes:
IPC and BNS sections map to shared **offence ids**, so a question about BNS 103 finds the IPC 302
murder precedents. That is the right recall decision.

It is the wrong *confidence* decision. An offence id says "same offence". The crosswalk gives no
sense of **how far** the law moved:

| IPC → BNS | What v1 assumes | What actually happened |
| --- | --- | --- |
| 302 → 103 | same offence | same text; a new mob-lynching sub-section (2) was added next to it |
| 323 → 115 | same offence | same offence; fine cap raised ₹1,000 → ₹10,000 |
| 304A → 106 | same offence | maximum raised 2 → 5 years; a new hit-and-run sub-section (10 years) |
| 124A → 152 | same offence | sedition dropped; a different offence (endangering sovereignty) with a new mental state ("purposely or knowingly") and 7 years instead of 3 |
| 497 → — | (no id) | adultery not re-enacted |

A lawyer reading a 2015 sedition judgment for a 2025 BNS 152 charge needs to know the judgment
interprets a different provision. v1 ranks it at full strength.

**Research question.** Can we measure, from the bare-act texts alone, how far each provision
moved, and use that measurement to (a) tell users what changed and (b) temper how much
IPC-era precedent counts for a BNS-era question?

## 2. Contributions

1. **A drift vector per provision** (`src/kanoon_bridge/drift/`). It is computed from IPC and BNS
   bare-act text and the existing crosswalk. Components:
   - clause-level **retention** (how much of the old text survives) and **novelty** (how much of the
     new text is new), split into *rewording* of matched clauses vs *added* clauses;
   - **renumbering-aware** comparison: IPC cross-references are rewritten into BNS numbering first;
   - **punishment structure**: death, life, min/max term, fine caps, community service, parsed from
     legal English including number words;
   - legal **elements**: mental-state words and quantified thresholds (ages, periods, head counts);
   - merge/split **structure**.
2. **A drift taxonomy** (identical, cosmetic, restructured, punishment, substantive, new, repealed,
   unverified). It also gives a **pairwise carry-over score** c(IPC s → BNS b) ∈ [0,1]. When a BNS
   section merges several IPC sections, each BNS clause is credited to the predecessor it matches
   best. So BNS 318's clauses from IPC 415 do not count as rewording of IPC 420.
3. **Drift-aware precedent transfer in ranking** (`drift/transfer.py`, `SearchOptions.drift`). For a
   BNS-era question, an IPC-era precedent is down-weighted by (1 − transfer). Recall is unchanged:
   the precedent still appears, ranked lower and explained.
4. **User-facing change explanations**. Every statute card and precedent in the app says what
   changed. A new **What changed** page (`app/web/changes.html`) shows class, carry-over, punishment
   before/after and a word-level diff for all 358 BNS sections. The CLI has `--drift ipc:124a`.
5. **Drift checks on generated answers** (layer 3). A sentence that cites an IPC section whose BNS
   successor changed is flagged. So is one grounded in an IPC-era precedent whose law changed.
6. **An evaluation against two independent gold sources**, with baselines, ablations, significance
   tests and a dev/test split (Section 4).
7. **A verified research agent and a computed, verified answer layer** (Section 6). The agent
   routes the question, refines the code family from what it retrieved and verifies its statutes;
   the answer computes the punishment from the statute text and checks every claim (quantities,
   sections, code in force, negation, verbatim quotes), then repairs what fails. Right punishment
   in 82% of 273 BNS questions vs 5% for v1; injected errors caught 82–100% vs 5–100%, with no
   false alarms on clean answers.

## 3. Method

```
IPC text ─┐                      ┌─ punishment.extract ──┐
          ├─ split_clauses ──────┤                       ├─ drift vector ─┬─ classify  -> class
BNS text ─┘  renumber_references ├─ diff.align (clauses) ┤                ├─ carry_over -> c(s -> b)
crosswalk ───────────────────────┴─ elements.compare ────┘                └─ (LR) P(changed)
```

**Alignment** (`drift/diff.py`). Both texts are split into clauses: sub-sections, items,
Explanations, Exceptions, provisos. Illustrations are left out because they explain the law
rather than state it, and our IPC source often omits them. Each new clause is matched to its
best old clause by word-bigram containment, and vice versa. A clause below 0.3 containment
counts as *added* or *lost*. Clause labels ("(1)", "(a)") and systematic BNS wording updates
("Sanhita", "person of unsound mind", "transportation for life") are normalised away first.

**Carry-over** (`drift/model.py`):

```
c_guilt    = retention · (1 − 0.5·reworded) · (1 − 0.15·added) · 0.6^[mens rea changed] · 0.7^[threshold changed]
c_sentence = c_guilt · 0.5^[punishment changed]
c          = 0.85·c_guilt + 0.15·c_sentence
```

The weights encode legal judgement and were set by hand, **not fitted to the gold labels**:

- rewording of a clause a precedent read matters more than a clause added beside it;
- a changed mental state or threshold changes what must be proved;
- a punishment change affects sentencing reasoning only.

When the old text in our source is incomplete, the score falls back to a prior by crosswalk
relation and the class is *unverified*.

**Transfer in ranking** (`drift/transfer.py`). A question counts as BNS-era if:

- its incident date is on or after 1 July 2024, or
- it names a BNS section and gives no date.

Its target sections are the BNS sections it names, or else those of its offence ids. For a
precedent p, transfer(p) is the best carry-over over (IPC section p cites, target section). It is
0 when every IPC section p applies was dropped. Scores at or above 0.85 are left alone. Below
that, `final −= μ·(1 − transfer)·spread`, with μ = 0.5. Here `spread` is the larger of the top
score and the score range, which keeps the penalty sign-safe for learned scores.

## 4. Evaluation

Reproduce: `make fetch && make drift && make drift-eval`. All tables are in
`results/tables/drift_*.csv` and all figures in `results/figures/drift_*.png`.

### Text sources

- **Old codes, BNSS and BSA: India Code.** The IPC, CrPC and Evidence Act come from India Code's
  current consolidated text, as packaged by [kanoonhq/open-laws](https://github.com/kanoonhq/open-laws)
  (CC0). So do the BNSS and BSA.
- **BNS: bns-study-platform.** The BNS text comes from that project, verified against the Gazette.
- **Cleaning.** Amendment footnote markers ("1[", "]"), "* * *" omissions and the State
  amendments India Code appends after the central text are removed before comparing.
- **Why this matters.** We first used the MIT indian-legal-mcp copies. Their IPC truncates
  sections (s.300 was a 10-word fragment; s.304B was empty), and their CrPC is partly pre-2009
  (no §41A–41D, old §41 and §54). Switching sources changed the results more than any modelling
  choice:

  | Test-half F1 (vs S1) | MIT copies | India Code |
  | --- | --- | --- |
  | IPC → BNS, drift rules (hand-set) | 0.68 | **0.85** |
  | IPC → BNS, learned model | 0.81 | **0.89** |
  | CrPC → BNSS, edit similarity | 0.68 | **0.78** |
  | CrPC → BNSS, drift rules (dev-tuned) | 0.73 | 0.75 |
  | Evidence Act → BSA, edit similarity | 0.12 | **0.53** |

  `drift.text_source: legacy` in `configs/default.yaml` reproduces the old runs.

### Gold labels: two independent sources

Neither source is used to build the drift reports.

- **S1:** the indian-legal-mcp mapping tables (MIT). Each BNS (sub-)section is marked Retained,
  Change, New Section or New Sub-Section, and each IPC section can be marked Deleted.
- **S2:** the change notes of the bns-study-platform project. The notes are free text; we turn
  them into categories (unchanged, structure, punishment, substance, new) with sentence-level
  rules in `drift/gold.py`.

Population: the 346 BNS sections that have an IPC predecessor and an S1 label. S1 marks 102 of
them changed and S2 marks 118. **The two sources agree with Cohen's κ = 0.75.** "Did it change?"
has more than one defensible answer.

### E11a — which provisions changed?

Test half (split by a hash of the section number). Thresholds and the dev-trained model only
ever saw the dev half.

| System | F1 changed (S1) | κ (S1) | F1 changed (S2) | κ (S2) |
| --- | --- | --- | --- | --- |
| v1 crosswalk relation ≠ "same" | 0.21 | −0.03 | 0.32 | 0.10 |
| Edit similarity, threshold tuned on dev | 0.62 | 0.39 | 0.68 | 0.45 |
| tf-idf cosine, threshold tuned on dev | 0.60 | 0.34 | 0.64 | 0.38 |
| **Drift rules, hand-set (no tuning)** | 0.85 | 0.78 | 0.76 | 0.65 |
| **Drift rules, thresholds tuned on dev** | 0.84 | 0.77 | 0.77 | 0.66 |
| **Drift vector + logistic regression (trained on dev)** | **0.89** | **0.85** | **0.81** | **0.73** |

- Every drift system beats v1 (McNemar's exact test, p < 0.001 on both sources).
- **Drift vs edit similarity.** The hand-set rules beat the best text-similarity baseline by
  0.22 F1 on S1 and 0.08 on S2. The 95% bootstrap CIs (in the CSV and
  `drift_change_detection.png`) do not overlap on S1: 0.77–0.91 vs 0.53–0.71.
- **The learned model's κ against S1 (0.85) is higher than the two sources' agreement with each
  other (0.75).** That should not be read as better-than-human. S1's labels may themselves come
  from comparing these texts, which would favour any text-based detector. S2, written as prose
  notes, is the more independent test; there the learned model reaches κ 0.73.
- The strongest standardised weights are punishment_changed (+1.8), lost_clauses (−1.0) and
  punishment_harsher (+1.0).

### E11b — ablation (hand-set rules, all 346 sections)

| Variant | F1 (S1) | F1 (S2) | AUC of 1−carry-over (S1) |
| --- | --- | --- | --- |
| full | 0.843 | 0.776 | 0.843 |
| − renumbering | 0.843 | 0.776 | 0.833 |
| − punishment | 0.196 | 0.228 | 0.725 |
| − elements (mens rea, thresholds) | 0.853 | 0.775 | 0.847 |
| − clause alignment | 0.857 | 0.779 | 0.832 |

- **Punishment parsing carries almost all of the detection signal.** Most BNS changes are raised
  fine caps and terms, one-word edits that text similarity cannot see.
- **The other components matter for the score and the explanation, not for detection.**
  - Renumbering and clause alignment add about 0.01 AUC to the carry-over score. They are what
    let the explanation say "a sub-section was added" or "section 334 became 122" instead of
    "the text changed".
  - The mental-state reading slightly *hurts* detection, a negative result. Mental-state words
    are flagged when the BNS adds a clause that happens to contain "knowingly". It stays for the
    explanation: it catches "purposely or knowingly" in BNS 152.
  - With the clean India Code text, dropping clause alignment no longer costs F1. Its earlier
    benefit came from routing around truncated IPC clauses.

### E11c/d — punishment

- **Change detection** against S2's "punishment" category: precision 0.80, recall 0.90,
  F1 0.85, κ 0.80 (346 sections).
- **Extraction accuracy:** `punishment.extract` on the bare act vs on the source's own one-line
  punishment summary, field by field (273 sections):

  | Field | Agreement |
  | --- | --- |
  | death | 95% |
  | life | 97% |
  | fine | 99.6% |
  | community service | 100% |
  | maximum term | 98% |
  | minimum term | 100% |
  | fine cap | 96% |

### E11e — repeal detection (gold: S1 Deleted)

| Family | System | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| IPC (19 deleted) | v1: not in the crosswalk | 0.24 | 0.84 | 0.38 | — |
| IPC | drift: no successor after a text search | **0.56** | 0.53 | **0.54** | **0.96** |
| CrPC (12 deleted) | v1 | 0.28 | 0.42 | 0.33 | — |
| CrPC | drift: no successor or carry-over < 0.5 | **0.70** | 0.58 | **0.64** | **0.98** |
| Evidence Act (7 deleted) | v1 | 0.50 | 0.57 | 0.53 | — |
| Evidence Act | drift: no successor or carry-over < 0.5 | 0.56 | 0.71 | **0.63** | **0.97** |

- The text search finds IPC sections the crosswalk misses but whose wording sits in a BNS section
  (48, mostly definitions consolidated into BNS 1–3). Sections repealed long before 2023 ("Rep. by
  the Prevention of Corruption Act, 1988") are excluded.
- The misses are sections the crosswalk maps to a replacement offence, such as 124A → 152. The
  carry-over score gives those low values instead.

### E11f — carry-over by S2 category (mean)

| S2 category | Mean carry-over |
| --- | --- |
| unchanged | 0.95 |
| structure | 0.89 |
| substance | 0.84 |
| punishment | 0.84 |

The score falls once a provision changes. It does not separate substance from punishment changes.

### E12 — hazard audit of v1's offence ids

**14 of 343** offence ids equate an IPC section with a BNS section whose carry-over is below 0.5
(`drift_hazard_offences.csv`). It was 26 with the truncated MIT text.

- Real changes found:
  - **sedition** (124A → 152, carry-over 0.24): a different offence;
  - **house-breaking by night** (446 → 330, 0.14): the "by night" aggravation moved out of the
    definition into the punishment section 331, according to the S2 note.
- Others are consolidations rather than changes of law: IPC 33 → BNS 3 (general explanations),
  and the coin and currency offences (242 → 180, 230 → 178).

This is an **upper bound that needs legal review**, not a count of errors. It shows where silent
transfer is possible.

### E13 — transfer inside retrieval

`make drift-retrieval` needs the IL-PCSR indexes, which are not redistributable and not in this
environment, so **we have not run it on the real corpus yet**. It asks one E3-style BNS-era
question per BNS section that the corpus cites ("cases under section 152 BNS"). It runs the full
system with drift off and on and reports, per drift class:

- citation-MAP (E3's gold);
- **stale@10**: the share of the top 10 whose transfer is below 0.5;
- moved@10.

The hypothesis: nothing moves for identical and cosmetic sections, and stale@10 falls for
substantive ones, at some cost in citation-MAP, because citation relevance counts a stale
precedent as relevant. On the synthetic sample (`make sample`) the script runs end to end. A
murder question leaves the ranking unchanged (carry-over ≥ 0.85); a BNS 152 question
down-weights sedition-era precedent. **Run it with the real indexes before reporting E13.**

## 5. Extension: CrPC → BNSS and Evidence Act → BSA

The same day the BNS replaced the IPC, the BNSS replaced the CrPC and the BSA replaced the
Evidence Act. Kanoon-Bridge v1 already bridges both through shared provision ids. The drift
layer now covers them too, with the same API: `DriftAnalyzer.from_sources(family="crpc")` or
`family="iea"`. Their tables are committed as `data/crosswalk/drift_crpc_bnss_*.csv` and
`drift_iea_bsa_*.csv`. Search, explanations, answer checks, the CLI and the What changed page
use all three families.

### What had to change

- **Code families** (`drift/provisions.py: FAMILIES`): each pair of codes has its own labels,
  source files and crosswalk. User-facing reasons say "CrPC 438", "BNSS 482", "Evidence Act 65B".
- **No punishment parsing for procedure and evidence law.** Those codes rarely prescribe
  punishment. Their "offence punishable with death… detention for fifteen days" read as a
  punishment change (CrPC 167). Time limits are read by the threshold reader instead.
- **Cross-code renumbering.** CrPC text cites "section 302 of the Indian Penal Code", and BNSS
  text cites "section 103 of the Bharatiya Nyaya Sanhita". A reference followed by another
  code's name is renumbered through that code's crosswalk. The six Act names are treated as
  cosmetic. Ranges ("sections 121 to 126") are not renumbered.
- **Short qualifiers.** S1 counts a six-word insertion in a long section ("but within six hours
  from such arrest") as a change. Two features were added for this: absolute inserted/deleted
  content words, and newly added time or count limits. They were designed **after error
  analysis on the S1 procedure labels**, so the procedure numbers below are optimistic. The
  hand-set rules keep them off; only dev tuning switches them on.
- **Source repairs.**
  - The sections filed under the wrong number in the source (BNSS 58 holds §59's text, as does
    India Code's own extraction; BSA 80) are detected and marked *unverified*.
  - `ingest/` was still looking for the indian-legal-mcp checkout's old file names, so BNSS and
    BSA sections had silently dropped out of the corpus. They are indexed again (531 + 170
    sections). The crosswalks rebuilt from the renamed files are identical to the committed ones.

### Results (gold: S1 only; no S2 notes exist for these codes)

Test half; dev-tuned rows were tuned on the dev half only.

| System | CrPC → BNSS F1 (κ) | Evidence Act → BSA F1 (κ) |
| --- | --- | --- |
| v1 crosswalk relation | 0.27 (0.16) | 0.40 (0.23) |
| Edit similarity (tuned) | **0.78** (0.68) | **0.53** (0.43) |
| Drift rules, penal thresholds | 0.48 (0.37) | 0.46 (0.36) |
| Drift rules, dev-tuned (what the committed tables use) | 0.75 (0.62) | 0.40 (0.29) |
| Drift vector + logistic regression | 0.74 (0.64) | 0.45 (0.37) |

- **Edit similarity is the best change detector for procedure and evidence law,** a negative
  result for the drift vector outside penal law.
  - The penal signal was punishment, which these codes do not have.
  - What is left is wording, and a whole-text edit ratio measures wording as well as our
    clause-level features do.
  - On CrPC the drift systems are close (0.75 vs 0.78, CIs 0.67–0.82 vs 0.71–0.85).
- **The committed procedure tables use the dev-tuned thresholds** (`drift.family_params` in
  `configs/default.yaml`). With them, the "insertion" rule fires on CrPC 167 → BNSS 187, where
  the 15 days of police custody may now be taken "in parts, at any time during the initial forty
  days or sixty days". Its carry-over stays 0.91: the carry-over formula does not use inserted
  words yet (future work).
- **The drift layer is still what the system uses for these codes.** Its value is the carry-over
  score per (old, new) pair and the explanation (which clause, which time limit, which
  renumbered reference), which edit similarity does not give.
- **Evidence Act → BSA:** 168 sections and only 24 changed per S1, so every number here has wide
  CIs (0.28–0.73 for edit similarity). S1 is also questionable: it marks Evidence Act 65B
  "Deleted", though BSA 63 is its well-known successor.
- **Repeal detection** is strong (E11e above: AUC 0.98 and 0.97).
- **Hazard audit (E12):** 5 of 511 CrPC/BNSS provision ids and 5 of 168 Evidence/BSA ids pair
  sections at carry-over < 0.5. It was 13 and 8 with the MIT text. The remaining CrPC ones are
  mostly crosswalk errors from text-similarity alignment (`ingest/align_codes.py`): CrPC 10, 15
  and 19 merged into BNSS 13. The CrPC 39 → BNSS 33 range references ("sections 121 to 126")
  are not renumbered. Treat them as leads for review.

`make drift-eval` writes these as `results/tables/drift_procedure_*.csv` and
`results/figures/drift_families.png`.

## 6. Advanced layers: agent v2 and RAG v2

Layers 2 (research agent) and 3 (grounded answers) were rebuilt around one idea: **compute and
verify what can be computed, never trust the generator with a section number or a punishment.**
Both v1 layers stay available (`agent.version: 1`, `rag.version: 1` in `configs/default.yaml`)
and are the baselines below.

### Agent v2 (`agent/router.py`, `verify.py`, `issues.py`, `memo.py`, `research.py`)

1. **Route** the question (cf. Adaptive-RAG, Jeong et al., NAACL 2024): `case_text` (a pasted
   judgment, > 50 words: no pseudo-relevance feedback, case text down-weighted),
   `section_lookup` (names a section) or `fact_question`.
2. **Pick the code family** (penal / procedure / evidence). Cue words give a first guess; after
   the first round it is *refined from the evidence*: the codes of the top retrieved statutes
   and the codes the retrieved precedents cite. Cue words alone misrouted questions such as
   "false evidence" (a penal offence), and dropped E9 to 0.32 before this step.
3. **Family sub-query**: a statutes-only search restricted to the family's code in force.
4. **Citation votes**: statutes cited by several retrieved precedents get a fused bonus
   (at least `min_voters` precedents). Votes are used only when no top-5 statute heading matches
   the question well (`vote_gate`).
5. **Verify** the top statutes: family match, code in force on the incident date, stemmed
   heading specificity, and a penalty for definition / general-explanation sections
   (BNS 3 was being pulled up by IPC 34 citations).
6. **Issues**: for a fact question, the chosen statute is split into its sub-sections (issues),
   each issue's key terms are looked for in the retrieved precedents, and the coverage is
   reported; uncovered issues get one more targeted search round.
7. **Research memo**: law in force with its old-code counterpart, punishment, what changed
   (drift table), issues covered or [OPEN], searches made.

### RAG v2 (`rag/structured.py`, `claim_check.py`, `correct.py`, `answer.py`)

1. **Structured answer** (offline generator): the law in force named with its heading, the
   punishment *computed* by the drift punishment parser per sub-section, one quoted precedent
   sentence; the counterpart section and what changed go in separate notes, not in the answer.
   With an API key the LLM answers instead and is checked the same way.
2. **Claim verification** of every sentence against its cited source (full statute text, not
   the chunk): idf-weighted support; every quantity (years, rupees, days) must appear in the
   source; every section named must be the source's own, cited by it, or its crosswalk
   equivalent; code in force; a modal negation the source does not have (or a dropped one) is a
   contradiction; quotations must be verbatim.
3. **Correction** (cf. CRAG, Yan et al. 2024): an LLM answer is regenerated once with the list of
   problems; then deterministically, a wrong-code section is swapped to its counterpart in force
   (and re-verified), a contradicted sentence is replaced by the source's own words, an
   unsupported one is removed. Every action is listed in `answer.corrections`.

### Results (open corpus: `make open && make layers-eval`)

The open corpus has all six codes from India Code plus the BNS, and the eight synthetic
precedents; no IL-PCSR and no API key are needed, so every number below is reproducible.

**E14 — punishment QA, end to end.** 273 questions, "What is the punishment for
<BNS heading>?", incident date 2025-01-01. Gold: bns-study-platform's one-line punishment
summary (written independently of the bare act), parsed; a punishment is right only if every
field matches (death, life, maximum and minimum term, fine and its caps, community service).

| System | Right statute @1 | Right punishment | … given the right statute | Corrections / answer |
| --- | --- | --- | --- | --- |
| core + RAG v1 (extractive quote) | 0.894 | 0.051 | 0.053 | — |
| core + RAG v2 (structured + verified) | 0.894 | 0.821 | 0.914 | 0.022 |
| agent v2 + RAG v2 | **0.908** | **0.824** | 0.903 | 0.026 |

v1 quotes the best sentence of the statute, which usually is not the punishment clause; v2
computes it. The remaining errors are mostly gold summaries that describe one sub-section only.

**E15 — does verification catch errors? (error injection).** One error per type is injected
into each right-statute answer, in the sentence that cites the statute.

| Injected error | n | v1 checks (cosine + code) | v2 claim verification |
| --- | --- | --- | --- |
| term of years changed | 185 | 0.05 | **0.89** |
| fine amount × 10 | 63 | 0.71 | **1.00** |
| negation flipped | 241 | 0.25 | **0.82** |
| old-code section for a 2025 incident | 235 | 1.00 | 1.00 |
| section number changed | 241 | 0.66 | **1.00** |
| another statute's punishment added | 244 | 0.40 | **0.84** |
| *false alarm on the clean answers* | 244 | 0.98 | **0.00** |

On RAG v1's extractive answers as the base, v2 still detects 75–100% per type with a 0.8%
false-alarm rate (v1: 50%). Figure: `results/figures/layers_error_injection.png`.

**E16 — the agent on the statute-level sets** (right provision of the right code at rank 1).

| System | E2 collisions (expected@1) | E7 penal (both@1) | E9 procedure (both@1) |
| --- | --- | --- | --- |
| baseline BM25 | 0.83 | 0.40 | 0.37 |
| core | 0.81 | 0.79 | 0.75 |
| agent v1 | 0.88 | 0.79 | 0.75 |
| **agent v2** | **0.92** | **0.88** | **0.83** |
| v2 − verification | 0.92 | 0.79 | 0.71 |
| v2 − citation votes | 0.93 | 0.88 | 0.83 |
| v2, votes not gated | 0.92 | 0.86 | 0.82 |
| v2 − family search | 0.88 | 0.90 | 0.86 |

- **Verification is the part that matters**: without it E7 and E9 fall back to core.
- **Family search is mixed**: +0.05 on E2, −0.02 / −0.03 on E7 / E9. On E9 the family is
  sometimes refined from the eight synthetic precedents, which are mostly penal, so code@1 drops
  from 0.95 (core) to 0.92. With a real precedent pool this should improve; it is reported as is.
- **Citation votes do nothing here once gated**, and cost 1–2 points ungated. With eight
  precedents there is little to vote with; they are kept for the IL-PCSR corpus, untested there.

### Caveats

- E7, E9 and E14 questions are written from section headings, which favours heading matching
  (the verifier's specificity term). Typed, fact-style questions (E4) need the IL-PCSR indexes
  and the judged pool, and were not re-run.
- E15 injects the error types the verifier was designed to catch: it is an upper bound, not an
  estimate of how often real LLM errors are caught.
- The vote gate (`vote_gate`) was set after looking at E16; there is no held-out set for it.
- No LLM was called (no API key in the build environment): the regeneration step is tested only
  by unit tests, and RAG v2's numbers are for the structured generator.

## 7. Limitations

- **Source text.** India Code's consolidated text is current but machine-extracted: one BNSS
  and one BSA section are filed under the wrong number (flagged), and footnote and State-amendment
  cleanup is rule-based.
- **Gold labels are coarse, and S1 may be text-derived.** S1 is section-level Retained/Change;
  S2 is our rule-based reading of free-text notes. Neither labels carry-over directly, so
  carry-over is validated only indirectly (E11f, AUC).
- **Procedure crosswalk.** It is built by text similarity and has some wrong merges. The
  open-laws dataset has a 107-provision curated crosswalk that could be used to check it.
- **Carry-over weights are hand-set.** They are principled but not learned. A small set of
  lawyer-graded (precedent, BNS section) pairs would let us fit and test them.
- **Citation relevance ≠ legal applicability.** IL-PCSR's gold cannot reward drift-aware ranking;
  E13 therefore reports stale@10 next to MAP.

## 8. Future work

- Learn the carry-over weights from graded pairs; give the mental-state reading a learned weight;
  let inserted qualifiers (time limits, new conditions) lower the carry-over.
- Procedure and evidence (Section 5): check the procedure crosswalk against open-laws' curated
  mappings, and get graded labels with more positives for the Evidence Act.
- Zone-aware transfer: down-weight only a precedent's sentencing paragraphs when only the
  punishment changed.
- Agent v2 and RAG v2 on IL-PCSR and with an LLM generator: typed fact questions (E4), real
  LLM errors instead of injected ones, and a held-out set for the vote gate.
- A user study with law students: does the "What changed" view reduce wrong-code citations?

## 9. Files

| File | Role |
| --- | --- |
| `src/kanoon_bridge/drift/provisions.py` | code families, India Code / legacy text loaders, clause splitting, cross-reference renumbering |
| `src/kanoon_bridge/drift/punishment.py` | punishment parser and comparison |
| `src/kanoon_bridge/drift/elements.py` | mental-state words, thresholds, content words |
| `src/kanoon_bridge/drift/diff.py` | clause alignment, word diff |
| `src/kanoon_bridge/drift/model.py` | drift vector, classes, carry-over, `DriftTable` (CSV) |
| `src/kanoon_bridge/drift/classify.py` | logistic regression (numpy, Newton/IRLS) + CV |
| `src/kanoon_bridge/drift/gold.py` | S1 and S2 gold readers (evaluation only) |
| `src/kanoon_bridge/drift/transfer.py` | transfer score, ranking hook, answer checks |
| `src/kanoon_bridge/drift/explain.py` | "What changed?" view for CLI and web |
| `src/kanoon_bridge/eval/drift_eval.py` | E11–E12 metrics, baselines, McNemar, bootstrap |
| `scripts/13_build_drift.py`, `14_eval_drift.py`, `15_drift_retrieval.py` | build, evaluate, retrieval experiment |
| `data/crosswalk/drift_sections.csv`, `drift_pairs.csv` | the committed drift table (IPC → BNS) |
| `data/crosswalk/drift_crpc_bnss_*.csv`, `drift_iea_bsa_*.csv` | the same for CrPC → BNSS and Evidence Act → BSA |
| `app/web/changes.html`, `changes.js` | the "What changed" page |
| `tests/test_drift.py` | unit tests; `tests/test_layers_integration.py::test_drift_layer_on_the_engine` |
| `src/kanoon_bridge/agent/router.py`, `verify.py`, `issues.py`, `memo.py` | agent v2: routing and family, statute verifier, issue coverage, research memo |
| `src/kanoon_bridge/rag/structured.py`, `claim_check.py`, `correct.py` | RAG v2: structured answer, claim verification, correction |
| `src/kanoon_bridge/eval/open_eval.py`, `layers_eval.py`, `scripts/17_eval_layers.py` | E14–E16 on the open corpus |
| `tests/test_layers_v2.py` | unit tests; `tests/test_layers_integration.py::test_agent_v2_and_rag_v2` |

## 10. Sources

- Bare-act text:
  - BNS via bns-study-platform (PSKprem); only `number`, `title` and `bareActText` are used to
    build reports. Its `changeNote` and `punishment` fields are used only as evaluation labels.
  - IPC, CrPC, Evidence Act, BNSS and BSA: India Code's consolidated text via
    [kanoonhq/open-laws](https://github.com/kanoonhq/open-laws) (CC0). The earlier runs used the
    GSMS-B/indian-legal-mcp copies (MIT), which still serve as the fallback and as gold source S1.
- No raw source is committed.
- Related ideas:
  - legal change and versioning of legislation (consolidated-law diffing, e.g. the EU's
    EUR-Lex consolidation);
  - containment as a near-duplicate measure (Broder, SEQUENCES 1997);
  - McNemar's test for paired classifiers (Dietterich, *Neural Computation* 1998);
  - Cohen's κ (1960);
  - query routing by question type (Adaptive-RAG, Jeong et al., NAACL 2024); corrective
    retrieval-augmented generation (CRAG, Yan et al. 2024); citation evaluation (ALCE, Gao et al., EMNLP 2023).
