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

### Gold labels: two independent sources

Neither source is used to build the drift reports.

- **S1:** the indian-legal-mcp mapping tables (MIT). Each BNS (sub-)section is marked Retained,
  Change, New Section or New Sub-Section, and each IPC section can be marked Deleted.
- **S2:** the change notes of the bns-study-platform project. The notes are free text; we turn
  them into categories (unchanged, structure, punishment, substance, new) with sentence-level
  rules in `drift/gold.py`.

Population: the 347 BNS sections that have an IPC predecessor and an S1 label. S1 marks 103 of
them changed and S2 marks 119. **The two sources agree with Cohen's κ = 0.75**. That is a rough
ceiling for any automatic detector: "did it change?" has more than one defensible answer.

### E11a — which provisions changed?

Test half (184 sections, split by a hash of the section number). Thresholds and the
dev-trained model only ever saw the dev half.

| System | F1 changed (S1) | κ (S1) | F1 changed (S2) | κ (S2) |
| --- | --- | --- | --- | --- |
| v1 crosswalk relation ≠ "same" | 0.20 | −0.03 | 0.32 | 0.10 |
| Edit similarity, threshold tuned on dev | 0.52 | 0.23 | 0.63 | 0.35 |
| tf-idf cosine, threshold tuned on dev | 0.55 | 0.24 | 0.59 | 0.27 |
| **Drift rules, hand-set (no tuning)** | 0.68 | 0.50 | 0.68 | 0.46 |
| **Drift rules, thresholds tuned on dev** | 0.72 | 0.57 | 0.70 | 0.52 |
| **Drift vector + logistic regression (trained on dev)** | **0.81** | **0.72** | **0.75** | **0.61** |

- McNemar's exact test against v1: p = 0.0003 (S1) and 0.015 (S2) for the dev-tuned rules;
  p < 0.001 for the learned model on both sources.
- 95% bootstrap CIs are in the CSV and in `drift_change_detection.png`.
- The learned detector's agreement with S1 (κ = 0.72) is about as high as the two human
  sources' agreement with each other (κ = 0.75).
- Its strongest standardised weights are punishment_changed (+2.1) and lost_clauses (−1.1).
- These numbers include the inserted-word and threshold-added features introduced for the
  procedure codes (Section 5). They moved the learned model from 0.79 to 0.81 on S1 and from
  0.76 to 0.75 on S2. The hand-set rules keep those features off by default, so their rows
  did not change.
  The negative weight is a source artefact: our IPC text lacks some Explanations.

### E11b — ablation (hand-set rules, all 347 sections)

| Variant | F1 (S1) | F1 (S2) | AUC of 1−carry-over (S1) |
| --- | --- | --- | --- |
| full | 0.655 | 0.660 | 0.748 |
| − renumbering | 0.653 | 0.665 | 0.741 |
| − punishment | 0.333 | 0.369 | 0.662 |
| − elements (mens rea, thresholds) | 0.664 | 0.669 | 0.803 |
| − clause alignment | 0.634 | 0.640 | 0.659 |

- **Punishment parsing carries most of the signal.** Many BNS changes are raised fine caps
  and terms, one-word edits that text similarity cannot see.
- **Clause alignment helps.** It is the difference between "a sub-section was added" and "the
  section was rewritten".
- **Renumbering hardly matters for detection.** It does matter for the explanation shown to users
  (e.g. "section 334" → "section 122" is shown as renumbered, not as a change).
- **Negative result: the regex reading of mental-state words hurts detection slightly** (removing
  it raises AUC from 0.75 to 0.80). Mental-state words get flagged when BNS adds a clause that
  happens to contain "knowingly". We keep the component because the explanation needs it (it
  catches "purposely or knowingly" in BNS 152). Its detection weight should be lower or learned;
  see future work.

### E11c/d — punishment

- **Change detection** against S2's "punishment" category: precision 0.71, recall 0.89,
  F1 0.79, κ 0.72 (344 sections).
- **Extraction accuracy:** `punishment.extract` on the bare act vs on the source's own one-line
  punishment summary, field by field (273 sections):

  | Field | Agreement |
  | --- | --- |
  | death | 95% |
  | life | 96% |
  | fine | 99.6% |
  | community service | 100% |
  | maximum term | 99.6% |
  | minimum term | 100% |
  | fine cap | 96% |

### E11e — repeal detection (553 IPC sections; gold: S1 Deleted, 19 sections)

| System | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- |
| v1: IPC section not in the crosswalk | 0.25 | 0.84 | 0.38 | — |
| drift: no successor after a text search of the BNS | **0.60** | 0.47 | **0.53** | **0.95** |

- The text search found 50 IPC sections that the crosswalk misses but whose wording sits in a
  BNS section. Most are definitions consolidated into BNS 1–3. This removes most of v1's false
  repeals.
- The misses are sections the crosswalk maps to a replacement offence: 124A, 309, 377, 497 →
  replacement. Our carry-over gives them low scores instead.

### E11f — carry-over by S2 category (mean)

| S2 category | Mean carry-over |
| --- | --- |
| unchanged | 0.92 |
| punishment | 0.81 |
| substance | 0.80 |
| structure | 0.78 |

The score falls once a provision changes. It does not separate substance from structure well;
merges score low partly because of source-text gaps.

### E12 — hazard audit of v1's offence ids

**26 of 343** offence ids equate an IPC section with a BNS section whose carry-over is below 0.5
(`drift_hazard_offences.csv`).

- Real changes found:
  - **sedition** (124A → 152): a different offence;
  - **house-breaking by night** (446 → 330): the "by night" aggravation moved out of the
    definition into the punishment section 331, according to the S2 note;
  - **dishonest misappropriation** (403 → 314): a new mandatory minimum of 6 months. That is a
    real change, though our class calls it *substantive* where *punishment* would be more
    accurate.
- Others are artefacts of incomplete IPC text or definition chapters, e.g. general explanations.

So this is an **upper bound that needs legal review**, not a count of errors. It shows where
silent transfer is possible.

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
murder question leaves the ranking unchanged (carry-over 0.89 ≥ 0.85); a BNS 152 question
down-weights sedition-era precedent. **Run it with the real indexes before reporting E13.**

## 5. Extension: CrPC → BNSS and Evidence Act → BSA

The same day the BNS replaced the IPC, the BNSS replaced the CrPC and the BSA replaced the
Evidence Act. Kanoon-Bridge v1 already bridges both through shared provision ids. The drift
layer now covers them too, with the same API: `DriftAnalyzer.from_sources(family="crpc")` or
`family="iea"`. Their tables are committed as `data/crosswalk/drift_crpc_bnss_*.csv` and
`drift_iea_bsa_*.csv`. Search, explanations, answer checks, the CLI and the What changed page
use all three families. For example, a 2025 bail question citing BNSS 482 down-weights CrPC 438
precedents (carry-over 0.53). The diff shows why: BNSS 482 dropped the list of factors CrPC
438(1) told the court to weigh.

### What had to change

- **Code families** (`drift/provisions.py: FAMILIES`): each pair of codes has its own labels,
  source files and crosswalk. User-facing reasons say "CrPC 438", "BNSS 482", "Evidence Act 65B".
- **No punishment parsing for procedure and evidence law.** Those codes rarely prescribe
  punishment. Their "offence punishable with death… detention for fifteen days" read as a
  punishment change (CrPC 167). Time limits are read by the threshold reader instead.
- **Cross-code renumbering.** CrPC text cites "section 302 of the Indian Penal Code", and BNSS
  text cites "section 103 of the Bharatiya Nyaya Sanhita". A reference followed by another
  code's name is renumbered through that code's crosswalk. The six Act names are treated as
  cosmetic.
- **Source repairs.**
  - Headings that wrap over a line break are stripped.
  - Three new-code sections whose source text is actually the next section's (BNSS 58 and 275,
    BSA 80) are detected and marked *unverified*. This is a chunking error in the source.
  - `ingest/` was still looking for the checkout's old file names, so BNSS and BSA sections had
    silently dropped out of the corpus. They are indexed again (531 + 170 sections). The
    crosswalks rebuilt from the renamed files are identical to the committed ones.

### Results (gold: S1 only; no S2 notes exist for these codes)

Test half; dev-tuned rows were tuned on the dev half only.

| System | CrPC → BNSS F1 (κ) | Evidence Act → BSA F1 (κ) |
| --- | --- | --- |
| v1 crosswalk relation | 0.27 (0.16) | **0.40** (0.23) |
| Edit similarity (tuned) | 0.68 (0.50) | 0.12 (0.05) |
| Drift rules, hand-set | 0.53 (0.39) | 0.29 (0.09) |
| Drift rules, dev-tuned | **0.73** (0.61) | 0.32 (0.17) |
| Drift vector + logistic regression | 0.62 (0.48) | 0.00 (−0.02) |

- **CrPC → BNSS** (513 sections, 171 changed per S1; 82 changed in the test half):
  - The dev-tuned rules are best, but only slightly ahead of edit similarity: the 95% CIs
    overlap (0.65–0.80 vs 0.60–0.75), so this is not a significant win.
  - The hand-set rules, calibrated on penal law, are too coarse here. S1 counts a six-word
    insertion in a long section ("but within six hours from such arrest") as a change.
  - The two features that capture this were added **after error analysis on these same
    labels**: absolute inserted/deleted content words, and newly added time or count limits.
    The dev/test split guards the tuned thresholds, but the feature design saw both halves'
    errors, so treat the CrPC numbers as optimistic.
  - The dev half chose novelty ≥ 0.25, retention ≤ 0.7, ≥ 1 added clause, or ≥ 4 inserted
    content words.
- **Evidence Act → BSA** (168 sections, only 24 changed per S1; 13 in the test half):
  - Every method is unreliable. The learned model predicts "unchanged" everywhere, and v1's
    similarity-based relation is the best of a weak field.
  - Ranking quality is better than these F1s suggest (AUC 0.73–0.75 for edit similarity and
    the drift rules), but 13 positives are too few to set a threshold.
  - S1 is also questionable here: it marks Evidence Act 65B "Deleted", though BSA 63 is its
    well-known successor.
- **Repeal detection** stays strong:

  | Family | F1 | AUC | v1 F1 |
  | --- | --- | --- | --- |
  | CrPC | 0.59 (precision 1.0) | 0.98 | 0.56 |
  | Evidence Act | 0.67 | 0.96 | 0.57 |

- **Hazard audit (E12).** 13 of 511 CrPC/BNSS provision ids and 8 of 168 Evidence/BSA ids
  pair sections at carry-over < 0.5. Read them with two source problems in mind:
  - **The CrPC source mixes versions.** It lacks sections 41A–41D, 55A, 60A, 195A and 437A
    (the 2008–2010 amendments), and holds pre-2009 text for at least §41 (arrest without warrant)
    and §54 (medical examination). BNSS 35 and 53 carry over the *amended* CrPC text. So their
    low carry-over (0.40 and 0.23) mostly measures the outdated source, not the BNSS.
  - **The procedure crosswalk is built by text similarity** (`ingest/align_codes.py`), and some
    pairs are wrong: CrPC 10, 15 and 19 merged into BNSS 13; CrPC 57 and 58 into BNSS 58. This
    also makes "v1 relation" a similarity baseline rather than an independent one.
  - Until a current CrPC text and a checked crosswalk are used, treat the procedure hazards as
    leads for review, not findings.

`make drift-eval` writes these as `results/tables/drift_procedure_*.csv` and
`results/figures/drift_families.png`.

## 6. Limitations

- **Old-code source versions (procedure).** The CrPC text in the MIT checkout is partly
  pre-2009 (Section 5). Drift on amended sections is overstated. A current consolidated CrPC
  (India Code) would fix this; the loader takes any file in the same format.

- **IPC source text.** The MIT IPC text omits some Explanations and illustrations and truncates
  a few sections (e.g. s.300). We detect the worst cases (*unverified*: 4 sections) and prefer
  IL-PCSR's IPC text when the corpus is built. Some false "substantive" calls remain.
- **Gold labels are coarse.** S1 is section-level Retained/Change; S2 is our rule-based reading of
  free-text notes. Neither labels carry-over directly, so carry-over is validated only indirectly
  (E11f, AUC).
- **Carry-over weights are hand-set.** They are principled but not learned. A small set of
  lawyer-graded (precedent, BNS section) pairs would let us fit and test them.
- **Citation relevance ≠ legal applicability.** IL-PCSR's gold cannot reward drift-aware ranking;
  E13 therefore reports stale@10 next to MAP.

## 7. Future work

- Learn the carry-over weights from graded pairs; give the mental-state reading a learned weight.
- Procedure and evidence (Section 5): a better crosswalk than text-similarity alignment, and
  graded labels with more positives for the Evidence Act.
- Zone-aware transfer: down-weight only a precedent's sentencing paragraphs when only the
  punishment changed.
- A user study with law students: does the "What changed" view reduce wrong-code citations?

## 8. Files

| File | Role |
| --- | --- |
| `src/kanoon_bridge/drift/provisions.py` | IPC/BNS texts, clause splitting, cross-reference renumbering |
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

## 9. Sources

- Bare-act text:
  - BNS via bns-study-platform (PSKprem); only `number`, `title` and `bareActText` are used to
    build reports. Its `changeNote` and `punishment` fields are used only as evaluation labels.
  - IPC via GSMS-B/indian-legal-mcp (MIT).
- Neither raw source is committed.
- Related ideas:
  - legal change and versioning of legislation (consolidated-law diffing, e.g. the EU's
    EUR-Lex consolidation);
  - containment as a near-duplicate measure (Broder, SEQUENCES 1997);
  - McNemar's test for paired classifiers (Dietterich, *Neural Computation* 1998);
  - Cohen's κ (1960).
