# Annotation guidelines (Kanoon-Bridge)

Thank you for helping. Three sheets, about 4–5 hours in total. Open each `.tsv` in Excel or
Google Sheets. Fill only the columns starting with `LABEL_`. The `system_` columns at the right
are what our system said: **cover or hide them until you have labelled the row**, so they do not
influence you. Write anything unusual in `LABEL_notes`.

If two people annotate, each works on their own copy: `drift.<yourname>.tsv`, and so on.

---

## 1. `drift.tsv` — did the law change from the IPC to the BNS? (111 rows, ~2.5 h)

Each row is one BNS section with the IPC section(s) it replaced, text side by side
(India Code's consolidated text). Rows with no IPC text are new BNS provisions.

| Column | Values | Meaning |
| --- | --- | --- |
| `LABEL_changed` | `yes` / `no` | Is there any change a lawyer would care about? Renumbering, "Code" → "Sanhita", or updated cross-references alone are **no**. |
| `LABEL_kind` | `none`, `cosmetic`, `punishment`, `elements`, `structure`, `other` | The main kind of change. `punishment`: terms, fines, community service. `elements`: what must be proved (intent, knowledge, thresholds, new conduct covered). `structure`: sections merged or split, explanations moved, with the substance kept. |
| `LABEL_precedent_use` | `0`–`3` | Can a judgment decided under the IPC section be relied on for an offence under the BNS section? `3` fully, `2` mostly (check one point), `1` only in part, `0` not at all (new offence or a different one). |

## 2. `treatment.tsv` — was this precedent overruled? (48 rows, ~1 h)

Each row is a sentence from a judgment (`citing_case`) that, according to our system, says
something about another judgment (`cited_case`).

| Column | Values | Meaning |
| --- | --- | --- |
| `LABEL_target_correct` | `yes` / `no` | Is the sentence really about `cited_case` (not another case it mentions)? |
| `LABEL_treatment` | `overruled`, `doubted`, `followed`, `other` | What the sentence does to `cited_case`. `overruled` includes "per incuriam" and "no longer good law"; `doubted` includes "disapproved"; `followed` includes "relied on", "approved". Use `other` when it is only mentioned or distinguished. |

If you know from outside the sentence that a case was overruled, write it in `LABEL_notes`.
That helps us measure what the system misses.

## 3. `answers.tsv` — are the system's answers right? (60 rows, ~1 h)

Each row is a question (with the incident date) and the answer the system gives.

| Column | Values | Meaning |
| --- | --- | --- |
| `LABEL_law_correct` | `yes` / `partly` / `no` | Is the section named the one that applies to an incident on that date? (IPC before 1 July 2024, BNS on or after.) |
| `LABEL_punishment_correct` | `yes` / `no` / `n.a.` | Is the punishment stated complete and correct for that section? `n.a.` if the answer states none. |
| `LABEL_misleading` | `yes` / `no` | Would anything in the answer mislead a non-lawyer? For example, a quoted precedent that is beside the point. |

---

When done, send the files back. `python scripts/22_annotation_kit.py score` computes agreement
with the system (and between annotators, if there are two), written to
`results/tables/annotation_scores.csv`. Your name goes in the acknowledgements of the paper.
