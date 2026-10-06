# Crosswalk files (owner B)

Built by `src/kanoon_bridge/ingest/parse_crosswalk.py` from the government comparison table
(https://www.keralaprisons.gov.in/userfiles/act-and-rules/comparison_summary_BNS_to_IPC.pdf),
then hand-checked. The rows currently in these files are a **small seed for testing only** —
verify every row against the bare acts before relying on it, and replace with the full table.

Lines starting with `#` are comments and are skipped by the loaders.

## ipc_bns.csv

| column | meaning |
| --- | --- |
| `ipc_section` | IPC section, lower-case (`302`, `498a`, `304b`); empty if the BNS section is new |
| `bns_section` | BNS section incl. sub-section if relevant (`103(1)`); empty if dropped |
| `relation` | `same`, `modified`, `split`, `merge`, `new`, `dropped` |
| `note` | free text |

One row per (IPC, BNS) pair. A split IPC section has several rows.

## offence_ids.csv

| column | meaning |
| --- | --- |
| `offence_id` | canonical id with prefix, e.g. `off:murder` |
| `label` | human label |
| `ipc_sections` | `;`-separated IPC sections |
| `bns_sections` | `;`-separated BNS sections |
| `keywords` | `;`-separated words that signal this offence in text (used by `text/collision.py`) |

The famous collision: IPC 302 (murder) is BNS 103(1), while BNS 302 is the old IPC 298.
