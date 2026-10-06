"""Build data/crosswalk/ipc_bns.csv and offence_ids.csv from the government table.  [owner: B]

Source PDF (government copy, put in data/raw/crosswalk/):
https://www.keralaprisons.gov.in/userfiles/act-and-rules/comparison_summary_BNS_to_IPC.pdf

Output formats (see data/crosswalk/README.md):
    ipc_bns.csv       ipc_section, bns_section, relation, note
    offence_ids.csv   offence_id, label, ipc_sections, bns_sections, keywords

Relations: same | modified | split | merge | new | dropped. One IPC section can map to
several BNS sub-sections and vice versa — keep one row per pair, never collapse them.

Plan:
  1. extract tables with pdfplumber (page.extract_tables()); fall back to regex on text
  2. write ipc_bns.csv, then hand-check the 80 most-cited IPC sections in IL-PCSR
  3. group pairs into offences -> offence_ids.csv (connected components of the bipartite
     IPC-BNS graph give a good first grouping; name each by its BNS heading)
  4. add keywords per offence (used by text/collision.py)
"""

from __future__ import annotations

from pathlib import Path


def extract_rows(pdf_path: str | Path) -> list[dict]:
    """TODO(B): PDF -> list of {ipc_section, bns_section, relation, note}."""
    raise NotImplementedError("TODO(B): extract crosswalk rows from the PDF")


def write_crosswalk(rows: list[dict], out_path: str | Path) -> None:
    """TODO(B): write ipc_bns.csv (sorted by ipc_section numerically)."""
    raise NotImplementedError("TODO(B): write ipc_bns.csv")


def build_offence_ids(rows: list[dict], out_path: str | Path) -> None:
    """TODO(B): group mappings into canonical offences and write offence_ids.csv."""
    raise NotImplementedError("TODO(B): build offence_ids.csv")


if __name__ == "__main__":
    import sys

    rows = extract_rows(sys.argv[1] if len(sys.argv) > 1 else "data/raw/crosswalk/comparison_summary_BNS_to_IPC.pdf")
    write_crosswalk(rows, "data/crosswalk/ipc_bns.csv")
    build_offence_ids(rows, "data/crosswalk/offence_ids.csv")
