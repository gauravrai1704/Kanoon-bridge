"""Load the IL-PCSR corpus (IIT Kharagpur + IIT Kanpur, 2025).  [owner: A]

Source: https://huggingface.co/datasets/Exploration-Lab/IL-PCSR
        https://github.com/Exploration-Lab/IL-PCSR
Put the download in data/raw/ilpcsr/ (or let `datasets` cache it).

What the paper says the release contains (CHECK the real files in hour 0 and fix below):
  * 6,271 query judgments, split 8:1:1 train/val/test (5,021 / 627 / 627)
  * 3,183 precedent candidates, 936 statute candidates (Articles/Sections of 92 Central Acts)
  * gold = cases/statutes actually cited by the query; citations MASKED in query text with
    [SECTION], [ACT], [PRECEDENT], [ENTITY] placeholders; precedent texts are NOT masked
  * possibly per-paragraph rhetorical-role labels (used by their Para-GNN) -> zones

Outputs used by the rest of the code:
  load_queries(split)        -> list[Document]  (doc_type=QUERY_CASE, split set)
  load_precedents()          -> list[Document]  (doc_type=PRECEDENT)
  load_statute_candidates()  -> list[Document]  (doc_type=STATUTE, code from the Act name)
  load_qrels(split, target)  -> {query_id: {doc_id: 1}}  target = "precedent" | "statute"
"""

from __future__ import annotations

from pathlib import Path

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.schema import Document

HF_DATASET = "Exploration-Lab/IL-PCSR"


def raw_dir() -> Path:
    return project_path(load_config().paths.raw) / "ilpcsr"


def inspect() -> None:
    """Hour-0 helper: print the files/configs/columns that the release actually has.

    TODO(A): try `datasets.load_dataset(HF_DATASET)` (pip install -e ".[data]") and also list
    raw_dir(); print column names and one example of each split. Then fill in the loaders.
    """
    raise NotImplementedError("TODO(A): inspect IL-PCSR release format")


def load_queries(split: str) -> list[Document]:
    """TODO(A): query judgments for 'train' | 'val' | 'test' as Documents (paragraphs + zones if given)."""
    raise NotImplementedError("TODO(A): load IL-PCSR queries")


def load_precedents() -> list[Document]:
    """TODO(A): precedent candidates; keep their own cited statutes in Document.statutes_cited."""
    raise NotImplementedError("TODO(A): load IL-PCSR precedent pool")


def load_statute_candidates() -> list[Document]:
    """TODO(A): statute candidates; set code=IPC for IPC sections, section number, title."""
    raise NotImplementedError("TODO(A): load IL-PCSR statute pool")


def load_qrels(split: str, target: str = "precedent") -> dict[str, dict[str, int]]:
    """TODO(A): gold citations per query for the split. NEVER use test qrels for tuning (rule 3)."""
    raise NotImplementedError("TODO(A): load IL-PCSR qrels")
