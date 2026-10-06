"""Relevance judgments: load, save, and build rule-based grades.  [owner: D — working]

File format for our hand-built sets (data/queries/qrels/*.tsv), tab-separated, header row:
    query_id    doc_id    grade    judge

`merge_judges` combines two judges' grades (after discussion, disagreements are settled
by hand; this just averages and rounds as a first pass).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from kanoon_bridge.index.facets import DocMeta
from kanoon_bridge.rank.authority import binding_status

Qrels = dict[str, dict[str, int]]


def load_tsv(path: str | Path, judge: str | None = None) -> Qrels:
    """Load a qrels TSV; optionally only one judge's rows."""
    out: Qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"):
            if judge and row.get("judge") != judge:
                continue
            out[row["query_id"]][row["doc_id"]] = int(row["grade"])
    return dict(out)


def load_by_judge(path: str | Path) -> dict[str, Qrels]:
    """judge -> qrels (for eval/agreement.py)."""
    by: dict[str, Qrels] = defaultdict(lambda: defaultdict(dict))
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"):
            by[row["judge"]][row["query_id"]][row["doc_id"]] = int(row["grade"])
    return {j: dict(q) for j, q in by.items()}


def save_tsv(qrels: Qrels, path: str | Path, judge: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query_id", "doc_id", "grade", "judge"])
        for q, rels in qrels.items():
            for d, g in rels.items():
                w.writerow([q, d, g, judge])


def merge_judges(a: Qrels, b: Qrels) -> Qrels:
    out: Qrels = {}
    for q in set(a) | set(b):
        docs = set(a.get(q, {})) | set(b.get(q, {}))
        out[q] = {d: round((a.get(q, {}).get(d, 0) + b.get(q, {}).get(d, 0)) / 2) for d in docs}
    return out


def jurisdiction_grades(topical: Qrels, metas: dict[str, DocMeta], query_states: dict[str, str]) -> Qrels:
    """E6 grades: topically relevant AND binding = 2, relevant AND persuasive = 1, else 0."""
    out: Qrels = {}
    for q, rels in topical.items():
        state = query_states.get(q)
        out[q] = {}
        for d, g in rels.items():
            if g <= 0 or d not in metas:
                out[q][d] = 0
                continue
            out[q][d] = 2 if binding_status(metas[d], state) == "binding" else 1
    return out
