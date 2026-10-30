"""E19: written facts -> charged sections, on real case data.  [owner: Gaurav — working]

E7, E9 and E14 ask in section-heading words. Here the query is a court's own statement of the
facts and the gold is the sections the accused was charged under, so heading matching cannot
carry the score.

Sources (each optional; the set is built from whichever is present):
    bail    IndianBailJudgments-1200 (Deshmukh et al. 2025, CC-BY-4.0; HF
            SnehaDeshmukh/IndianBailJudgments-1200): `facts` -> `ipc_sections`. 1,200 bail orders,
            2005-2025; 217 decided after 1 July 2024, whose charges are still IPC (the incidents
            predate the BNS).
    ilsi    IL-TUR's statute identification (Paul et al., ACL 2024; HF Exploration-Lab/IL-TUR,
            config lsi, gated, CC-BY-NC-SA-4.0): facts -> IPC sections (100 target sections).

Scoring is code-blind: a retrieved statute counts for a gold IPC section if it is that section or
its BNS counterpart (crosswalk). Sub-sections are compared at section level; gold sections of
other Acts (POCSO, NDPS, ...) are dropped, and a row with no IPC section left is skipped.

    R@5, R@10      share of the gold sections retrieved in the top 5 / 10
    MAP            over the gold sections
    hit@1          the top statute is one of the gold sections

Query length: the first `words` words of the facts (default 60), a typed-question proxy; the full
facts text is the `--words 0` variant.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict

from kanoon_bridge.config import project_path
from kanoon_bridge.schema import Query

BAIL = "data/raw/bail/indian_bail_judgments.json"
ILSI_DIR = "data/raw/il-tur/lsi"


def _sections(raw) -> list[str]:
    items = ast.literal_eval(raw) if isinstance(raw, str) and raw.strip().startswith("[") else raw
    if isinstance(items, str):
        items = re.split(r"[,;/]", items)
    out = []
    for s in items or []:
        m = re.match(r"\s*(?:section\s+)?(\d{1,3}[a-z]{0,2})", str(s), re.I)
        if m:
            ref = f"ipc:{m.group(1).lower()}"
            if ref not in out:
                out.append(ref)
    return out


def _head(text: str, words: int) -> str:
    w = (text or "").split()
    return " ".join(w[:words]) if words else " ".join(w)


def load_bail(words: int = 60, limit: int | None = None) -> list[dict]:
    path = project_path(BAIL)
    if not path.exists():
        return []
    rows = []
    for r in json.load(open(path, encoding="utf-8")):
        gold = _sections(r.get("ipc_sections"))
        if gold and r.get("facts"):
            rows.append({"id": f"bail-{r['case_id']}", "text": _head(r["facts"], words), "gold": gold,
                         "date": r.get("date", ""), "source": "bail"})
    return rows[:limit] if limit else rows


def load_ilsi(words: int = 60, limit: int | None = None, split: str = "test") -> list[dict]:
    """IL-TUR lsi parquet (columns as published: id, text / facts, labels)."""
    d = project_path(ILSI_DIR)
    files = sorted(d.glob(f"{split}-*.parquet")) if d.exists() else []
    if not files:
        return []
    import pyarrow.parquet as pq

    rows = []
    for f in files:
        for r in pq.read_table(f).to_pylist():
            text = r.get("text") or r.get("facts") or ""
            if isinstance(text, list):
                text = " ".join(text)
            gold = _sections([re.sub(r"(?i)^section\s+", "", str(x)) for x in (r.get("labels") or [])])
            if gold and text:
                rows.append({"id": f"ilsi-{r.get('id')}", "text": _head(text, words), "gold": gold, "source": "ilsi"})
    return rows[:limit] if limit else rows


def _base(ref: str) -> str:
    return re.sub(r"\(.*$", "", ref)


def score(rows: list[dict], run, engine, per_query: dict | None = None) -> dict[str, float]:
    """`run(Query) -> result with .statutes`; code-blind scoring (module docstring). `per_query`,
    if given, is filled with {metric: {row id: value}} for paired significance tests."""
    from kanoon_bridge.eval.metrics import average_precision
    from kanoon_bridge.eval.open_eval import statute_ref

    norm = engine.analyzer.text_res.normalizer
    acc: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        gold = {}
        for g in row["gold"]:
            g = _base(g)
            for alias in [g, *(_base(e) for e in norm.equivalents(g))]:
                gold.setdefault(alias, g)
        if not gold:
            continue
        q = Query(row["text"])
        q.query_id = row["id"]
        res = run(q)
        seen, ranking = set(), []
        for h in res.statutes:
            g = gold.get(_base(statute_ref(engine, h.doc_id)))
            ranking.append(g if g and g not in seen else f"x{len(ranking)}")
            if g:
                seen.add(g)
        rels = {g: 1 for g in set(gold.values())}
        acc["R@5"].append(len(set(ranking[:5]) & set(rels)) / len(rels))
        acc["R@10"].append(len(set(ranking[:10]) & set(rels)) / len(rels))
        acc["MAP"].append(average_precision(ranking, rels))
        acc["hit@1"].append(float(bool(ranking) and ranking[0] in rels))
        if per_query is not None:
            for k in ("MAP", "hit@1"):
                per_query.setdefault(k, {})[row["id"]] = acc[k][-1]
    return {**{k: round(sum(v) / len(v), 4) for k, v in acc.items()}, "n": len(acc["MAP"])}


__all__ = ["load_bail", "load_ilsi", "score"]
