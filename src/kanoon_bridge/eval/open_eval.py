"""Statute-level evaluation on the open corpus (no IL-PCSR needed).  [owner: Gaurav — working]

E2, E7 and E9 are judged by section REFERENCE ("ipc:84", "bnss:103") and offence id, not by
IL-PCSR document id, so they can be scored on the open statute corpus (make open: all six
codes from India Code + the BNS). Per question, over the statute ranking:

    E2 collision   expected@1   the top statute is the expected section ("expected" in the row)
                   wrong_hit@10 a colliding wrong section ("wrong_refs") is in the top 10
    E7 / E9        code@1       the top statute belongs to the code in force on the incident date
                   offence@1    the top statute carries the row's offence / provision id
                   both@1       code@1 and offence@1 together: the right provision of the right code

Any system with `run(query) -> result with .statutes` can be scored: the core engine (wrapped),
the research agent v1 and v2.
"""

from __future__ import annotations

import json
from collections import defaultdict

from kanoon_bridge.config import project_path
from kanoon_bridge.schema import Query

SETS = {
    "e2_collision": "data/queries/e2_collision.jsonl",
    "e7_temporal": "data/queries/e7_temporal.jsonl",
    "e9_procedure_temporal": "data/queries/e9_procedure_temporal.jsonl",
}


def load_rows(name: str, limit: int | None = None) -> list[dict]:
    with open(project_path(SETS[name]), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def statute_ref(engine, doc_id: str) -> str:
    terms = getattr(getattr(engine, "bridge", None), "statute_terms", {}) or {}
    sec = next((t for t in terms.get(doc_id, []) if t.startswith("sec:")), None)
    return sec[4:] if sec else doc_id


def score_set(name: str, run, engine, limit: int | None = None) -> dict[str, float]:
    """`run(Query) -> result`; returns the set's metrics (see module docstring)."""
    rows = load_rows(name, limit)
    norm = engine.analyzer.text_res.normalizer
    acc: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        q = Query.from_dict(row)
        q.query_id = row.get("id")
        res = run(q)
        refs = [statute_ref(engine, h.doc_id) for h in res.statutes]
        top = refs[0] if refs else ""
        if name.startswith("e2"):
            if row.get("expected"):
                acc["expected@1"].append(float(top == row["expected"]))
            if row.get("wrong_refs"):
                acc["wrong_hit@10"].append(float(any(r in row["wrong_refs"] for r in refs[:10])))
        else:
            expected = row.get("expected_code")
            code_ok = bool(top) and top.split(":")[0] == expected
            offs = set(norm.offences_for(top)) if top else set()
            off_ok = row.get("offence") in offs
            acc["code@1"].append(float(code_ok))
            acc["offence@1"].append(float(off_ok))
            acc["both@1"].append(float(code_ok and off_ok))
        acc["searches"].append(float(getattr(res, "n_searches", 1)))
    return {k: round(sum(v) / len(v), 4) for k, v in acc.items() if v} | {"n": len(rows)}


__all__ = ["SETS", "load_rows", "score_set", "statute_ref"]
