"""Typo robustness: known-item statute search with simulated misspellings.  [owner: Gaurav — working]

Known-item queries are simulated from the collection itself (Azzopardi, de Rijke & Balog,
"Building simulated queries for known-item topics", SIGIR 2007): the query is a statute's own
title ("Punishment for murder"), the single relevant document is that statute. Typos are then
injected with the four Damerau error types (delete, insert, substitute with a keyboard
neighbour, swap adjacent letters), at least one per query, in words of 5+ letters.

Conditions (same queries):
    clean               original titles, spelling correction on
    typos, no spelling  corrupted titles, analyzer step "spelling" off
    typos + spelling    corrupted titles, spelling correction on

Metrics: MRR@10, Success@1, Success@10 of the target statute. Output: results/tables/typos.csv,
plus results/tables/typo_queries.jsonl (each corrupted query and what the speller did).
"""

from __future__ import annotations

import json
import random
import statistics

from kanoon_bridge.config import Config, load_config, project_path

_KEYBOARD = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def _neighbours(ch: str) -> str:
    for row in _KEYBOARD:
        i = row.find(ch)
        if i >= 0:
            return row[max(0, i - 1):i] + row[i + 1:i + 2]
    return "aeiou"


def corrupt_word(word: str, rng: random.Random) -> str:
    """One Damerau error at a random inner position (the first letter is kept, as in most typos)."""
    if len(word) < 3:
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.choice(("delete", "insert", "substitute", "transpose"))
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "insert":
        return word[:i] + rng.choice(_neighbours(word[i])) + word[i:]
    if op == "substitute":
        return word[:i] + rng.choice(_neighbours(word[i]) or "e") + word[i + 1:]
    return word[:i] + word[i + 1] + word[i] + word[i + 2:] if i + 1 < len(word) else word


def corrupt(text: str, rng: random.Random, rate: float = 1.0) -> str:
    words = text.split()
    eligible = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not eligible:
        return text
    n = max(1, min(len(eligible), round(rate * len(eligible) / 2)))
    for i in rng.sample(eligible, n):
        bad = corrupt_word(words[i].lower(), rng)
        words[i] = bad if bad != words[i].lower() else corrupt_word(bad, rng)
    return " ".join(words)


def _items(engine, docs, n: int, seed: int) -> list[tuple[str, str, str]]:
    """(statute doc_id, title, incident date in its code's era)."""
    stop = engine.analyzer.text_res.stopwords
    out = []
    for d in docs.values():
        if d.doc_type.value != "statute" or not d.title or d.code.value not in ("ipc", "bns"):
            continue
        content = [w for w in d.title.lower().split() if w.isalpha() and w not in stop]
        if len(content) >= 2 and any(len(w) >= 5 for w in content):
            out.append((d.doc_id, d.title, "2025-01-03" if d.code.value == "bns" else "2020-01-03"))
    out.sort()
    random.Random(seed).shuffle(out)
    return out[:n]


def run(engine=None, docs=None, ev: Config | None = None, n_items: int | None = None) -> list[dict]:
    from kanoon_bridge.eval.ablation import write_table
    from kanoon_bridge.schema import Query
    from kanoon_bridge.search import SearchEngine, SearchOptions

    ev = ev or load_config("eval.yaml")
    tcfg = ev.get("typos", {}) or {}
    engine = engine or SearchEngine.load()
    if docs is None:
        from kanoon_bridge.index.docstore import DocStore

        docs = DocStore.load(engine.cfg)
    items = _items(engine, docs, n_items or int(tcfg.get("n_items", 300)), int(tcfg.get("seed", 13)))
    if not items:
        print("  typos: no statutes with usable titles - skipped")
        return []
    rng = random.Random(int(tcfg.get("seed", 13)))
    corrupted = [corrupt(title, rng, float(tcfg.get("rate", 1.0))) for _, title, _ in items]

    an = engine.analyzer
    saved = dict(an.steps)
    opt = SearchOptions(top_k=10)
    rows, log = [], []
    try:
        for name, texts, spelling in (("clean", [t for _, t, _ in items], True),
                                      ("typos, no spelling", corrupted, False),
                                      ("typos + spelling", corrupted, True)):
            an.steps = {**saved, "spelling": spelling}
            rr, s1, s10 = [], [], []
            for (doc_id, title, when), text in zip(items, texts):
                res = engine.search(Query(text, incident_date=when), opt)
                ids = [h.doc_id for h in res.statutes]
                rank = ids.index(doc_id) + 1 if doc_id in ids else 0
                rr.append(1.0 / rank if rank else 0.0)
                s1.append(1.0 if rank == 1 else 0.0)
                s10.append(1.0 if rank else 0.0)
                if name == "typos + spelling":
                    log.append({"doc": doc_id, "title": title, "query": text, "rank": rank,
                                "did_you_mean": res.query.suggestion,
                                "corrections": [(c.word, c.display) for c in res.query.corrections]})
            rows.append({"condition": name, "queries": len(items), "MRR@10": round(statistics.fmean(rr), 4),
                         "Success@1": round(statistics.fmean(s1), 4), "Success@10": round(statistics.fmean(s10), 4)})
            print(f"  typos {name:20s} MRR@10={rows[-1]['MRR@10']:.4f}  S@1={rows[-1]['Success@1']:.4f}")
    finally:
        an.steps = saved
    tables = project_path(ev.outputs.tables)
    write_table(rows, tables / "typos.csv")
    with open(tables / "typo_queries.jsonl", "w", encoding="utf-8") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows
