"""Ablation ladder and language ablation.  [owner: Gaurav — working]

Ladder (configs/eval.yaml: ablation_ladder), one bar per step in the report chart:
    bm25 -> +zones -> +code_filter -> +bridge -> +authority -> +dense -> +qpp -> +jurisdiction
Steps whose component is not available (dense channel off, no authority file) behave like the
previous step; the CSV says so in the `note` column.

Language ablation (E4): analyzer steps switched on one at a time
    none -> +translit -> +lexicon -> +phonetic -> +dense, P@5 per query language.

Outputs: results/tables/ablation_<set>.csv, results/tables/language_e4.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

from kanoon_bridge.config import Config, load_config, project_path


def write_table(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def _note(engine, step: dict) -> str:
    missing = []
    if step.get("dense") and engine.dense is None:
        missing.append("dense off")
    if step.get("authority") and engine.authority is None:
        missing.append("no authority file")
    if step.get("bridge") and engine.bridge is None:
        missing.append("no statute terms")
    return "; ".join(missing)


def run_ladder(test_set: str = "e1_ilpcsr", engine=None, limit: int | None = None,
               ev: Config | None = None) -> list[dict]:
    """Every ladder step on one test set -> rows (also written to CSV). Tunes nothing on test."""
    from kanoon_bridge.eval.run_eval import evaluate_set, load_test_set
    from kanoon_bridge.search import SearchEngine, SearchOptions

    ev = ev or load_config("eval.yaml")
    engine = engine or SearchEngine.load()
    ts = load_test_set(test_set, ev=ev, limit=limit)
    rows = []
    for step in ev.ablation_ladder:
        scores = evaluate_set(engine, ts, SearchOptions.from_dict(step), step["name"], ev,
                              project_path(ev.outputs.runs) / "ablation")
        rows.append({"step": step["name"], **{k: round(v, 4) for k, v in scores.items()}, "note": _note(engine, step)})
        print(f"  {test_set} {step['name']:14s} MAP={scores.get('MAP', 0):.4f} F1@k={scores.get('F1@k_val', scores.get('F1@10', 0)):.4f}")
    write_table(rows, project_path(ev.outputs.tables) / f"ablation_{test_set}.csv")
    return rows


def run_language_ablation(test_set: str = "e4_multilingual", engine=None, ev: Config | None = None) -> list[dict]:
    """Analyzer steps on/off for E4; P@5 overall and per language."""
    from kanoon_bridge.eval.run_eval import evaluate_set, load_test_set
    from kanoon_bridge.search import SearchEngine, SearchOptions

    ev = ev or load_config("eval.yaml")
    engine = engine or SearchEngine.load()
    ts = load_test_set(test_set, ev=ev)
    if not ts.judged:
        print(f"  {test_set}: no judged queries yet - skipped")
        return []
    an = engine.analyzer
    saved = (dict(an.steps), an.phonetic)
    rows = []
    try:
        for step in ev.language_ablation:
            an.steps = {"transliterate": step["transliterate"], "lexicon": step["lexicon"], "phonetic": step["phonetic"]}
            an.phonetic = saved[1] if step["phonetic"] else None
            scores = evaluate_set(engine, ts, SearchOptions(dense=step["dense"]), f"lang_{step['name']}", ev)
            rows.append({"step": step["name"], **{k: round(v, 4) for k, v in scores.items()},
                         "note": "dense off" if step["dense"] and engine.dense is None else ""})
    finally:
        an.steps, an.phonetic = saved
    write_table(rows, project_path(ev.outputs.tables) / "language_e4.csv")
    return rows
