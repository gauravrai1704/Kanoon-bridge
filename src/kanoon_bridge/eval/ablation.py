"""Ablation ladder and language ablation.  [owner: D]

Ladder (configs/eval.yaml: ablation_ladder), one bar per step in the report chart:
    bm25 -> +zones -> +bridge -> +authority -> +dense -> +qpp -> +jurisdiction

Language ablation (E4): none -> +transliteration -> +phonetic -> +dense, and stemming on/off.

Output: results/tables/ablation_<set>.csv with one row per step and one column per metric.
"""

from __future__ import annotations


def run_ladder(test_set: str) -> list[dict]:
    """TODO(D): for each ladder step, SearchOptions.from_dict(step) -> run_eval.run_queries ->
    metrics.evaluate; return rows and write the CSV. Tune nothing on test (rule 3)."""
    raise NotImplementedError("TODO(D): ablation ladder")


def run_language_ablation(test_set: str = "e4_multilingual") -> list[dict]:
    """TODO(D, with C): toggle analyzer steps (transliterate, phonetic, dense) and stemming;
    report P@5 per query language (en / hi / hinglish)."""
    raise NotImplementedError("TODO(D): language ablation")
