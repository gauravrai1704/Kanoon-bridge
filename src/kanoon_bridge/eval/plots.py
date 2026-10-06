"""Charts for the report and video, saved to results/figures/.  [owner: D]

Planned figures (one finding per chart; title says the finding):
    ablation_e1.png       bar per ladder step, MAP and F1@k
    collision_e2.png      wrong-offence rate in top-10: baseline vs normalised
    language_e4.png       P@5 by query language, per language-ablation step
    jurisdiction_e6.png   share of binding precedent in top-5, g(d) vs g(d | state)
    efficiency.png        latency vs Recall@20 per scoring mode

Use matplotlib with a single consistent style; label units; no 3D, no pie charts.
"""

from __future__ import annotations

from pathlib import Path


def bar_chart(rows: list[dict], x: str, y: str, title: str, out: str | Path) -> Path:
    """TODO(D): simple labelled bar chart from table rows; value labels on bars."""
    raise NotImplementedError("TODO(D): bar chart")


def make_all() -> None:
    """TODO(D): read results/tables/*.csv and produce every figure listed above."""
    raise NotImplementedError("TODO(D): all report figures")
