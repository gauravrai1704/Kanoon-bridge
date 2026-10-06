# Report

The report is a shared Claude doc: **Kanoon-Bridge: Hackathon Report**
(https://claude.ai/code/artifact/85fe34dd-d427-4876-8b28-cc888214ecee). Edit it there and
export it as PDF or Word from the doc's menu.

The video script is a second doc: **Kanoon-Bridge: Demo Video Script**
(https://claude.ai/code/artifact/1e5dfc11-f9b5-4cf6-86e8-dd2707941b18). A Markdown copy is in
`VIDEO_SCRIPT.md` in this folder.

Files used by the report:

| File | What |
| --- | --- |
| `pipeline.png` / `pipeline.svg` | Figure 1, the pipeline diagram |
| `screens/*.png` | App screenshots (Figures 2–3 and spares for the video) |
| `figures/fig_main.png` | Figure 4, headline metric per test set |
| `figures/fig_ablation.png` | Figure 5, E1 ablation ladder |
| `figures/fig_e3.png` | Figure 6, IPC vs BNS wording (E3) |
| `figures/fig_language.png` | Figure 7, E4 language ablation |

Regenerate the figures after a new evaluation run with `python scripts/10_report_figures.py`.
All numbers come from `results/tables/*.csv` (written by `make eval`).
