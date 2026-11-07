"""Turn E20 results (results/tables/misgrounding_<tag>.csv) into a Markdown and a LaTeX table.  [owner: Gaurav — working]

    python scripts/25_report_e20.py                       # every misgrounding_*.csv found
    python scripts/25_report_e20.py llama3_2_3b llama3_1_8b

Prints, per model, right@1 and misgrounded@1 over all questions for each condition, for the
README and the paper (docs/paper_acm, section "LLM temporal misgrounding").
"""

from __future__ import annotations

import csv
import sys

from kanoon_bridge.config import project_path

ORDER = ["closed-book", "naive RAG", "RAG v1", "RAG v2", "structured"]


def load(tag: str) -> dict[str, dict]:
    path = project_path("results/tables") / f"misgrounding_{tag}.csv"
    with open(path, encoding="utf-8") as f:
        return {r["condition"]: r for r in csv.DictReader(f) if r["set"] == "all"}


def main() -> None:
    tables = project_path("results/tables")
    tags = sys.argv[1:] or sorted(p.stem.removeprefix("misgrounding_") for p in tables.glob("misgrounding_*.csv")
                                  if not p.stem.endswith("_answers"))
    if not tags:
        raise SystemExit("no results/tables/misgrounding_*.csv yet: run scripts/run_e20_ollama.sh first")
    data = {t: load(t) for t in tags}
    head = "| Condition | " + " | ".join(f"{t}: right@1 / misgrounded@1" for t in tags) + " |"
    print(head)
    print("| --- |" + " --- |" * len(tags))
    for c in ORDER:
        cells = [f"{float(data[t][c]['right@1']):.2f} / {float(data[t][c]['misgrounded@1']):.2f}" if c in data[t] else "–"
                 for t in tags]
        print(f"| {c} | " + " | ".join(cells) + " |")
    print("\n% LaTeX")
    print("\\begin{tabular}{l" + "cc" * len(tags) + "}\n\\toprule")
    print("Condition & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{t.replace('_', '.')}}}" for t in tags) + " \\\\")
    print(" & " + " & ".join("right@1 & misgr.@1" for _ in tags) + " \\\\\n\\midrule")
    for c in ORDER:
        cells = [f"{float(data[t][c]['right@1']):.2f} & {float(data[t][c]['misgrounded@1']):.2f}" if c in data[t] else "-- & --"
                 for t in tags]
        print(f"{c} & " + " & ".join(cells) + " \\\\")
    print("\\bottomrule\n\\end{tabular}")


if __name__ == "__main__":
    main()
