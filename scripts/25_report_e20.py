"""Turn E20 results (results/tables/misgrounding_<tag>.csv) into a Markdown and a LaTeX table.  [owner: Gaurav — working]

    python scripts/25_report_e20.py                       # every misgrounding_*.csv found
    python scripts/25_report_e20.py llama3_2_3b llama3_1_8b
    python scripts/25_report_e20.py --rescore llama3_2_3b    # re-score saved answers with the current
                                                             # scorer (no LLM calls) + paired tests

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


def rescore(tag: str) -> None:
    """Re-score misgrounding_<tag>_answers.jsonl with the current scorer, rewrite the CSV, and test
    the conditions pairwise on the same questions (exact McNemar on right@1 and misgrounded@1,
    Holm-corrected)."""
    import json
    from math import comb

    from kanoon_bridge.config import load_config
    from kanoon_bridge.eval.misgrounding import rescore as _rescore
    from kanoon_bridge.eval.significance import holm
    from kanoon_bridge.text.version_norm import VersionNormalizer

    tables = project_path("results/tables")
    table, rows = _rescore(tables / f"misgrounding_{tag}_answers.jsonl", VersionNormalizer.load(load_config()))
    with open(tables / f"misgrounding_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in table for k in r)))
        w.writeheader()
        w.writerows(table)
    with open(tables / f"misgrounding_{tag}_answers.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {}
    for r in rows:
        by[(r["condition"], r["id"])] = r

    def mcnemar(b: int, c: int) -> float:
        n = b + c
        if n == 0:
            return 1.0
        k = min(b, c)
        return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

    ids = sorted({r["id"] for r in rows})
    pairs = [("closed-book", "naive RAG"), ("naive RAG", "RAG v1"), ("naive RAG", "RAG v2"), ("RAG v1", "RAG v2"),
             ("RAG v2", "structured")]
    sig = []
    for a, b in pairs:
        for metric in ("right@1", "misgrounded@1"):
            va = [by[(a, i)][metric] for i in ids]
            vb = [by[(b, i)][metric] for i in ids]
            only_a = sum(x > y for x, y in zip(va, vb))
            only_b = sum(y > x for x, y in zip(va, vb))
            sig.append({"a": a, "b": b, "metric": metric, "n": len(ids), "mean_a": round(sum(va) / len(ids), 4),
                        "mean_b": round(sum(vb) / len(ids), 4), "only_a": only_a, "only_b": only_b,
                        "p": round(mcnemar(only_a, only_b), 6)})
    for row, p in zip(sig, holm([r["p"] for r in sig])):
        row["p_holm"] = round(p, 6)
    with open(tables / f"misgrounding_{tag}_significance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sig[0]))
        w.writeheader()
        w.writerows(sig)
    for r in sig:
        print(f"{r['a']:12s} -> {r['b']:12s} {r['metric']:14s} {r['mean_a']:.3f} -> {r['mean_b']:.3f}  "
              f"({r['only_a']} vs {r['only_b']} discordant)  p_holm {r['p_holm']}")


def main() -> None:
    if sys.argv[1:2] == ["--rescore"]:
        for tag in sys.argv[2:]:
            rescore(tag)
        sys.argv = sys.argv[:1] + sys.argv[2:]
    tables = project_path("results/tables")
    tags = sys.argv[1:] or sorted(p.stem.removeprefix("misgrounding_") for p in tables.glob("misgrounding_*.csv")
                                  if not p.stem.endswith(("_answers", "_significance")))
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
