"""Copy the best validation setting of a tuning table into configs/default.yaml.  [owner: Gaurav]

    python scripts/apply_tuning.py short        # fusion_short.trigram / dense_max from tuning_short_val.csv
    python scripts/apply_tuning.py dense-on     # dense.enabled: true (after scripts/04_encode_dense.py)

Edits only the numbers on the matching lines (comments and layout stay), and prints what changed.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def best(table: str) -> dict:
    with open(ROOT / "results" / "tables" / table, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return max(rows, key=lambda r: float(r["MAP_val"]))


def set_value(text: str, block: str, key: str, value: float) -> str:
    pat = re.compile(rf"(^{block}:\s*(?:#.*)?\n(?:(?:[ \t]+.*|\s*)\n)*?[ \t]+{key}:\s*)([0-9.]+)", re.M)
    new, n = pat.subn(lambda m: f"{m.group(1)}{value}", text, count=1)
    if not n:
        raise SystemExit(f"{block}.{key} not found in configs/default.yaml")
    return new


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "short"
    cfg = ROOT / "configs" / "default.yaml"
    text = cfg.read_text(encoding="utf-8")
    if which == "dense-on":
        text, n = re.subn(r"(^dense:\s*\n[ \t]+enabled:\s*)false", r"\1true", text, count=1, flags=re.M)
        print("dense.enabled: true" if n else "dense.enabled was already true")
    elif which == "short":
        b = best("tuning_short_val.csv")
        text = set_value(text, "fusion_short", "trigram", float(b["trigram"]))
        text = set_value(text, "fusion_short", "dense_max", float(b["dense_max"]))
        print(f"fusion_short: trigram={b['trigram']} dense_max={b['dense_max']} (val MAP {b['MAP_val']})")
    cfg.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
