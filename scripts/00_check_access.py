"""Hour-0 check, and a progress board for the rest of the hackathon.  [working]

    python scripts/00_check_access.py

Prints PASS / MISSING for: config, data files, optional packages; then, per owner, how
many TODO stubs are still unimplemented in their modules.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "kanoon_bridge"


def line(ok: bool, label: str, detail: str = "") -> None:
    mark = "PASS   " if ok else "MISSING"
    print(f"  [{mark}] {label}{('  - ' + detail) if detail else ''}")


def check_setup() -> None:
    print("Setup")
    try:
        from kanoon_bridge.config import load_config

        cfg = load_config()
        line(True, "package imports and configs/default.yaml loads")
    except Exception as err:  # noqa: BLE001
        line(False, "package import / config", str(err))
        sys.exit(1)

    print("\nData (data/raw is gitignored; download per README)")
    checks = [
        (ROOT / cfg.paths.ilpcsr_dir / "queries", "IL-PCSR export in data/raw/ilpcsr/ (scripts/00_fetch_data.py)"),
        (ROOT / cfg.paths.bns_dir / "data" / "sections", "BNS sections in data/raw/bns-study-platform/ (00_fetch_data.py)"),
        (ROOT / cfg.paths.crosswalk_pdf, "optional: government crosswalk PDF for --compare"),
        (ROOT / cfg.paths.crosswalk, "data/crosswalk/ipc_bns.csv"),
        (ROOT / cfg.paths.offence_ids, "data/crosswalk/offence_ids.csv"),
        (ROOT / cfg.paths.lexicon_hinglish, "Hinglish legal lexicon"),
        (ROOT / cfg.paths.court_to_states, "court -> states table"),
        (ROOT / cfg.paths.docs, "processed corpus (after make data)"),
        (ROOT / cfg.paths.index_dir / "precedents_zone.pkl", "built index (after make index)"),
    ]
    for path, label in checks:
        exists = path.exists() and (path.is_file() or any(path.iterdir()))
        line(exists, label)

    print("\nOptional packages")
    for mod, why in [("datasets", "load IL-PCSR from Hugging Face"), ("sentence_transformers", "dense channel"),
                     ("streamlit", "demo app"), ("anthropic", "RAG layer")]:
        line(importlib.util.find_spec(mod) is not None, mod, why)


def check_progress() -> None:
    print("\nImplementation progress (TODO stubs left, by owner)")
    todo_re = re.compile(r'NotImplementedError\("TODO\((\w+)')
    plain_todo = re.compile(r'NotImplementedError\("TODO:')
    by_owner: dict[str, list[str]] = {}
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        text = path.read_text(encoding="utf-8")
        for owner in todo_re.findall(text):
            by_owner.setdefault(owner.split(",")[0], []).append(rel)
        for _ in plain_todo.findall(text):
            by_owner.setdefault("RAG/any", []).append(rel)
    if not by_owner:
        print("  all stubs implemented")
    for owner in sorted(by_owner):
        files = by_owner[owner]
        counts: dict[str, int] = {}
        for f in files:
            counts[f] = counts.get(f, 0) + 1
        summary = ", ".join(f"{f} ({n})" for f, n in counts.items())
        print(f"  {owner:8s} {len(files):3d} left: {summary}")


if __name__ == "__main__":
    check_setup()
    check_progress()
