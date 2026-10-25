"""Build the statutory drift table (drift/).  [owner: Gaurav — working]

Reads the BNS and IPC bare-act text (make fetch) and the committed crosswalk; writes
    data/crosswalk/drift_sections.csv   one row per BNS section (+ IPC sections with no successor)
    data/crosswalk/drift_pairs.csv      carry-over c(ipc -> bns) for every crosswalk pair
Both are committed, so the search engine and the app use drift without the raw data.

    python scripts/13_build_drift.py                 # IPC text: MIT checkout, upgraded with IL-PCSR text if built
    python scripts/13_build_drift.py --mit-only      # IPC text from the MIT checkout only (what we commit)
    python scripts/13_build_drift.py --show 152 106  # print the reports of some BNS sections
"""

from __future__ import annotations

import argparse
import time

from kanoon_bridge.config import load_config
from kanoon_bridge.drift.model import DriftAnalyzer, DriftTable
from kanoon_bridge.drift.provisions import load_bns, load_ipc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mit-only", action="store_true", help="ignore IL-PCSR IPC text even if docs.jsonl exists")
    ap.add_argument("--show", nargs="*", default=[], help="BNS sections to print")
    args = ap.parse_args()
    cfg = load_config()
    t0 = time.perf_counter()
    ipc = load_ipc(cfg, docs_path="/nonexistent" if args.mit_only else None)
    bns = load_bns(cfg)
    an = DriftAnalyzer.from_sources(cfg, ipc=ipc, bns=bns)
    table = DriftTable.build(an)
    sp, pp = table.save_csv()
    sources = sorted({p.source for p in ipc.values()})
    print(f"drift: {len(table.sections)} BNS sections, {len(table.pairs)} pairs, {len(table.repealed)} IPC sections "
          f"not carried over, {len(table.inferred)} unmapped IPC sections found in the BNS text "
          f"({time.perf_counter() - t0:.1f}s; IPC text from {', '.join(sources)})")
    print("classes:", table.summary())
    print(f"wrote {sp} and {pp}")
    for b in args.show:
        r = table.sections.get(b.lower())
        if r is None:
            print(f"BNS {b}: not found")
            continue
        print(f"\nBNS {b} [{r.drift_class.value}, carry-over {r.carry_over:.2f}]")
        for line in r.reasons:
            print("  -", line)


if __name__ == "__main__":
    main()
