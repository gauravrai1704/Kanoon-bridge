"""Build the statutory drift table (drift/).  [owner: Gaurav — working]

Reads the bare-act text (make fetch) and the committed crosswalks of all three code families;
writes, per family,
    data/crosswalk/drift_sections.csv, drift_pairs.csv                  IPC -> BNS
    data/crosswalk/drift_crpc_bnss_sections.csv, ..._pairs.csv          CrPC -> BNSS
    data/crosswalk/drift_iea_bsa_sections.csv, ..._pairs.csv            Evidence Act -> BSA
(one row per new-code section plus old-code sections with no successor; carry-over per pair).
All are committed, so the search engine and the app use drift without the raw data.

    python scripts/13_build_drift.py                 # IPC text: MIT checkout, upgraded with IL-PCSR text if built
    python scripts/13_build_drift.py --mit-only      # IPC text from the MIT checkout only (what we commit)
    python scripts/13_build_drift.py --family crpc   # one family only
    python scripts/13_build_drift.py --show bns:152 bnss:482   # print some reports
"""

from __future__ import annotations

import argparse
import time

from kanoon_bridge.config import load_config
from kanoon_bridge.drift.model import DriftAnalyzer, DriftTable
from kanoon_bridge.drift.provisions import CODE_FAMILY, FAMILIES, load_bns, load_ipc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=[*FAMILIES, "all"], default="all",
                    help="ipc (IPC->BNS), crpc (CrPC->BNSS), iea (Evidence Act->BSA) or all")
    ap.add_argument("--mit-only", action="store_true", help="ignore IL-PCSR IPC text even if docs.jsonl exists")
    ap.add_argument("--show", nargs="*", default=[], help="new-code sections to print, e.g. bns:152 bnss:482")
    args = ap.parse_args()
    cfg = load_config()
    tables = {}
    for fam in (FAMILIES if args.family == "all" else [args.family]):
        F = FAMILIES[fam]
        t0 = time.perf_counter()
        if fam == "ipc":
            ipc = load_ipc(cfg, docs_path="/nonexistent" if args.mit_only else None)
            an = DriftAnalyzer.from_sources(cfg, old=ipc, new=load_bns(cfg))
        else:
            an = DriftAnalyzer.from_sources(cfg, family=fam)
        table = tables[fam] = DriftTable.build(an)
        sp, pp = table.save_csv()
        sources = sorted({p.source for p in an.old_prov.values()})
        print(f"{F.old_label} -> {F.new_label}: {len(table.sections)} {F.new_label} sections, {len(table.pairs)} pairs, "
              f"{len(table.repealed)} {F.old_label} sections not carried over, {len(table.inferred)} unmapped "
              f"{F.old_label} sections found in the {F.new_label} text ({time.perf_counter() - t0:.1f}s; "
              f"{F.old_label} text from {', '.join(sources)})")
        print("  classes:", table.summary())
        print(f"  wrote {sp.name} and {pp.name}")
    for ref in args.show:
        code, _, sec = ref.lower().partition(":")
        fam = CODE_FAMILY.get(code)
        table = tables.get(fam.old) if fam else None
        r = table.sections.get(sec) if table else None
        if r is None:
            print(f"{ref}: not found (give it as bns:152, bnss:482 or bsa:63)")
            continue
        print(f"\n{fam.new_label} {sec.upper()} [{r.drift_class.value}, carry-over {r.carry_over:.2f}]")
        for line in r.reasons:
            print("  -", line)


if __name__ == "__main__":
    main()
