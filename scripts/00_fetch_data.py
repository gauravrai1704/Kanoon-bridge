"""Download the raw data into data/raw/ (run once per machine).  [owner: Gaurav — working]

    huggingface-cli login            # IL-PCSR is gated: accept its terms on the HF page first
    python scripts/00_fetch_data.py  # both sources
    python scripts/00_fetch_data.py --only bns
    python scripts/00_fetch_data.py --only procedure    # BNSS + BSA (CrPC/IEA bridge)

Sources
-------
1. IL-PCSR (IIT Kharagpur + IIT Kanpur), CC-BY-NC-SA 4.0
   https://huggingface.co/datasets/Exploration-Lab/IL-PCSR
   -> data/raw/ilpcsr/<config>/<split>.parquet   (queries / statutes / precedents)
   Exported locally so every later step works offline and identically for the whole team.

2. BNS bare-act text + IPC correspondences, from the bns-study-platform repository
   https://github.com/PSKprem/bns-study-platform  (data/sections/*.json)
   -> data/raw/bns-study-platform/
   We use only the bare-act text (government material) and the section correspondences
   (facts); none of the repository's commentary. Credit it in the README and report.

Check the result with: python scripts/00_check_access.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from kanoon_bridge.config import load_config, project_path

BNS_REPO = "https://github.com/PSKprem/bns-study-platform"
PROCEDURE_REPO = "https://github.com/GSMS-B/indian-legal-mcp"     # BNSS + BSA bare acts, CrPC + IEA (MIT)


def fetch_procedure(cfg) -> None:
    dest = project_path(cfg.paths.procedure_dir)
    if (dest / "data" / "raw" / "bnss_sections.json").exists():
        print(f"procedure: already present at {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", PROCEDURE_REPO, str(dest)], check=True)
    print(f"procedure: cloned to {dest}")


def fetch_bns(cfg) -> None:
    dest = project_path(cfg.paths.bns_dir)
    if (dest / "data" / "sections").exists():
        print(f"bns: already present at {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", BNS_REPO, str(dest)], check=True)
    print(f"bns: cloned to {dest}")


def fetch_ilpcsr(cfg) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit('ilpcsr: pip install -e ".[data]" first')
    out = project_path(cfg.paths.ilpcsr_dir)
    for config in ("queries", "statutes", "precedents"):
        ds = load_dataset(cfg.ilpcsr.hf_name, config)
        for split, table in ds.items():
            path = out / config / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(str(path))
            print(f"ilpcsr: {config}/{split}: {len(table)} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ilpcsr", "bns", "procedure"])
    args = ap.parse_args()
    cfg = load_config()
    if args.only in (None, "bns"):
        fetch_bns(cfg)
    if args.only in (None, "procedure"):
        fetch_procedure(cfg)
    if args.only in (None, "ilpcsr"):
        fetch_ilpcsr(cfg)


if __name__ == "__main__":
    main()
