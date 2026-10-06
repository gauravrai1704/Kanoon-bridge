"""Version-aware normalisation: section tokens -> canonical offence IDs.  [owner: B — CORE NOVELTY]

The idea
--------
"IPC 302" in a 1995 judgment and "BNS 103" in a 2026 query are the same offence (murder).
Like stemming maps word forms to one term, we map section numbers from both codes to one
canonical offence ID, at index time AND query time:

    sec:ipc:302     -> sec:ipc:302, off:murder
    sec:bns:103(1)  -> sec:bns:103(1), off:murder

The original section token is kept (exact-section search still works); the offence token is
added after it. Split/merge mappings give weights below 1.0 (see `to_offences`).

Data files (see data/crosswalk/README.md for the formats)
    data/crosswalk/offence_ids.csv   offence_id, label, ipc_sections, bns_sections, keywords
    data/crosswalk/ipc_bns.csv       ipc_section, bns_section, relation, note

What is already done: loading both CSVs into lookup tables.
What B builds: `to_offences`, `equivalents`, `normalize_tokens` (+ tests/test_version_norm.py).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field

from kanoon_bridge.config import Config, load_config, project_path
from kanoon_bridge.text.tokenize import SECTION_PREFIX

OFFENCE_PREFIX = "off:"


@dataclass
class CrosswalkRow:
    ipc_section: str          # "" for a section new in BNS
    bns_section: str          # "" for a section dropped from BNS
    relation: str             # same | split | merge | new | dropped | modified
    note: str = ""


@dataclass
class VersionNormalizer:
    # section ref ("ipc:302") -> list of offence ids
    section_to_offences: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    # offence id -> {"label":..., "ipc": [...], "bns": [...], "keywords": [...]}
    offences: dict[str, dict] = field(default_factory=dict)
    crosswalk: list[CrosswalkRow] = field(default_factory=list)

    # ------------------------------------------------------------------ loading (working)
    @classmethod
    def load(cls, cfg: Config | None = None) -> "VersionNormalizer":
        cfg = cfg or load_config()
        norm = cls()
        off_path = project_path(cfg.paths.offence_ids)
        cw_path = project_path(cfg.paths.crosswalk)
        if not off_path.exists():
            raise FileNotFoundError(f"{off_path} missing")
        with open(off_path, encoding="utf-8") as f:
            for row in csv.DictReader(_skip_comments(f)):
                oid = row["offence_id"].strip().lower()
                ipc = _split(row.get("ipc_sections", ""))
                bns = _split(row.get("bns_sections", ""))
                norm.offences[oid] = {
                    "label": row.get("label", ""),
                    "ipc": ipc,
                    "bns": bns,
                    "keywords": _split(row.get("keywords", "")),
                }
                for s in ipc:
                    norm.section_to_offences[f"ipc:{s}"].append(oid)
                for s in bns:
                    norm.section_to_offences[f"bns:{s}"].append(oid)
        if cw_path.exists():
            with open(cw_path, encoding="utf-8") as f:
                for row in csv.DictReader(_skip_comments(f)):
                    norm.crosswalk.append(
                        CrosswalkRow(
                            ipc_section=row["ipc_section"].strip().lower(),
                            bns_section=row["bns_section"].strip().lower(),
                            relation=row["relation"].strip().lower(),
                            note=row.get("note", ""),
                        )
                    )
        return norm

    # ------------------------------------------------------------------ B builds these
    def to_offences(self, section_ref: str) -> list[tuple[str, float]]:
        """'ipc:302' -> [('off:murder', 1.0)].

        TODO(B):
          * look up the section; also try its base number ("103(1)" -> "103") when the exact
            sub-section is not listed
          * a section mapped to k offences (split) gets weight 1/k each, or use a weight
            column if you add one to offence_ids.csv
          * unknown code ("?:302") -> return [] (collision.py must resolve it first)
        """
        raise NotImplementedError("TODO(B): section -> offence ids")

    def equivalents(self, section_ref: str) -> list[str]:
        """Sections in the OTHER code with the same offence: 'ipc:302' -> ['bns:103(1)'].

        Used by the demo ("IPC 302 = BNS 103") and by query expansion in query/analyzer.py.
        TODO(B): go section -> offences -> sections of the other code; drop the input itself.
        """
        raise NotImplementedError("TODO(B): cross-code equivalent sections")

    def normalize_tokens(self, tokens: list[str]) -> list[str]:
        """Insert an 'off:<id>' token after every section token that maps to an offence.

        TODO(B): for each token starting with SECTION_PREFIX, keep it and append the offence
        tokens from `to_offences` (weights are ignored at token level; index weights come
        later if needed). Non-section tokens pass through unchanged.
        """
        raise NotImplementedError("TODO(B): add offence tokens")


def _split(cell: str) -> list[str]:
    return [p.strip().lower() for p in (cell or "").split(";") if p.strip()]


def _skip_comments(lines):
    return (line for line in lines if not line.lstrip().startswith("#"))


__all__ = ["VersionNormalizer", "CrosswalkRow", "OFFENCE_PREFIX", "SECTION_PREFIX"]
