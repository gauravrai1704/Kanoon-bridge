"""Two independent sets of change labels for evaluating the drift model.  [owner: Gaurav — working]

Neither source is used to BUILD drift reports (model.py reads only bare-act text and the
crosswalk); both are read at evaluation time from data/raw/ and never committed.

S1  indian-legal-mcp mapping tables (MIT): `bns_to_ipc.json` gives every BNS (sub-)section a
    status -- Retained | Change | New Section | New Sub-Section -- and `ipc_to_bns.json` marks
    IPC sections Deleted. Section label:
        new        if any record of the section is "New Section"
        changed    if any record is "Change" or "New Sub-Section"
        retained   otherwise
S2  bns-study-platform `ipcReference.changeNote` (free text per BNS section, e.g. "Re-enacts
    s.282 IPC; the fine limit is raised to ₹5,000."). `note_labels` turns a note into a set of
    categories with sentence-level rules:
        new           a new provision / offence
        punishment    a maximum, minimum, fine or kind of punishment changed
        substance     scope, elements or wording that matters (added, omitted, extended, replaces ...)
        structure     merged, combined, split, moved, re-lettered
        unchanged     "wording retained", "without change of substance" and nothing else
    Sentences about illustrations, lettering or cross-reference wording count as structure.
    Binary: changed = new | punishment | substance.

S1 and S2 disagree often (S1 marks many fine-limit increases "Retained"), which is itself a
finding: "did the provision change?" has more than one right answer. The drift model answers
with a vector, and each source is matched by the part of the vector it means.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from kanoon_bridge.config import Config, load_config, project_path


def _raw(cfg: Config, name: str) -> Path:
    return project_path(cfg.paths.procedure_dir) / "data" / "raw" / name


def _sec(s) -> str:
    return re.sub(r"\(.*", "", str(s or "")).strip().lower()


def s1_section_labels(cfg: Config | None = None) -> dict[str, str]:
    """BNS section -> new | changed | retained (S1)."""
    cfg = cfg or load_config()
    path = _raw(cfg, "bns_to_ipc.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing (make fetch)")
    statuses: dict[str, set[str]] = defaultdict(set)
    for rec in json.loads(path.read_text(encoding="utf-8")):
        statuses[_sec(rec.get("source", {}).get("section"))].add(rec.get("status", ""))
    out = {}
    for sec, st in statuses.items():
        if not sec:
            continue
        out[sec] = "new" if "New Section" in st else "changed" if st & {"Change", "New Sub-Section"} else "retained"
    return out


def s1_deleted_ipc(cfg: Config | None = None) -> set[str]:
    """IPC sections S1 marks Deleted (not carried into the BNS)."""
    cfg = cfg or load_config()
    path = _raw(cfg, "ipc_to_bns.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing (make fetch)")
    return {_sec(r.get("source", {}).get("section")) for r in json.loads(path.read_text(encoding="utf-8"))
            if r.get("status") == "Deleted"}


def s1_ipc_sections(cfg: Config | None = None) -> set[str]:
    """Every IPC section S1 has a record for (the population for repeal detection)."""
    cfg = cfg or load_config()
    return {_sec(r.get("source", {}).get("section"))
            for r in json.loads(_raw(cfg, "ipc_to_bns.json").read_text(encoding="utf-8"))}


_NEW = re.compile(r"\bnew provision\b|^\s*new\b|no ipc equivalent|\bnew offence\b|\bnewly (?:created|introduced)\b", re.I)
_PUNISH = re.compile(r"fine (?:limit|is now|now|becomes|of ₹|is (?:raised|increased|made))|fine limits?\b|"
                     r"\b(?:maximum|minimum)\b(?! is unchanged)|mandatory|compulsory|community service|"
                     r"\braised from\b|\brises? from\b|\bincreased from\b|\breduced from\b|\blowered\b|"
                     r"\bpunishment (?:is |was )?(?:raised|enhanced|increased|reduced|changed|now)|"
                     r"\bcarried up to\b|\bimprisonment for life\b.*\binstead\b|\bdeath penalty\b|"
                     r"\bmore severe\b|\bharsher\b|\blighter\b", re.I)
_PUNISH_SAME = re.compile(r"punishments? (?:is |are )?unchanged|same punishment|without change of substance or punishment", re.I)
_SUBST = re.compile(r"\badds?\b|\badded\b|\bexpressly\b|\bnow (?:covers|includes|protects|applies|extends|reads|requires)\b|"
                    r"\bextend(?:s|ed)?\b|\bexpand|\bwider\b|\bnarrow|\bomit|\bdrops?\b|\bdropped\b|\bno longer\b|"
                    r"\breplaces?\b|\bgender[- ]neutral\b|\belectronic\b|\bnot retained\b|\bsub-sections? .*\bnew\b|"
                    r"\bis new\b|\bare new\b|\bfocus shifts\b|\bdeleted\b|\bremoved\b|\binserted\b|\bnow (?:a|an)\b offence|"
                    r"\bprotects only\b|\bdefinition (?:is )?(?:widened|changed)", re.I)
_STRUCT = re.compile(r"\bmerges?\b|\bcombines?\b|\bconsolidat|\bsplit|\bseparate section\b|\bnow sits\b|\bmoved\b|"
                     r"\blettered\b|\bnumbered\b|\brenumbered\b|\bnow (?:s\.|section)", re.I)
_COSMETIC_SENT = re.compile(r"\billustrations?\b|\blettered\b|\bnumbered\b|\brefers to\b|\bcross-reference|"
                            r"\binstead of 'the last preceding section'", re.I)
_SAME = re.compile(r"wording(?:,| and)?.*retained|without change|\bunchanged\b|\bsame\b|re-enact(?:s|ed)? s\.\d+[a-z]* ipc\.?$|"
                   r"^\s*corresponds(?: broadly)? to (?:s\.|ss\.)\s*[\w(), -]+ipc\.?\s*$", re.I)


def note_labels(note: str) -> set[str]:
    """Categories of one bns-study-platform change note (see module docstring)."""
    note = (note or "").strip()
    if not note:
        return set()
    out: set[str] = set()
    for sent in re.split(r"(?<=[.;])\s+(?=[A-Z'‘(s])", note):
        s = sent.strip()
        if not s:
            continue
        if _NEW.search(s):
            out.add("new")
            continue
        if _COSMETIC_SENT.search(s) and not _PUNISH.search(s):
            out.add("structure")
            continue
        hit = False
        if _PUNISH.search(s) and not _PUNISH_SAME.search(s):
            out.add("punishment")
            hit = True
        if _SUBST.search(re.sub(r"\bfine\b.*", "", s) if hit else s):
            out.add("substance")
            hit = True
        if _STRUCT.search(s):
            out.add("structure")
            hit = True
        if not hit and _SAME.search(s):
            out.add("unchanged")
    if not out:
        out.add("unchanged")
    return out


def s2_section_labels(cfg: Config | None = None) -> dict[str, set[str]]:
    """BNS section -> categories (S2)."""
    from kanoon_bridge.ingest.load_statutes import iter_section_records

    cfg = cfg or load_config()
    out = {}
    for r in iter_section_records(project_path(cfg.paths.bns_dir)):
        ref = r.get("ipcReference") or {}
        note = ref.get("changeNote", "") if isinstance(ref, dict) else ""
        out[str(r.get("number", "")).strip().lower()] = note_labels(note)
    return out


def s2_changed(labels: set[str]) -> bool:
    return bool(labels & {"new", "punishment", "substance"})


def bns_punishment_summaries(cfg: Config | None = None) -> dict[str, str]:
    """BNS section -> the source's own one-line punishment summary (for checking punishment.py)."""
    from kanoon_bridge.ingest.load_statutes import iter_section_records

    cfg = cfg or load_config()
    return {str(r.get("number", "")).strip().lower(): r.get("punishment", "") or ""
            for r in iter_section_records(project_path(cfg.paths.bns_dir))}


__all__ = ["s1_section_labels", "s1_deleted_ipc", "s1_ipc_sections", "note_labels", "s2_section_labels",
           "s2_changed", "bns_punishment_summaries"]
