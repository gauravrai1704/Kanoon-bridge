"""Old-code and new-code section texts as clause lists.  [owner: Gaurav — working]

Three code families changed on 1 July 2024 (`FAMILIES`): IPC -> BNS, CrPC -> BNSS and the
Indian Evidence Act -> BSA. `load_family(family)` returns (old provisions, new provisions,
crosswalk rows) for any of them.

Sources for the penal family (both bare-act text, i.e. government material):
    BNS  data/raw/bns-study-platform/data/sections/*.json   (`number`, `title`, `bareActText` only)
    IPC  data/raw/indian-legal-mcp/data/raw/indian_penal_code_1860.json  (MIT checkout)
         or, when the corpus has been built, the IL-PCSR IPC statute documents in docs.jsonl;
         per section the longer text wins (the MIT file truncates some sections, e.g. s.300).

A provision is split into clauses: numbered sub-sections "(1)", lettered items "(a)",
Explanations, Exceptions, provisos and Illustrations. Clauses are the unit the diff aligns.

Procedure and evidence (indian-legal-mcp, MIT): code_of_criminal_procedure_1973.json and
indian_evidence_act_1872.json (old), bharatiya_nagarik_suraksha_sanhita_2023.json and
bharatiya_sakshya_adhiniyam_2023.json (new); the older file names crpc.json, iea.json,
bnss_sections.json and bsa_sections.json are accepted too. Their crosswalks are
data/crosswalk/crpc_bnss.csv and iea_bsa.csv, aligned by ingest/align_codes.py.

Cross-references are renumbered before comparing: an IPC text that says "section 300" is
rewritten to "section 101" (its BNS successor, when the crosswalk gives exactly one), so a
renumbered reference is not counted as a change of substance.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from kanoon_bridge.config import Config, load_config, project_path

_CLAUSE_START = re.compile(
    r"(?=(?:\(\d{1,2}[A-Z]?\)\s)|(?:\((?:[a-z]|[ivx]{1,4})\)\s)|(?:\bExplanation\b)|(?:\bExceptions?\b)"
    r"|(?:\bProvided\s+(?:further\s+)?that\b)|(?:\bIllustrations?\b))")
_XREF = re.compile(r"\b(sections?|ss?\.)\s+(\d{1,3}[A-Z]{0,2})\b", re.I)


@dataclass
class Provision:
    code: str                     # "ipc" | "bns"
    section: str                  # lower-case, no sub-section: "304b"
    title: str
    text: str
    source: str = ""
    clauses: list[str] = field(default_factory=list)
    defective: str = ""           # why the source text cannot be trusted ("" = fine); judged unverified

    @property
    def ref(self) -> str:
        return f"{self.code}:{self.section}"

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def split_clauses(text: str, min_words: int = 4) -> list[str]:
    """Sub-sections, items, Explanations, Exceptions, provisos and Illustrations, in order.
    Very short pieces (an item label on its own) are glued to the next clause."""
    text = " ".join((text or "").replace("––", " ").replace("—", " ").split())
    if not text:
        return []
    starts = sorted({0, *(m.start() for m in _CLAUSE_START.finditer(text))})
    parts = [text[a:b].strip(" ;:") for a, b in zip(starts, starts[1:] + [len(text)])]
    out: list[str] = []
    carry = ""
    for p in parts:
        p = (carry + " " + p).strip() if carry else p
        if len(p.split()) < min_words:
            carry = p
            continue
        carry = ""
        out.append(p)
    if carry:
        if out:
            out[-1] = out[-1] + " " + carry
        else:
            out.append(carry)
    return out


# act names as they appear in old-code text -> the code their section numbers belong to
_ACT_NAMES = [(re.compile(r"indian\s+penal\s+code", re.I), "ipc"),
              (re.compile(r"code\s+of\s+criminal\s+procedure", re.I), "crpc"),
              (re.compile(r"indian\s+evidence\s+act", re.I), "iea")]


def renumber_references(text: str, mapping: dict[str, str], other: dict[str, dict[str, str]] | None = None,
                        window: int = 160) -> str:
    """'section 300' -> 'section 101' when mapping['300'] == '101' (old text into new numbering).

    `other` maps another code to its renumbering (e.g. {"ipc": IPC->BNS}): a reference followed,
    within `window` characters and before the sentence ends, by that code's name ("section 302 of
    the Indian Penal Code") is renumbered with that code's map instead. CrPC text cites IPC
    sections and BNSS text cites the BNS ones, so without this every such reference reads as
    a change.
    """
    if not mapping and not other:
        return text
    acts = sorted((m.start(), code) for rx, code in _ACT_NAMES for m in rx.finditer(text)) if other else []

    def sub(m: re.Match) -> str:
        table = mapping
        for pos, code in acts:
            if pos < m.end():
                continue
            if pos - m.end() <= window and "." not in text[m.end():pos].replace("s.", "").replace("ss.", ""):
                table = other.get(code, mapping)
            break
        new = (table or {}).get(m.group(2).lower())
        return f"{m.group(1)} {new}" if new else m.group(0)

    return _XREF.sub(sub, text)


# --------------------------------------------------------------------------- families


@dataclass(frozen=True)
class Family:
    old: str                      # old code: "ipc" | "crpc" | "iea"
    new: str                      # new code: "bns" | "bnss" | "bsa"
    old_label: str                # for reasons shown to users: "IPC", "CrPC", "Evidence Act"
    new_label: str                # "BNS", "BNSS", "BSA"
    crosswalk: str                # config paths key of the old -> new crosswalk CSV
    old_files: tuple[str, ...] = ()   # indian-legal-mcp file names, newest naming first
    new_files: tuple[str, ...] = ()


FAMILIES: dict[str, Family] = {
    "ipc": Family("ipc", "bns", "IPC", "BNS", "crosswalk"),
    "crpc": Family("crpc", "bnss", "CrPC", "BNSS", "crpc_bnss",
                   ("code_of_criminal_procedure_1973.json", "crpc.json"),
                   ("bharatiya_nagarik_suraksha_sanhita_2023.json", "bnss_sections.json")),
    "iea": Family("iea", "bsa", "Evidence Act", "BSA", "iea_bsa",
                  ("indian_evidence_act_1872.json", "iea.json"),
                  ("bharatiya_sakshya_adhiniyam_2023.json", "bsa_sections.json")),
}
CODE_FAMILY = {c: f for f in FAMILIES.values() for c in (f.old, f.new)}   # "bnss" -> the CrPC family


def procedure_file(cfg: Config, names: tuple[str, ...]) -> Path:
    """First of `names` that exists in the indian-legal-mcp checkout (data/raw/...)."""
    from kanoon_bridge.ingest.align_codes import raw_file

    root = project_path(cfg.paths.procedure_dir) / "data" / "raw"
    for n in names:
        if raw_file(root, n).exists():
            return raw_file(root, n)
    raise FileNotFoundError(f"none of {', '.join(names)} in {root} (make fetch)")


# --------------------------------------------------------------------------- loaders


def _section_key(num: str) -> tuple[int, str]:
    m = re.match(r"(\d+)(.*)", num)
    return (int(m.group(1)), m.group(2)) if m else (10**6, num)


def load_bns(cfg: Config | None = None) -> dict[str, Provision]:
    from kanoon_bridge.ingest.load_statutes import iter_section_records

    cfg = cfg or load_config()
    root = project_path(cfg.paths.bns_dir)
    out: dict[str, Provision] = {}
    for r in iter_section_records(root):
        num = str(r.get("number", "")).strip().lower()
        if not num:
            continue
        text = r.get("bareActText", "") or ""
        out[num] = Provision("bns", num, (r.get("title") or "").strip(), text, "bns-study-platform (bare act)",
                             split_clauses(text))
    return out


def ipc_json_path(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    return project_path(cfg.paths.procedure_dir) / "data" / "raw" / "indian_penal_code_1860.json"


# --------------------------------------------------------------------------- India Code (kanoonhq/open-laws)

INDIA_CODE_ACTS = {"ipc": "A1860-45", "crpc": "A1974-2", "iea": "A1872-1",
                   "bns": "A2023-45", "bnss": "A2023-46", "bsa": "A2023-47"}
_FOOTNOTE = re.compile(r"(?<![\w\]])\d{1,2}\[")      # "1[(a) who commits ..." -> "(a) who commits ..."
_OMISSION = re.compile(r"(?:\*\s*){3,}")             # "* * *" marks text omitted by an amendment


def use_india_code(cfg: Config) -> bool:
    return (cfg.get("drift") or {}).get("text_source", "india_code") == "india_code"


def clean_india_code(text: str) -> str:
    """Strip amendment footnote markers ("1[", "]"), omission stars and the State amendments India
    Code appends after the central text ("STATE AMENDMENTS West Bengal.— ..."; 51 CrPC and 16 IPC
    sections): the drift is between the central codes."""
    text = (text or "").split("STATE AMENDMENT", 1)[0]
    text = _FOOTNOTE.sub("", text)
    text = _OMISSION.sub(" ", text.replace("]", ""))
    text = re.sub(r"^[\s”\"’'.,;:]*[—–]+\s*", "", text)    # "”.—A “special law” is ...": a heading's tail
    return " ".join(text.split())


def load_india_code(code: str, cfg: Config | None = None) -> dict[str, Provision] | None:
    """Sections of one of the six codes as consolidated on India Code (kanoonhq/open-laws, CC0),
    or None when the checkout is missing (make fetch)."""
    cfg = cfg or load_config()
    root = project_path(cfg.paths.get("open_laws_dir", "data/raw/open-laws"))
    path = root / "data" / "indian_central_act_sections.parquet"
    if not path.exists():
        return None
    import pandas as pd

    df = pd.read_parquet(path, columns=["act_id", "section_number", "heading", "full_text"])
    df = df[df.act_id == INDIA_CODE_ACTS[code]]
    out: dict[str, Provision] = {}
    for num, head, text in zip(df.section_number, df.heading, df.full_text):
        n = re.sub(r"^\s*section\s+", "", str(num), flags=re.I).strip().rstrip(".").lower()
        if not n or n in out:
            continue
        body = clean_india_code(text)
        out[n] = Provision(code, n, clean_india_code(head).rstrip("."), body, "India Code (kanoonhq/open-laws)",
                           split_clauses(body))
    return dict(sorted(out.items(), key=lambda kv: _section_key(kv[0]))) or None


def load_ipc(cfg: Config | None = None, docs_path: str | Path | None = None) -> dict[str, Provision]:
    """IPC sections: India Code's consolidated text when available (drift.text_source); otherwise
    the MIT checkout, upgraded with IL-PCSR statute text where that is longer."""
    cfg = cfg or load_config()
    if use_india_code(cfg):
        ic = load_india_code("ipc", cfg)
        if ic:
            return ic
    out: dict[str, Provision] = {}
    path = ipc_json_path(cfg)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for r in json.load(f):
                num = str(r.get("Section", "")).strip().lower()
                if not num:
                    continue
                text = (r.get("section_desc") or "").strip()
                out[num] = Provision("ipc", num, (r.get("section_title") or "").strip(" -"), text,
                                     "indian-legal-mcp (bare act)", split_clauses(text))
    docs_path = Path(docs_path) if docs_path else project_path(cfg.paths.docs)
    if docs_path.exists():
        from kanoon_bridge.schema import DocType, read_documents

        for d in read_documents(docs_path):
            if d.doc_type != DocType.STATUTE or d.code.value != "ipc" or not d.section:
                continue
            num = d.section.lower()
            body = " ".join(p.text for p in d.paragraphs if p.para_id >= 0)
            if num not in out or len(body.split()) > out[num].n_words:
                title = out[num].title if num in out else d.title
                out[num] = Provision("ipc", num, title, body, "IL-PCSR statute text", split_clauses(body))
    if not out:
        raise FileNotFoundError(f"no IPC text: expected {path} (make fetch) or IPC statutes in {docs_path}")
    return dict(sorted(out.items(), key=lambda kv: _section_key(kv[0])))


__all__ = ["Provision", "Family", "flag_shifted", "load_india_code", "clean_india_code", "INDIA_CODE_ACTS", "FAMILIES", "CODE_FAMILY", "split_clauses", "renumber_references", "load_bns",
           "load_ipc", "load_procedure", "crosswalk_rows", "procedure_file"]


# "34. Duty of officers ... to make certain\nreport.—(1)": the PDF-extracted new codes sometimes keep a
# heading that wraps over a line break, which ingest/align_codes.load_new (single-line) leaves in
_HEADING = re.compile(r"^\s*\d{1,3}[A-Z]?\.\s+[^—–]{3,200}?\.?\s*[—–]{1,2}\s*", re.S)


_HEADING_TAIL = re.compile(r"^\s*[a-z][^.\n—–]{0,80}\.?\s*[—–]{1,2}\s*")


def flag_shifted(provisions: dict[str, Provision], threshold: float = 0.8) -> list[str]:
    """Mark sections whose body is (almost) the next section's body: the PDF-extracted source
    sometimes files a section's text under the previous number (BNSS 58 holds s.59's text).
    Their drift cannot be judged, so the analyser treats them as unverified."""
    from kanoon_bridge.drift.diff import containment, normalise

    keys = list(provisions)
    out = []
    for a, b in zip(keys, keys[1:]):
        ta, tb = normalise(provisions[a].text), normalise(provisions[b].text)
        if len(ta) > 10 and containment(ta, tb) > threshold:
            provisions[a].defective = f"its text in the source is that of section {b}"
            out.append(a)
    return out


def load_procedure(fam: Family, cfg: Config | None = None) -> tuple[dict[str, Provision], dict[str, Provision]]:
    """Old and new sections of the CrPC or Evidence Act family: India Code's consolidated text
    when available (drift.text_source), else the indian-legal-mcp checkout."""
    from kanoon_bridge.ingest.align_codes import load_new, load_old

    cfg = cfg or load_config()
    if use_india_code(cfg):
        old_ic, new_ic = load_india_code(fam.old, cfg), load_india_code(fam.new, cfg)
        if old_ic and new_ic:
            flag_shifted(new_ic)
            return old_ic, new_ic
    src = "indian-legal-mcp (bare act)"
    old = {s.number: Provision(fam.old, s.number, s.title, s.text, src, split_clauses(s.text))
           for s in load_old(procedure_file(cfg, fam.old_files))}
    new = {}
    for s in load_new(procedure_file(cfg, fam.new_files)):
        text = _HEADING.sub("", s.text, count=1)        # a heading that wrapped onto two lines
        text = _HEADING_TAIL.sub("", text, count=1)     # "four hours.—Officers": the rest of a cut heading
        new[s.number] = Provision(fam.new, s.number, s.title, text, src, split_clauses(text))
    flag_shifted(new)
    key = lambda kv: _section_key(kv[0])  # noqa: E731
    return dict(sorted(old.items(), key=key)), dict(sorted(new.items(), key=key))


def crosswalk_rows(fam: Family, cfg: Config | None = None, normalizer=None) -> list[tuple[str, str, str, str]]:
    """(old section, new section, relation, note) rows of the family's committed crosswalk."""
    import csv

    cfg = cfg or load_config()
    if fam.old == "ipc":
        from kanoon_bridge.text.version_norm import VersionNormalizer

        norm = normalizer or VersionNormalizer.load(cfg)
        return [(r.ipc_section, r.bns_section, r.relation, r.note) for r in norm.crosswalk]
    path = project_path(cfg.paths[fam.crosswalk])
    with open(path, encoding="utf-8") as f:
        rows = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        return [(r[f"{fam.old}_section"].strip().lower(), r[f"{fam.new}_section"].strip().lower(),
                 r.get("relation", "").strip().lower(), r.get("note", "") or "") for r in rows]
