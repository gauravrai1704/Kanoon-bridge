"""IPC and BNS section texts as clause lists.  [owner: Gaurav — working]

Sources (both bare-act text, i.e. government material):
    BNS  data/raw/bns-study-platform/data/sections/*.json   (`number`, `title`, `bareActText` only)
    IPC  data/raw/indian-legal-mcp/data/raw/indian_penal_code_1860.json  (MIT checkout)
         or, when the corpus has been built, the IL-PCSR IPC statute documents in docs.jsonl;
         per section the longer text wins (the MIT file truncates some sections, e.g. s.300).

A provision is split into clauses: numbered sub-sections "(1)", lettered items "(a)",
Explanations, Exceptions, provisos and Illustrations. Clauses are the unit the diff aligns.

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


def renumber_references(text: str, mapping: dict[str, str]) -> str:
    """'section 300' -> 'section 101' when mapping['300'] == '101' (IPC text into BNS numbering)."""
    if not mapping:
        return text

    def sub(m: re.Match) -> str:
        new = mapping.get(m.group(2).lower())
        return f"{m.group(1)} {new}" if new else m.group(0)

    return _XREF.sub(sub, text)


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


def load_ipc(cfg: Config | None = None, docs_path: str | Path | None = None) -> dict[str, Provision]:
    """IPC sections from the MIT checkout, upgraded with IL-PCSR statute text where that is longer."""
    cfg = cfg or load_config()
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


__all__ = ["Provision", "split_clauses", "renumber_references", "load_bns", "load_ipc"]
