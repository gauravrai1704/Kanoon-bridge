"""Load the Bharatiya Nyaya Sanhita, 2023 into one Document per section.  [owner: Gaurav — working]

Primary source: data/raw/bns-study-platform/data/sections/*.json (scripts/00_fetch_data.py),
358 sections whose bare-act text that project verified against the Gazette of India.
Only `number`, `title`, `bareActText` and `ipcReference` are read — the bare act is government
material; none of that project's commentary is used.

Fallback: a plain-text bare act (e.g. copied from India Code) with headings like
"103. Punishment for murder.—".

Each section becomes:
    Document(doc_id="bns:103", doc_type=STATUTE, code=BNS, section="103",
             title="Punishment for murder", paragraphs=[...zone 'statute'...],
             decision_date=2024-07-01, meta={"ipc_reference": "302 IPC"})
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from kanoon_bridge.ingest.segment import segment_statute
from kanoon_bridge.schema import Code, Document, DocType, Paragraph, section_ref

BNS_IN_FORCE = date(2024, 7, 1)


def section_files(root: str | Path) -> list[Path]:
    root = Path(root)
    sec_dir = root / "data" / "sections" if (root / "data" / "sections").exists() else root
    return sorted(sec_dir.glob("*.json"))


def iter_section_records(root: str | Path) -> list[dict]:
    """Raw section records from the bns-study-platform JSON files, sorted by section number."""
    recs: list[dict] = []
    for path in section_files(root):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        recs += data if isinstance(data, list) else data.get("sections", [])
    return sorted(recs, key=lambda r: _section_key(str(r.get("number", ""))))


def _section_key(num: str) -> tuple[int, str]:
    m = re.match(r"(\d+)(.*)", num)
    return (int(m.group(1)), m.group(2)) if m else (10**6, num)


def _doc(number: str, title: str, text: str, ipc_ref: str = "") -> Document:
    number = number.strip().lower()
    doc = Document(
        doc_id=section_ref(Code.BNS, number),
        doc_type=DocType.STATUTE,
        title=title.strip(),
        paragraphs=[Paragraph(text.strip() or title, "statute", 0)],
        code=Code.BNS,
        section=number,
        decision_date=BNS_IN_FORCE,
        meta={"act": "Bharatiya Nyaya Sanhita, 2023", "act_code": "bns", "ref": section_ref(Code.BNS, number),
              "ipc_reference": ipc_ref, "source": "bns-study-platform (bare act)"},
    )
    return segment_statute(doc)


def parse_bns_json(root: str | Path) -> list[Document]:
    """All sections from the JSON source."""
    docs = []
    for r in iter_section_records(root):
        ipc = r.get("ipcReference") or ""
        ipc = ipc.get("section", "") if isinstance(ipc, dict) else str(ipc)
        docs.append(_doc(str(r["number"]), r.get("title", ""), r.get("bareActText", ""), ipc))
    return docs


_HEADING_RE = re.compile(r"^\s*(\d{1,3}[A-Z]?)\.\s+([^\n]{3,200}?)\.?\s*[—–-]{1,2}", re.M)


def parse_bns_text(path: str | Path) -> list[Document]:
    """Fallback: split a plain-text bare act on '103. Title.—' headings."""
    text = Path(path).read_text(encoding="utf-8")
    heads = list(_HEADING_RE.finditer(text))
    docs = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        docs.append(_doc(m.group(1), m.group(2), text[m.end():end]))
    return docs


def parse_bns(path: str | Path) -> list[Document]:
    """BNS sections from either source: a directory of JSON (preferred) or a text file."""
    path = Path(path)
    if path.is_dir():
        return parse_bns_json(path)
    if path.suffix == ".json":
        return parse_bns_json(path.parent)
    return parse_bns_text(path)
