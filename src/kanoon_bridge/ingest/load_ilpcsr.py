"""Load the IL-PCSR corpus (IIT Kharagpur + IIT Kanpur, 2025).  [owner: Gaurav — working]

Source: https://huggingface.co/datasets/Exploration-Lab/IL-PCSR (gated, CC-BY-NC-SA 4.0).
`scripts/00_fetch_data.py` exports it to data/raw/ilpcsr/<config>/<split>.parquet; this module
reads that export (parquet, jsonl or json), falling back to the HF hub if no export exists.

Real schema (from the dataset card and the authors' dataset.py):

  config "queries"     splits train_queries (5,017) / dev_queries (627) / test_queries (627)
  config "precedents"  split  precedent_candidates (3,183)    -- same columns as queries
      id, case_title, date (YYYY-MM-DD), jurisdiction, text (list of paragraphs),
      rhetorical_roles (list, one per paragraph; older exports call it "rr"),
      relevant_statutes, relevant_statute_ids, relevant_precedents, relevant_precedent_ids
  config "statutes"    split  statute_candidates (936)
      id, provision_name, text (list)

Query texts have citations masked ([SECTION], [ACT], [PRECEDENT], [ENTITY]); precedents do not.

Our names: split "val" = their "dev". Statutes get canonical section refs ("ipc:302") parsed from
provision_name, so precedents' statutes_cited line up with the crosswalk and the facets.

    docs = load_precedents()            # list[Document], doc_type PRECEDENT
    qrels = load_qrels("train", "precedent")
"""

from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from kanoon_bridge.config import Config, load_config, project_path
from kanoon_bridge.ingest.segment import segment_judgment
from kanoon_bridge.schema import Code, Document, DocType, Paragraph, section_ref

SPLIT_NAMES = {"train": "train_queries", "val": "dev_queries", "dev": "dev_queries", "test": "test_queries"}

# Act name -> code string used in section refs. Order matters (most specific first).
ACT_PATTERNS: list[tuple[str, str]] = [
    (r"bharatiya\s+nyaya\s+sanhita", "bns"),
    (r"bharatiya\s+nagarik\s+suraksha\s+sanhita", "bnss"),
    (r"bharatiya\s+sakshya\s+adhiniyam", "bsa"),
    (r"indian\s+penal\s+code|\bi\.?p\.?c\b", "ipc"),
    (r"code\s+of\s+criminal\s+procedure|\bcr\.?\s?p\.?\s?c\b", "crpc"),
    (r"code\s+of\s+civil\s+procedure|\bc\.?p\.?c\b", "cpc"),
    (r"constitution", "constitution"),
    (r"evidence\s+act", "iea"),
    (r"narcotic\s+drugs", "ndps"),
    (r"prevention\s+of\s+corruption", "pca"),
    (r"motor\s+vehicles\s+act", "mva"),
    (r"negotiable\s+instruments", "nia"),
    (r"income[\s-]+tax\s+act", "ita"),
    (r"dowry\s+prohibition", "dpa"),
]
_CODE_ENUM = {"ipc": Code.IPC, "bns": Code.BNS, "crpc": Code.CRPC, "bnss": Code.BNSS}
_SECTION_RE = re.compile(r"\b(?:section|sec\.?|s\.|article|art\.?|rule|order)\s*(\d+[a-z]*(?:\s?\(\s?[0-9a-z]+\s?\))*)", re.I)
_YEAR_RE = re.compile(r",?\s*\b(1[89]\d\d|20\d\d)\b")


# --------------------------------------------------------------------------- provision names


def parse_provision(name: str) -> tuple[str, str | None, str]:
    """'Section 302 in The Indian Penal Code, 1860' -> ('ipc', '302', 'The Indian Penal Code, 1860').

    Handles the common shapes ('Section 302 in The X', 'X_Section 302', 'X - Section 302',
    'Article 21 in Constitution of India'). Unknown acts get a slug as their code.
    """
    low = name.lower().replace("_", " ")
    m = _SECTION_RE.search(low)
    section = re.sub(r"\s+", "", m.group(1)) if m else None
    act = _SECTION_RE.sub(" ", low)
    act = re.sub(r"\b(in|of|under)\s+the\b|\bthe\b|\s-\s|\bin\b", " ", act)
    act = re.sub(r"\s+", " ", act).strip(" ,.-")
    for pattern, code in ACT_PATTERNS:
        if re.search(pattern, low):
            return code, section, act
    slug = re.sub(r"[^a-z0-9]+", "_", _YEAR_RE.sub("", act)).strip("_") or "unknown_act"
    return slug, section, act


# --------------------------------------------------------------------------- raw rows


def _raw_dir(cfg: Config) -> Path:
    return project_path(cfg.paths.ilpcsr_dir)


def _read_table(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        return [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()} for row in df.to_dict("records")]
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else list(data.values())


@lru_cache(maxsize=16)
def _rows(config: str, split: str, root: str) -> tuple[dict, ...]:
    base = Path(root) / config
    for folder in (base, Path(root)):        # <config>/<split>.* (our export) or flat <split>.* (HF download)
        for ext in (".parquet", ".jsonl", ".json"):
            path = folder / f"{split}{ext}"
            if path.exists():
                return tuple(_read_table(path))
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise FileNotFoundError(
            f"{base}/{split}.* not found and `datasets` is not installed - run scripts/00_fetch_data.py"
        ) from err
    cfg = load_config()
    return tuple(load_dataset(cfg.ilpcsr.hf_name, config)[split])


def rows(config: str, split: str, cfg: Config | None = None) -> list[dict]:
    """Raw rows of one config/split, as dicts with the dataset's own column names."""
    cfg = cfg or load_config()
    return list(_rows(config, split, str(_raw_dir(cfg))))


def inspect(cfg: Config | None = None, n: int = 2) -> None:
    """Print columns, sizes and examples of every split (hour-0 check)."""
    cfg = cfg or load_config()
    for config, split in [("queries", "train_queries"), ("queries", "dev_queries"), ("queries", "test_queries"),
                          ("statutes", "statute_candidates"), ("precedents", "precedent_candidates")]:
        r = rows(config, split, cfg)
        print(f"\n== {config}/{split}: {len(r)} rows; columns: {list(r[0]) if r else []}")
        for row in r[:n]:
            short = {k: (str(v)[:100] + "...") if len(str(v)) > 100 else v for k, v in row.items()}
            print("  ", short)
    ids = [s["id"] for s in rows("statutes", "statute_candidates", cfg)[:15]]
    names = [s.get("provision_name") for s in rows("statutes", "statute_candidates", cfg)[:15]]
    print("\nstatute ids:", ids)
    print("provision names -> parsed:")
    for name in names:
        print("  ", name, "->", parse_provision(name or ""))


# --------------------------------------------------------------------------- documents


def _date(value) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _as_list(value) -> list:
    """A list column, also when an export stored it as a string ("['a', 'b']")."""
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        if v.startswith("[") and v.endswith("]"):
            import ast

            try:
                return list(ast.literal_eval(v))
            except (ValueError, SyntaxError):
                pass
        return [value]
    return list(value)


def _roles(row: dict) -> list[str] | None:
    roles = _as_list(row.get("rhetorical_roles", row.get("rr")))
    return roles or None


def _paragraph_texts(row: dict) -> list[str]:
    return [str(t) for t in _as_list(row.get("text"))]


@lru_cache(maxsize=4)
def _statute_refs(root: str) -> dict[str, str]:
    """IL-PCSR statute id -> canonical section ref ('ipc:302')."""
    out = {}
    for row in _rows("statutes", "statute_candidates", root):
        code, section, _ = parse_provision(row.get("provision_name") or row["id"])
        out[row["id"]] = section_ref(code, section) if section else f"{code}:{row['id']}"
    return out


def _judgment(row: dict, doc_type: DocType, split: str | None, cfg: Config) -> Document:
    refs = _statute_refs(str(_raw_dir(cfg)))
    paragraphs = segment_judgment(_paragraph_texts(row), _roles(row))
    return Document(
        doc_id=str(row["id"]),
        doc_type=doc_type,
        title=row.get("case_title") or "",
        paragraphs=paragraphs,
        decision_date=_date(row.get("date")),
        statutes_cited=sorted({refs.get(s, s) for s in _as_list(row.get("relevant_statute_ids"))}),
        precedents_cited=[str(p) for p in _as_list(row.get("relevant_precedent_ids"))],
        split=split,
        meta={"jurisdiction": row.get("jurisdiction") or "", "source": "il-pcsr"},
    )


def load_queries(split: str, cfg: Config | None = None) -> list[Document]:
    """Query judgments for 'train' | 'val' | 'test' (citations masked in the text)."""
    cfg = cfg or load_config()
    ours = "val" if split in ("val", "dev") else split
    return [_judgment(r, DocType.QUERY_CASE, ours, cfg) for r in rows("queries", SPLIT_NAMES[split], cfg)]


def load_precedents(cfg: Config | None = None) -> list[Document]:
    """The 3,183 candidate precedents (unmasked; their own citations kept)."""
    cfg = cfg or load_config()
    return [_judgment(r, DocType.PRECEDENT, None, cfg) for r in rows("precedents", "precedent_candidates", cfg)]


def load_statute_candidates(cfg: Config | None = None) -> list[Document]:
    """The 936 statute candidates, with code + section parsed from provision_name."""
    cfg = cfg or load_config()
    out = []
    for r in rows("statutes", "statute_candidates", cfg):
        name = r.get("provision_name") or str(r["id"])
        code, section, act = parse_provision(name)
        texts = _paragraph_texts(r)
        out.append(Document(
            doc_id=str(r["id"]),
            doc_type=DocType.STATUTE,
            title=name,
            paragraphs=[Paragraph(t, "statute", i) for i, t in enumerate(texts)],
            code=_CODE_ENUM.get(code, Code.OTHER),
            section=section,
            meta={"act": act, "act_code": code, "ref": section_ref(code, section) if section else "", "source": "il-pcsr"},
        ))
    return out


def load_qrels(split: str, target: str = "precedent", cfg: Config | None = None) -> dict[str, dict[str, int]]:
    """Gold citations per query. target: 'precedent' (doc ids) or 'statute' (statute doc ids).

    Rule 3: use 'train' for building things, 'val' for tuning, 'test' only for reporting.
    """
    cfg = cfg or load_config()
    key = "relevant_precedent_ids" if target == "precedent" else "relevant_statute_ids"
    return {str(r["id"]): {str(d): 1 for d in _as_list(r.get(key))} for r in rows("queries", SPLIT_NAMES[split], cfg)}


def iter_all(cfg: Config | None = None) -> Iterable[Document]:
    """Every IL-PCSR document: queries (all splits), precedents, statutes."""
    cfg = cfg or load_config()
    for split in ("train", "val", "test"):
        yield from load_queries(split, cfg)
    yield from load_precedents(cfg)
    yield from load_statute_candidates(cfg)


if __name__ == "__main__":
    inspect()
