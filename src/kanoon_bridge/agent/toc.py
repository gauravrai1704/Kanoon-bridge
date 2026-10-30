"""Structure-aware statute navigation: read the code like a table of contents.  [layer 2 v2 — working]

Inspired by STAIR (Kumar et al., "STructure Aware Information Retriever", IBM, arXiv 2609.03874),
which addresses a corpus through its table of contents instead of length-based chunks. STAIR
trains a generative retriever to walk the ToC; we use the statutes' own ToC with no training:

    code -> chapter -> (BNS: Gazette sub-heading) -> section heading

1. Every chapter is a ToC entry: its name, its sub-headings, and the headings of its sections.
2. The question (spelling-corrected) is matched to the entries of the codes it may concern
   (the routed family's code in force on the incident date, else both of that family's codes) by
   idf-weighted cosine; the top `top_c` chapters are chosen.
3. Only for questions that name no section (router kind fact_question). Statutes of the chosen chapters get w / (rrf_k + chapter rank) on top of their fused score;
   the best-matching section of the top chapter is added if retrieval missed it.

Chapters come from India Code (kanoonhq/open-laws: chapter_id, chapter_name per section) for all
six codes, and the BNS sub-headings from bns-study-platform's chapter files (ToC labels only;
its summaries and study notes are not used).
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

from kanoon_bridge.rag._util import cosine, idf_lookup, terms, tfidf


@dataclass
class Chapter:
    code: str
    chapter: str
    name: str
    labels: list[str] = field(default_factory=list)       # sub-headings (BNS)
    refs: list[str] = field(default_factory=list)         # "bns:103"
    headings: list[str] = field(default_factory=list)

    def entry(self) -> str:
        return " ".join([self.name, *self.labels, *self.headings])


@dataclass
class TableOfContents:
    chapters: list[Chapter]
    chapter_of: dict[str, int] = field(default_factory=dict)   # ref -> index into chapters

    @classmethod
    def build(cls, cfg) -> "TableOfContents":
        from kanoon_bridge.config import project_path
        from kanoon_bridge.drift.provisions import INDIA_CODE_ACTS

        path = project_path(cfg.paths.get("open_laws_dir", "data/raw/open-laws")) / "data" / "indian_central_act_sections.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing (python scripts/00_fetch_data.py --only india-code)")
        import pyarrow.parquet as pq

        act_code = {v: k for k, v in INDIA_CODE_ACTS.items()}
        rows = [r for r in pq.read_table(path, columns=["act_id", "section_number", "heading", "chapter_id",
                                                         "chapter_name"]).to_pylist() if r["act_id"] in act_code]
        chapters: dict[tuple, Chapter] = {}
        for r in rows:
            m = re.search(r"(\d+[A-Z]{0,3})", r["section_number"] or "")
            heading = (r["heading"] or "").strip().rstrip(".")
            if not m or not r["chapter_id"] or heading.lstrip("[ ").lower().startswith("repealed"):
                continue
            code = act_code[r["act_id"]]
            key = (code, r["chapter_id"])
            ch = chapters.setdefault(key, Chapter(code, r["chapter_id"], (r["chapter_name"] or "").title()))
            ch.refs.append(f"{code}:{m.group(1).lower()}")
            ch.headings.append(heading)
        bns_dir = project_path(cfg.paths.bns_dir) / "data" / "chapters"
        if bns_dir.exists():
            by_range = {}
            for f in sorted(bns_dir.glob("ch-*.json")):
                d = json.loads(f.read_text(encoding="utf-8"))
                subs = d.get("subHeadings") or []
                if isinstance(subs, str):
                    subs = ast.literal_eval(subs)
                by_range[str(d.get("number", "")).upper()] = [s.get("label", "") for s in subs if isinstance(s, dict)]
            for (code, cid), ch in chapters.items():
                if code == "bns":
                    roman = cid.replace("CHAPTER", "").strip().upper()
                    ch.labels = [x for x in by_range.get(roman, []) if x]
        out = cls(list(chapters.values()))
        for i, ch in enumerate(out.chapters):
            for ref in ch.refs:
                out.chapter_of.setdefault(ref, i)
        return out

    def navigate(self, question: str, codes: set[str], idf=None, top_c: int = 2) -> list[tuple[int, float]]:
        """(chapter index, score) of the best `top_c` entries among `codes` (score > 0)."""
        look = idf_lookup(idf)
        q = tfidf(terms(question), look)
        scored = [(i, cosine(q, tfidf(terms(ch.entry()), look))) for i, ch in enumerate(self.chapters)
                  if ch.code in codes]
        scored = [s for s in scored if s[1] > 0]
        return sorted(scored, key=lambda s: -s[1])[:top_c]


__all__ = ["TableOfContents", "Chapter"]
