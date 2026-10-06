"""Zone index: one positional index per zone, plus per-zone lengths.  [owner: A]

Feeds BM25F (rank/bm25f.py), which needs, for each (term, doc, zone): tf, zone length,
and the average length of that zone across the collection.

Zones come from ingest/segment.py and are listed in schema.ZONES. Statutes use the
single zone "statute".

    zidx = ZoneIndex.build(docs, analyze=lambda text, doc: analyze_text(text, date=doc.decision_date))
    zidx.tf("off:murder", "d1", "ratio")
    zidx.zone_len("d1", "ratio"); zidx.avg_zone_len("ratio")
    zidx.whole                       # PositionalIndex over all zones (for Boolean/phrase)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from kanoon_bridge.index.positional import PositionalIndex
from kanoon_bridge.schema import ZONES, Document

Analyzer = Callable[[str, Document], list[str]]


@dataclass
class ZoneIndex:
    zones: dict[str, PositionalIndex] = field(default_factory=lambda: {z: PositionalIndex() for z in ZONES})
    whole: PositionalIndex = field(default_factory=PositionalIndex)
    doc_ids: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, docs: Iterable[Document], analyze: Analyzer) -> "ZoneIndex":
        """Index every paragraph under its zone, and the full text in `whole`.

        TODO(A): for each doc, concatenate tokens of paragraphs in the same zone (keep
        positions continuous within a zone), add to zones[zone]; add all tokens to `whole`.
        Use tqdm for progress; the precedent corpus has ~3k long documents.
        """
        raise NotImplementedError("TODO(A): build zone index")

    # ------------------------------------------------------------------ accessors (implement with build)
    def tf(self, term: str, doc_id: str, zone: str) -> int:
        return self.zones[zone].tf(term, doc_id)

    def zone_len(self, doc_id: str, zone: str) -> int:
        return self.zones[zone].doc_len.get(doc_id, 0)

    def avg_zone_len(self, zone: str) -> float:
        lengths = self.zones[zone].doc_len.values()
        return sum(lengths) / len(lengths) if lengths else 0.0

    def df(self, term: str) -> int:
        """Document frequency over the whole document (what idf uses in BM25F)."""
        return self.whole.df(term)

    @property
    def n_docs(self) -> int:
        return self.whole.n_docs
