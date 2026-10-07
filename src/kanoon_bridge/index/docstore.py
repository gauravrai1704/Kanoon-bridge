"""Document store: doc_id -> Document, loaded from data/processed/docs.jsonl.  [shared — working]

The indexes hold terms, not text. Anything that needs the text back (agent reformulation,
RAG chunks, the judging tool, the app's snippets) gets it here.

    docs = DocStore.load()
    docs["bns:103"].title
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from kanoon_bridge.config import Config, load_config, project_path
from kanoon_bridge.schema import Document, read_documents


@dataclass
class DocStore(Mapping):
    docs: dict[str, Document] = field(default_factory=dict)

    @classmethod
    def load(cls, cfg: Config | None = None, include_queries: bool = False) -> "DocStore":
        cfg = cfg or load_config()
        path = project_path(cfg.paths.docs)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run scripts/01_build_corpus.py first")
        docs = {d.doc_id: d for d in read_documents(path)
                if include_queries or d.doc_type.value != "query_case"}
        return cls(docs=docs)

    def __getitem__(self, doc_id: str) -> Document:
        return self.docs[doc_id]

    def __iter__(self):
        return iter(self.docs)

    def __len__(self) -> int:
        return len(self.docs)

    def snippet(self, doc_id: str, chars: int = 300) -> str:
        d = self.docs.get(doc_id)
        return (d.text[:chars] + "...") if d and len(d.text) > chars else (d.text if d else "")
