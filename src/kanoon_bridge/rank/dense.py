"""Dense channel: multilingual paragraph embeddings.  [owner: C]

Model: NLLB-E5 (Hindi-BEIR, NAACL 2025) — zero-shot multilingual, handles Hindi without
Hindi training data. Fallback: intfloat/multilingual-e5-base. Set in configs: dense.model.

Precedents average ~7,500 words, far past the encoder limit, so:
  * encode each zone paragraph separately (scripts/04_encode_dense.py, ONCE, on a GPU)
  * document score = max cosine over its paragraphs (MaxP)

Files written by encode_corpus():
    data/processed/embeddings/<name>.npy        float32 [n_paragraphs, dim], L2-normalised
    data/processed/embeddings/<name>_ids.json   paragraph i -> doc_id

Needs: pip install -e ".[dense]"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from kanoon_bridge.config import Config, load_config
from kanoon_bridge.schema import Document


@dataclass
class DenseRetriever:
    cfg: Config
    model: object | None = None                     # sentence_transformers.SentenceTransformer
    matrix: np.ndarray | None = None                # paragraph embeddings
    para_doc: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, cfg: Config | None = None, name: str = "precedents") -> "DenseRetriever":
        """Load the model and pre-computed embeddings.

        TODO(C): lazy-import sentence_transformers; np.load the matrix; json-load ids.
        E5 models expect the prefixes "query: " and "passage: " — use them.
        """
        raise NotImplementedError("TODO(C): load dense model + embeddings")

    def encode_corpus(self, docs: Iterable[Document], name: str = "precedents", batch_size: int = 32) -> None:
        """Encode every paragraph and save (called by scripts/04_encode_dense.py). TODO(C)."""
        raise NotImplementedError("TODO(C): encode paragraphs")

    def score(self, text: str, candidates: set[str] | None = None) -> dict[str, float]:
        """MaxP cosine score per doc for the query text. TODO(C)."""
        raise NotImplementedError("TODO(C): dense MaxP scoring")
