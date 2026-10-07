"""Dense channel: multilingual paragraph embeddings.  [owner: C — working]

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
import json

import numpy as np

from kanoon_bridge.config import Config, load_config
from kanoon_bridge.schema import Document


@dataclass
class DenseRetriever:
    cfg: Config
    model: object | None = None
    matrix: np.ndarray | None = None
    para_doc: list[str] = field(default_factory=list)
    _rows: dict[str, np.ndarray] = field(default_factory=dict)   # doc -> its paragraph row indices

    # ------------------------------------------------------------------ model
    def _model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as err:
                raise RuntimeError('dense channel needs:  pip install -e ".[dense]"') from err
            name = self.cfg.dense.model
            try:
                self.model = SentenceTransformer(name)
            except Exception:                                        # noqa: BLE001 - fall back, say so
                fallback = self.cfg.dense.get("fallback_model", "intfloat/multilingual-e5-base")
                print(f"dense: could not load {name}; using {fallback}")
                self.model = SentenceTransformer(fallback)
        return self.model

    def _prefix(self, kind: str) -> str:
        """E5-family models expect 'query: ' / 'passage: ' prefixes."""
        return f"{kind}: " if self.cfg.dense.get("e5_prefixes", True) else ""

    def _encode(self, texts: list[str], kind: str, batch_size: int = 32) -> np.ndarray:
        vecs = self._model().encode([self._prefix(kind) + t for t in texts], batch_size=batch_size,
                                    normalize_embeddings=True, show_progress_bar=len(texts) > 256)
        vecs = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)

    # ------------------------------------------------------------------ files
    @staticmethod
    def _paths(cfg: Config, name: str):
        from kanoon_bridge.config import project_path

        base = project_path(cfg.paths.embeddings)
        return base / f"{name}.npy", base / f"{name}_ids.json"

    def _index_rows(self) -> None:
        rows: dict[str, list[int]] = {}
        for i, d in enumerate(self.para_doc):
            rows.setdefault(d, []).append(i)
        self._rows = {d: np.asarray(r) for d, r in rows.items()}

    @classmethod
    def load(
        cls,
        cfg: Config | None = None,
        name: str = "precedents",
    ) -> "DenseRetriever":
        """Load the model and pre-computed embeddings."""
        cfg = cfg or load_config()

        from sentence_transformers import SentenceTransformer

        retriever = cls(cfg=cfg)
        retriever.model = SentenceTransformer(cfg.dense.model)
        retriever.model.max_seq_length = cfg.dense.max_paragraph_tokens
        embeddings_dir = cfg.paths.embeddings
        matrix_path = embeddings_dir / f"{name}.npy"
        ids_path = embeddings_dir / f"{name}_ids.json"

        retriever.matrix = np.load(matrix_path).astype(np.float32)

        with open(ids_path, encoding="utf-8") as f:
            retriever.para_doc = json.load(f)

        if len(retriever.matrix) != len(retriever.para_doc):
            raise ValueError(
                f"Embedding/ID count mismatch: "
                f"{len(retriever.matrix)} embeddings vs "
                f"{len(retriever.para_doc)} document IDs"
            )

        return retriever

    def encode_corpus(
        self,
        docs: Iterable[Document],
        name: str = "precedents",
        batch_size: int = 32,
    ) -> None:
        """Encode every paragraph and save normalized embeddings."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.cfg.dense.model)
            self.model.max_seq_length = self.cfg.dense.max_paragraph_tokens

        texts: list[str] = []
        para_doc: list[str] = []

        for doc in docs:
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()

                if not text:
                    continue

                texts.append(f"passage: {text}")
                para_doc.append(doc.doc_id)

        if not texts:
            raise ValueError("No non-empty paragraphs found")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        self.matrix = np.asarray(embeddings, dtype=np.float32)
        self.para_doc = para_doc

        embeddings_dir = self.cfg.paths.embeddings
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        np.save(
            embeddings_dir / f"{name}.npy",
            self.matrix,
        )

        with open(
            embeddings_dir / f"{name}_ids.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.para_doc,
                f,
                ensure_ascii=False,
                indent=2,
            )

    def score(
        self,
        text: str,
        candidates: set[str] | None = None,
    ) -> dict[str, float]:
        """Return MaxP cosine score for each document."""
        if self.model is None:
            raise RuntimeError("DenseRetriever model is not loaded")

        if self.matrix is None or not self.para_doc:
            raise RuntimeError("Dense embeddings are not loaded")

        query_embedding = self.model.encode(
            [f"query: {text}"],
            normalize_embeddings=True,
        )[0]

        # Both vectors are L2-normalized, so dot product = cosine similarity.
        paragraph_scores = self.matrix @ np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        best: dict[str, float] = {}

        for i, doc_id in enumerate(self.para_doc):
            if candidates is not None and doc_id not in candidates:
                continue

            score = float(paragraph_scores[i])

            # MaxP: a document receives the score of its best paragraph.
            if doc_id not in best or score > best[doc_id]:
                best[doc_id] = score

        return best
