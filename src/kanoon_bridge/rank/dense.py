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

import numpy as np

from kanoon_bridge.config import Config, load_config
from kanoon_bridge.schema import Document


@dataclass
class DenseRetriever:
    cfg: Config
    model: object | None = None                     # sentence_transformers.SentenceTransformer (or any .encode())
    matrix: np.ndarray | None = None                # paragraph embeddings, L2-normalised
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
    def load(cls, cfg: Config | None = None, name: str = "precedents", model=None) -> "DenseRetriever":
        """Load pre-computed paragraph embeddings (scripts/04_encode_dense.py); the model itself
        is loaded lazily on the first query (or pass `model`)."""
        import json

        cfg = cfg or load_config()
        npy, ids = cls._paths(cfg, name)
        if not npy.exists():
            raise FileNotFoundError(f"{npy} not found - run scripts/04_encode_dense.py first (make dense)")
        r = cls(cfg=cfg, model=model, matrix=np.load(npy))
        r.para_doc = json.loads(ids.read_text(encoding="utf-8"))
        r._index_rows()
        return r

    def encode_corpus(self, docs: Iterable[Document], name: str = "precedents", batch_size: int = 32) -> None:
        """Encode every zone paragraph (each cut to dense.max_paragraph_tokens words) and save
        <name>.npy + <name>_ids.json. Called once by scripts/04_encode_dense.py."""
        import json

        max_words = int(self.cfg.dense.get("max_paragraph_tokens", 512))
        texts, owners = [], []
        for doc in docs:
            for para in doc.paragraphs:
                words = para.text.split()
                if words:
                    texts.append(" ".join(words[:max_words]))
                    owners.append(doc.doc_id)
        self.matrix = self._encode(texts, "passage", batch_size) if texts else np.zeros((0, 1), dtype=np.float32)
        self.para_doc = owners
        self._index_rows()
        npy, ids = self._paths(self.cfg, name)
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, self.matrix)
        ids.write_text(json.dumps(owners), encoding="utf-8")

    def score(self, text: str, candidates: set[str] | None = None) -> dict[str, float]:
        """MaxP: cosine of the query with each paragraph; a document scores its best paragraph."""
        if self.matrix is None or not len(self.para_doc):
            return {}
        q = self._encode([text], "query")[0]
        sims = self.matrix @ q
        docs = self._rows if candidates is None else {d: r for d, r in self._rows.items() if d in candidates}
        return {d: float(sims[rows].max()) for d, rows in docs.items()}
