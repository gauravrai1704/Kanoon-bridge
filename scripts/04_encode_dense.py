"""Pre-compute paragraph embeddings ONCE (GPU recommended).  [owner: C, wiring working]

Needs: pip install -e ".[dense]". Then set dense.enabled: true in configs/default.yaml.
Run this early (hours 14-18); it can take a while on CPU for ~3k long precedents.

    python scripts/04_encode_dense.py
"""

from __future__ import annotations

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.rank.dense import DenseRetriever
from kanoon_bridge.schema import DocType, read_documents


def main() -> None:
    cfg = load_config()
    docs = [d for d in read_documents(project_path(cfg.paths.docs)) if d.doc_type == DocType.PRECEDENT]
    statutes = [d for d in read_documents(project_path(cfg.paths.docs)) if d.doc_type == DocType.STATUTE]
    retriever = DenseRetriever(cfg=cfg)
    retriever.encode_corpus(docs, name="precedents")
    retriever.encode_corpus(statutes, name="statutes")
    print("embeddings saved to", project_path(cfg.paths.embeddings))


if __name__ == "__main__":
    main()
