"""Dense channel with a tiny fake encoder (no model download).  [C]"""

import numpy as np

from kanoon_bridge.config import load_config
from kanoon_bridge.rank.dense import DenseRetriever
from kanoon_bridge.schema import Document, DocType, Paragraph

VOCAB = ["murder", "knife", "bail", "theft"]


class FakeModel:
    def __init__(self):
        self.seen = []

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
        self.seen += texts
        return np.array([[t.lower().count(w) + 0.01 for w in VOCAB] for t in texts], dtype=np.float32)


def test_encode_save_load_maxp(tmp_path):
    cfg = load_config(overrides={"paths": {"embeddings": str(tmp_path)}})
    docs = [Document("a", DocType.PRECEDENT, paragraphs=[Paragraph("bail granted", "decision"),
                                                         Paragraph("murder with a knife", "ratio")]),
            Document("b", DocType.PRECEDENT, paragraphs=[Paragraph("theft of a phone", "facts")])]
    r = DenseRetriever(cfg=cfg, model=FakeModel())
    r.encode_corpus(docs)
    assert (tmp_path / "precedents.npy").exists() and r.para_doc == ["a", "a", "b"]
    assert r.model.seen[0].startswith("passage: ")

    loaded = DenseRetriever.load(cfg, model=FakeModel())
    scores = loaded.score("knife murder")
    assert loaded.model.seen == ["query: knife murder"]
    assert max(scores, key=scores.get) == "a"                   # MaxP: its best paragraph wins
    assert set(loaded.score("theft", candidates={"b"})) == {"b"}
    assert -1.0 <= scores["b"] <= 1.0
