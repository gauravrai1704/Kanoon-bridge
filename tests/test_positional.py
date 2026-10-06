"""Positional index and phrase queries.  [A]"""

from conftest import todo

from kanoon_bridge.index.positional import PositionalIndex


def build():
    idx = PositionalIndex()
    idx.add("d1", ["criminal", "breach", "of", "trust", "proved"])
    idx.add("d2", ["breach", "criminal", "trust"])
    return idx


@todo
def test_df_and_tf():
    idx = build()
    assert idx.df("criminal") == 2
    assert idx.tf("trust", "d1") == 1
    assert idx.doc_len["d1"] == 5


@todo
def test_phrase_requires_order():
    idx = build()
    assert idx.phrase(["criminal", "breach"]) == {"d1"}


@todo
def test_duplicate_doc_rejected():
    import pytest

    idx = build()
    with pytest.raises(ValueError):
        idx.add("d1", ["again"])
