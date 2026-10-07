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


def test_counts_only_index_and_compact_positions():
    import pytest
    from array import array

    from kanoon_bridge.index.positional import count

    idx = PositionalIndex(positions=False)
    idx.add("d1", ["bail", "bail", "parity"])
    assert idx.postings["bail"]["d1"] == 2 and idx.tf("bail", "d1") == 2 and idx.df("parity") == 1
    with pytest.raises(ValueError):
        idx.phrase(["bail", "parity"])
    full = build()
    assert isinstance(full.postings["criminal"]["d1"], array) and list(full.postings["criminal"]["d1"]) == [0]
    assert count([1, 2, 3]) == 3 and count(5) == 5          # old pickles (lists) still work


def test_zone_index_keeps_positions_only_in_whole():
    from kanoon_bridge.index.zones import ZoneIndex
    from kanoon_bridge.schema import Document, DocType, Paragraph

    z = ZoneIndex.build([Document("d", DocType.PRECEDENT, paragraphs=[Paragraph("a b a", "facts")])],
                        lambda text, doc: text.split())
    assert z.zones["facts"].postings["a"]["d"] == 2 and z.tf("a", "d", "facts") == 2
    assert list(z.whole.postings["a"]["d"]) == [0, 2]
