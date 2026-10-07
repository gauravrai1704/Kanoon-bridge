"""Boolean, proximity and query parsing.  [A]"""

from kanoon_bridge.index.positional import PositionalIndex
from kanoon_bridge.query.boolean import intersect, intersect_many
from kanoon_bridge.query.parser import extract_filters, has_operators
from kanoon_bridge.query.proximity import within


def test_filters_extracted():
    text, f = extract_filters(
        "bail dowry state:Delhi date:2025-03-01 code:bns"
    )
    assert text == "bail dowry"
    assert f == {
        "state": "delhi",
        "date": "2025-03-01",
        "code": "bns",
    }


def test_subsection_is_not_an_operator():
    assert not has_operators("section 103(1) bns")
    assert not has_operators("charged u/s 302")
    assert has_operators("bail /5 parity")
    assert has_operators('"criminal breach of trust"')


def test_intersect_linear_merge():
    assert intersect(
        ["a", "c", "e"],
        ["b", "c", "e", "f"],
    ) == ["c", "e"]


def test_intersect_many_empty_short_circuit():
    assert intersect_many(
        [["a"], ["b"], ["a", "b"]]
    ) == []


def test_proximity_window():
    idx = PositionalIndex()
    idx.add("d1", ["bail", "granted", "on", "parity"])
    idx.add("d2", ["bail"] + ["x"] * 10 + ["parity"])

    assert within(
        idx,
        "bail",
        "parity",
        5,
    ) == {"d1"}