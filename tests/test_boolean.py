"""Boolean, proximity and query parsing.  [A]"""

from conftest import todo

from kanoon_bridge.index.positional import PositionalIndex
from kanoon_bridge.query.boolean import intersect, intersect_many
from kanoon_bridge.query.parser import extract_filters, has_operators
from kanoon_bridge.query.proximity import within


def test_filters_extracted():
    text, f = extract_filters("bail dowry state:Delhi date:2025-03-01 code:bns")
    assert text == "bail dowry"
    assert f == {"state": "delhi", "date": "2025-03-01", "code": "bns"}


def test_subsection_is_not_an_operator():
    assert not has_operators("section 103(1) bns")
    assert not has_operators("charged u/s 302")
    assert has_operators("bail /5 parity")
    assert has_operators('"criminal breach of trust"')


@todo
def test_intersect_linear_merge():
    assert intersect(["a", "c", "e"], ["b", "c", "e", "f"]) == ["c", "e"]


@todo
def test_intersect_many_empty_short_circuit():
    assert intersect_many([["a"], ["b"], ["a", "b"]]) == []


@todo
def test_proximity_window():
    idx = PositionalIndex()
    idx.add("d1", ["bail", "granted", "on", "parity"])
    idx.add("d2", ["bail"] + ["x"] * 10 + ["parity"])
    assert within(idx, "bail", "parity", 5) == {"d1"}


# ---------------------------------------------------------------- parser, evaluation, search wiring
from kanoon_bridge.query.boolean import evaluate  # noqa: E402
from kanoon_bridge.query.parser import Op, parse, ranking_text, show  # noqa: E402


def test_parser_precedence_and_implicit_and():
    tree = parse('bail OR "dowry death" NOT acquittal state:delhi').tree
    assert tree.op == Op.OR
    assert show(tree) == '(bail OR ("dowry death" AND NOT acquittal))'
    assert show(parse("bail AND (parity OR delay) /5 custody").tree) == "(bail AND (parity OR delay) /5 custody)"
    assert parse("bail /5 parity").tree.k == 5


def test_parser_keeps_subsections_and_lowercase_words():
    p = parse("murder AND 103(1) bns and knife")
    assert show(p.tree) == "(murder AND 103(1) bns AND and AND knife)"     # section mention = one word
    assert parse("plain words only").tree is None


def test_parser_malformed_falls_back_to_free_text():
    p = parse("bail AND (parity")
    assert p.tree is None and p.text_for_ranking == "bail AND (parity"


def test_ranking_text_skips_negated_words():
    assert ranking_text(parse('bail AND NOT "anticipatory bail" OR parity').tree) == "bail parity"


def _idx():
    idx = PositionalIndex()
    idx.add("d1", ["bail", "grant", "parity", "dowri", "death"])
    idx.add("d2", ["bail", "reject", "dowri", "death", "acquitt"])
    idx.add("d3", ["murder", "off:murder_bns103", "knife"])
    idx.add("d4", ["dowri", "x", "x", "death"])
    return idx


def _an(text):
    from kanoon_bridge.text.pipeline import analyze_text

    return analyze_text(text)


def test_evaluate_operators():
    idx = _idx()
    q = lambda s: evaluate(parse(s).tree, idx, _an)  # noqa: E731
    assert q('"dowry death" AND NOT acquittal') == {"d1"}
    assert q("bail OR knife") == {"d1", "d2", "d3"}
    assert q("dowry /1 death") == {"d1", "d2"}
    assert q("dowry /3 death") == {"d1", "d2", "d4"}
    assert q("BNS 103 AND knife") == {"d3"}                  # section -> offence id (version-aware)
    assert q("IPC 302 OR nothing") == {"d3"}
    assert q("NOT bail") == {"d3", "d4"}


def test_facets_filter_and_date_range():
    from datetime import date

    from kanoon_bridge.index.facets import FacetIndex
    from kanoon_bridge.schema import Court, Document, DocType

    docs = [Document("a", DocType.PRECEDENT, court=Court.SUPREME_COURT, states=["*"], decision_date=date(2010, 1, 1),
                     statutes_cited=["ipc:302"]),
            Document("b", DocType.PRECEDENT, court=Court.HIGH_COURT, states=["delhi"], decision_date=date(2020, 1, 1)),
            Document("c", DocType.PRECEDENT, court=Court.HIGH_COURT, states=["maharashtra"], decision_date=date(2025, 1, 1)),
            Document("s", DocType.STATUTE, decision_date=date(2024, 7, 1))]
    fx = FacetIndex.build(docs)
    assert fx.filter(states=["delhi"]) == {"a", "b"}
    assert fx.filter(doc_type="precedent", date_from=date(2015, 1, 1)) == {"b", "c"}
    assert fx.filter(date_from="2019-12-31", date_to="2020-01-01") == {"b"}
    assert fx.filter(cites_any=["ipc:302"]) == {"a"}
    assert fx.filter() == {"a", "b", "c", "s"}
    assert fx.filter(court="high_court", states=["maharashtra"]) == {"c"}


def test_long_text_is_never_boolean():
    from kanoon_bridge.query.parser import has_operators, parse

    doc = 'The court held "that the accused" AND (others) were liable. ' * 20
    assert not has_operators(doc)
    assert not parse(doc).is_boolean
    assert has_operators('"dowry death" AND bail')
