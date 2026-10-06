"""Shared test helpers.

`todo` marks a test as a SPEC for code that is not written yet: while the function raises
NotImplementedError the test is SKIPPED (shown as 's'), and it runs for real as soon as the
owner implements it. So `make test` stays green, and the tests tell you when you're done.

`toy_docs` is a tiny corpus for index/ranking tests.
"""

from __future__ import annotations

import functools
from datetime import date

import pytest

from kanoon_bridge.schema import Code, Court, Document, DocType, Paragraph


def todo(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as err:
            pytest.skip(f"not implemented yet: {err}")
    return wrapper


@pytest.fixture
def toy_docs() -> list[Document]:
    def prec(doc_id, court, states, d, facts, ratio, cites):
        return Document(
            doc_id=doc_id, doc_type=DocType.PRECEDENT, court=court, states=states, decision_date=d,
            code=Code.IPC, statutes_cited=cites,
            paragraphs=[Paragraph(facts, "facts", 0), Paragraph(ratio, "ratio", 1)],
        )

    return [
        prec("p1", Court.SUPREME_COURT, ["*"], date(2010, 5, 1),
             "The accused stabbed the deceased with a knife.",
             "We hold that the offence of murder under Section 302 IPC is made out.", ["ipc:302"]),
        prec("p2", Court.HIGH_COURT, ["delhi"], date(2015, 3, 4),
             "The husband demanded dowry and the wife died within seven years of marriage.",
             "Dowry death under Section 304B IPC; bail refused.", ["ipc:304b"]),
        prec("p3", Court.HIGH_COURT, ["maharashtra"], date(2018, 7, 9),
             "The accused threatened the complainant with dire consequences.",
             "Criminal intimidation under Section 506 IPC is established.", ["ipc:506"]),
    ]
