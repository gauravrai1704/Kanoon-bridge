"""Collision resolution for bare section numbers.  [B]"""

from datetime import date

from conftest import todo

from kanoon_bridge.text.collision import CollisionResolver


def test_code_for_date():
    r = CollisionResolver.load()
    assert r.code_for_date(date(2024, 6, 30)) == "ipc"
    assert r.code_for_date(date(2024, 7, 1)) == "bns"
    assert r.code_for_date(None) is None


@todo
def test_date_decides_when_no_context():
    r = CollisionResolver.load()
    assert r.resolve("302", [], date(2023, 1, 1))[0].section_ref == "ipc:302"
    assert r.resolve("302", [], date(2025, 1, 1))[0].section_ref == "bns:302"


@todo
def test_context_beats_missing_date():
    r = CollisionResolver.load()
    top = r.resolve("302", ["murder", "stabbed"], None)[0]
    assert top.section_ref == "ipc:302"


@todo
def test_confidences_sum_to_one():
    readings = CollisionResolver.load().resolve("302", [], None)
    assert abs(sum(x.confidence for x in readings) - 1.0) < 1e-6
