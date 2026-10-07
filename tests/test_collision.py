"""Collision resolution for bare section numbers.  [Gaurav]"""

from datetime import date

from kanoon_bridge.text.collision import CollisionResolver
from kanoon_bridge.text.pipeline import analyze_text


def resolver():
    return CollisionResolver.load()


def test_code_for_date():
    r = resolver()
    assert r.code_for_date(date(2024, 6, 30)) == "ipc"
    assert r.code_for_date(date(2024, 7, 1)) == "bns"
    assert r.code_for_date(None) is None


def test_date_decides_when_no_context():
    r = resolver()
    assert r.resolve("302", [], date(2023, 1, 1))[0].section_ref == "ipc:302"
    assert r.resolve("302", [], date(2025, 1, 1))[0].section_ref == "bns:302"


def test_context_beats_missing_date():
    r = resolver()
    assert r.resolve("302", ["murder", "stab"], None)[0].section_ref == "ipc:302"
    assert r.resolve("302", ["religi", "feel"], None)[0].section_ref == "bns:302"


def test_strong_context_overrides_date():
    # a 2025 query about murder that says "section 302" most likely means the old IPC 302
    assert resolver().resolve("302", ["murder"], date(2025, 1, 1))[0].section_ref == "ipc:302"


def test_confidences_sum_to_one():
    readings = resolver().resolve("302", [], None)
    assert abs(sum(x.confidence for x in readings) - 1.0) < 1e-9


def test_number_only_in_one_code():
    # IPC has no section 358 equivalent issue; BNS stops at 358, so 420 exists only in IPC
    assert resolver().resolve("420", [], date(2025, 1, 1))[0].section_ref == "ipc:420"


def test_pipeline_resolves_bare_sections():
    toks = analyze_text("He stabbed and murdered the victim; charged under section 302", date=date(2020, 1, 1))
    assert "sec:ipc:302" in toks and any(t.startswith("off:murder") for t in toks)
