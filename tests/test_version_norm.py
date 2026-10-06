"""Version normalisation: both codes reach the same offence.  [B]"""

from conftest import todo

from kanoon_bridge.text.version_norm import VersionNormalizer


def norm():
    return VersionNormalizer.load()


def test_seed_files_load():
    n = norm()
    assert "off:murder" in n.offences
    assert "off:murder" in n.section_to_offences["ipc:302"]


@todo
def test_ipc_and_bns_meet():
    n = norm()
    a = {o for o, _ in n.to_offences("ipc:302")}
    b = {o for o, _ in n.to_offences("bns:103(1)")}
    assert a == b == {"off:murder"}


@todo
def test_subsection_falls_back_to_base():
    assert {o for o, _ in norm().to_offences("bns:103(2)")} == {"off:murder"}


@todo
def test_equivalents_cross_code():
    assert "bns:103(1)" in norm().equivalents("ipc:302")


@todo
def test_collision_numbers_differ():
    n = norm()
    assert {o for o, _ in n.to_offences("ipc:302")} != {o for o, _ in n.to_offences("bns:302")}


@todo
def test_normalize_tokens_appends_offence():
    assert norm().normalize_tokens(["stabbed", "sec:ipc:302"]) == ["stabbed", "sec:ipc:302", "off:murder"]
