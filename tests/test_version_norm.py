"""Version normalisation and crosswalk parsing: both codes reach the same offence.  [Gaurav]"""

from kanoon_bridge.ingest.parse_crosswalk import parse_ipc_reference
from kanoon_bridge.text.version_norm import VersionNormalizer


def norm():
    return VersionNormalizer.load()


def test_ipc_and_bns_meet():
    n = norm()
    assert n.offences_for("ipc:302") == n.offences_for("bns:103(1)") == n.offences_for("bns:103")
    assert n.label(n.offences_for("ipc:302")[0]) == "Punishment for murder"


def test_collision_numbers_differ():
    n = norm()
    assert set(n.offences_for("ipc:302")) != set(n.offences_for("bns:302"))
    assert n.offences_for("bns:302") == n.offences_for("ipc:298")


def test_equivalents_cross_code():
    n = norm()
    assert n.equivalents("ipc:302") == ["bns:103"]
    assert n.equivalents("bns:302") == ["ipc:298"]
    assert set(n.equivalents("ipc:498a")) == {"bns:85", "bns:86"}


def test_other_acts_and_unknown_pass_through():
    n = norm()
    assert n.to_offences("constitution:21") == [] and n.to_offences("?:302") == []


def test_procedure_and_evidence_are_bridged():
    n = norm()
    if not n.offences_for("crpc:438"):
        import pytest

        pytest.skip("provision_ids.csv not built (scripts/01_build_corpus.py with the indian-legal-mcp checkout)")
    assert n.equivalents("crpc:438") == ["bnss:482"] and n.equivalents("bnss:482") == ["crpc:438"]
    assert n.equivalents("iea:65b") == ["bsa:63"]
    assert n.offences_for("crpc:438") == n.offences_for("bnss:482")


def test_align_codes_on_toy_acts():
    from kanoon_bridge.ingest.align_codes import Section, align, build_provision_ids, spot_check

    old = [Section("438", "Direction for grant of bail to person apprehending arrest", "when any person has reason to believe that he may be arrested on accusation of having committed a non-bailable offence he may apply"),
           Section("154", "Information in cognizable cases", "every information relating to the commission of a cognizable offence if given orally to an officer in charge of a police station")]
    new = [Section("173", "Information in cognizable cases", "every information relating to the commission of a cognizable offence irrespective of the area where the offence is committed may be given orally or by electronic communication to an officer in charge of a police station"),
           Section("482", "Direction for grant of bail to person apprehending arrest", "when any person has reason to believe that he may be arrested on an accusation of having committed a non-bailable offence he may apply"),
           Section("530", "Trial and proceedings to be held in electronic mode", "all trials inquiries and proceedings may be held in electronic mode by use of electronic communication")]
    rows = align(old, new)
    assert spot_check(rows, [("438", "482"), ("154", "173")])["accuracy"] == 1.0
    assert any(r["new"] == "530" and r["relation"] == "new" for r in rows)
    ids = build_provision_ids(rows, "crpc", "bnss")
    assert any(r["old_sections"] == "438" and r["new_sections"] == "482" for r in ids)


def test_normalize_tokens_appends_offence():
    n = norm()
    out = n.normalize_tokens(["stab", "sec:ipc:302"])
    assert out[:2] == ["stab", "sec:ipc:302"] and out[2].startswith("off:murder")


def test_parse_ipc_reference_shapes():
    assert parse_ipc_reference("302 IPC") == (["302"], "")
    assert parse_ipc_reference("340, 342-348 IPC")[0] == ["340", "342", "343", "344", "345", "346", "347", "348"]
    assert parse_ipc_reference("228A(1), (2) IPC")[0] == ["228a(1)", "228a(2)"]
    assert parse_ipc_reference("171-I IPC")[0] == ["171i"]
    assert parse_ipc_reference("120A and 120B IPC")[0] == ["120a", "120b"]
    assert parse_ipc_reference("498A IPC (Explanation)") == (["498a"], "explanation")
    assert parse_ipc_reference("No IPC equivalent") == ([], "new")
