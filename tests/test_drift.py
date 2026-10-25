"""Statutory drift (drift/): punishment parsing, elements, alignment, classes, carry-over, transfer.

Uses hand-written IPC/BNS texts so it runs without the raw data; the committed drift CSVs are
checked separately when present.
"""

from __future__ import annotations

from datetime import date

import pytest

from kanoon_bridge.drift import diff, elements, punishment
from kanoon_bridge.drift.model import DriftAnalyzer, DriftClass, DriftTable
from kanoon_bridge.drift.provisions import Provision, renumber_references, split_clauses


def test_words_to_number():
    assert punishment.words_to_number("two thousand and five hundred") == 2500
    assert punishment.words_to_number("ten lakh") == 1_000_000
    assert punishment.words_to_number("twenty-five thousand") == 25000
    assert punishment.words_to_number("1,000") == 1000
    assert punishment.words_to_number("banana") is None


def test_extract_punishment():
    p = punishment.extract("shall be punished with imprisonment of either description for a term which may extend "
                           "to one year, or with fine which may extend to one thousand rupees, or with both.")
    assert p.max_years == 1 and p.fine and p.fine_max == 1000 and not p.death
    q = punishment.extract("shall be punished with rigorous imprisonment for a term which shall not be less than ten "
                           "years, but which may extend to imprisonment for life, and shall also be liable to fine")
    assert q.min_years == 10 and q.life and q.rigorous
    m = punishment.extract("Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.")
    assert m.death and m.life and m.fine
    assert punishment.extract("A person is said to abet an offence who instigates it.").empty


def test_compare_punishment_direction():
    a = punishment.extract("punished with imprisonment for a term which may extend to two years, or with fine")
    b = punishment.extract("punished with imprisonment for a term which may extend to five years, and shall also be liable to fine")
    d = punishment.compare(a, b)
    assert "max_years" in d.changed and d.direction == "harsher"
    assert not punishment.compare(a, a).any
    c = punishment.extract("punished with simple imprisonment for a term which may extend to one year, or with fine, or with community service")
    assert "community_service" in punishment.compare(punishment.extract("punished with simple imprisonment for a term which may extend to one year, or with fine"), c).lighter


def test_elements():
    assert elements.mens_rea("Whoever dishonestly and fraudulently, knowing it to be false") >= {"dishonesty", "fraud", "knowledge"}
    th = elements.thresholds("a wife not being under eighteen years of age; within seven years of her marriage; five or more persons")
    assert ("age", 18.0) in th and ("period_years", 7.0) in th and ("count_persons", 5.0) in th
    d = elements.compare("his own wife, the wife not being under fifteen years of age", "his own wife, the wife not being under eighteen years of age")
    assert d.threshold_changed


def test_split_and_renumber():
    cl = split_clauses("(1) Whoever commits murder shall be punished with death. (2) When a group of five or more "
                       "persons acting in concert commits murder each member shall be punished. Explanation.—This applies to every member of the group.")
    assert len(cl) == 3 and cl[0].startswith("(1)")
    assert renumber_references("except in the case provided for by section 334", {"334": "122"}).endswith("section 122")


def test_align_retention_and_novelty():
    old = ["Whoever commits murder shall be punished with death or imprisonment for life and shall also be liable to fine"]
    new = ["(1) Whoever commits murder shall be punished with death or imprisonment for life and shall also be liable to fine",
           "(2) When a group of five or more persons acting in concert commits murder on the ground of race caste or community each member shall be punished"]
    al = diff.align(old, new)
    assert al.retention > 0.95
    assert len(al.added_clauses) == 1 and al.novelty_matched < 0.02 and al.novelty > 0.3


IPC = {
    "302": "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
    "323": "Whoever, except in the case provided for by section 334, voluntarily causes hurt, shall be punished with "
           "imprisonment of either description for a term which may extend to one year, or with fine which may extend "
           "to one thousand rupees, or with both.",
    "124a": "Whoever by words, either spoken or written, or by signs, brings or attempts to bring into hatred or contempt, "
            "or excites or attempts to excite disaffection towards, the Government established by law in India, shall be "
            "punished with imprisonment for life, to which fine may be added, or with imprisonment which may extend to "
            "three years, to which fine may be added, or with fine.",
    "334": "Whoever voluntarily causes hurt on grave and sudden provocation shall be punished with imprisonment which may extend to one month.",
    "497": "Whoever has sexual intercourse with a person who is the wife of another man is guilty of the offence of adultery, "
           "and shall be punished with imprisonment for a term which may extend to five years.",
}
BNS = {
    "103": "(1) Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine. "
           "(2) When a group of five or more persons acting in concert commits murder on the ground of race, caste or "
           "community, each member of such group shall be punished with death or with imprisonment for life.",
    "115": "Whoever, except in the case provided for by section 122, voluntarily causes hurt, shall be punished with "
           "imprisonment of either description for a term which may extend to one year, or with fine which may extend "
           "to ten thousand rupees, or with both.",
    "152": "Whoever, purposely or knowingly, by words, either spoken or written, or by electronic communication or by use "
           "of financial mean, excites or attempts to excite, secession or armed rebellion or subversive activities, or "
           "endangers sovereignty or unity and integrity of India shall be punished with imprisonment for life or with "
           "imprisonment which may extend to seven years, and shall also be liable to fine.",
    "122": "Whoever voluntarily causes hurt on grave and sudden provocation shall be punished with imprisonment which may extend to one month.",
}


@pytest.fixture(scope="module")
def table():
    from kanoon_bridge.config import load_config
    from kanoon_bridge.text.version_norm import CrosswalkRow, VersionNormalizer

    ipc = {k: Provision("ipc", k, k, v, clauses=split_clauses(v)) for k, v in IPC.items()}
    bns = {k: Provision("bns", k, k, v, clauses=split_clauses(v)) for k, v in BNS.items()}
    norm = VersionNormalizer()
    norm.crosswalk = [CrosswalkRow("302", "103(1)", "same"), CrosswalkRow("323", "115", "same"),
                      CrosswalkRow("124a", "152", "modified"), CrosswalkRow("334", "122", "same")]
    norm.offences = {"off:murder": {"label": "murder", "ipc": ["302"], "bns": ["103"], "keywords": []},
                     "off:sedition": {"label": "sovereignty", "ipc": ["124a"], "bns": ["152"], "keywords": []}}
    norm.section_to_offences.update({"ipc:302": ["off:murder"], "bns:103": ["off:murder"],
                                     "ipc:124a": ["off:sedition"], "bns:152": ["off:sedition"]})
    an = DriftAnalyzer.from_sources(load_config(), ipc=ipc, bns=bns, normalizer=norm)
    t = DriftTable.build(an)
    t._norm = norm
    return t


def test_classes(table):
    assert table.sections["152"].drift_class == DriftClass.SUBSTANTIVE
    assert table.sections["115"].drift_class == DriftClass.PUNISHMENT      # fine limit raised, 334 -> 122 renumbered
    assert table.sections["122"].drift_class == DriftClass.IDENTICAL
    assert "497" in table.repealed and table.carry_over("ipc:497", "bns:103") == 0.0


def test_carry_over_order(table):
    murder = table.carry_over("ipc:302", "bns:103")
    hurt = table.carry_over("ipc:323", "bns:115")
    sedition = table.carry_over("ipc:124a", "bns:152")
    assert murder > 0.85                    # the added mob-lynching clause barely stales murder precedent
    assert murder > hurt > sedition
    assert sedition < 0.5


def test_csv_round_trip(table, tmp_path):
    table.save_csv(tmp_path)
    back = DriftTable.load(tmp_path)
    assert back.sections["152"].drift_class == DriftClass.SUBSTANTIVE
    assert back.carry_over("ipc:124a", "bns:152") == pytest.approx(table.carry_over("ipc:124a", "bns:152"), abs=1e-4)
    assert "497" in back.repealed


def test_transfer_and_rerank(table):
    from kanoon_bridge.drift.transfer import TransferScorer
    from kanoon_bridge.index.facets import DocMeta
    from kanoon_bridge.schema import AnalyzedQuery, Code, Query

    ts = TransferScorer(table, table._norm, mu=0.5)
    aq = AnalyzedQuery(Query("speech against the state", incident_date=date(2025, 3, 1)),
                       code_in_force=Code.BNS, offence_ids=["off:sedition", "off:murder"])
    assert ts.bns_era(aq) and set(ts.targets(aq)) == {"152", "103"}
    t = ts.transfer(["ipc:124a"], ts.targets(aq))
    assert t.ipc_ref == "ipc:124a" and t.score < 0.5 and "BNS 152" in t.reason()
    assert ts.transfer(["bns:152"], ["152"]).score == 1.0
    metas = {d: DocMeta("precedent", "high_court", "", [], date(2015, 1, 1), "ipc", c)
             for d, c in (("old", ["ipc:124a"]), ("new", ["bns:152"]))}
    comps: dict = {}
    final, hit = ts.rerank({"old": 1.0, "new": 0.9}, aq, metas, comps)
    assert hit == ["old"] and final["new"] > final["old"] and comps["old"]["transfer"] < 0.5
    ipc_era = AnalyzedQuery(Query("speech", incident_date=date(2020, 1, 1)), code_in_force=Code.IPC, offence_ids=["off:sedition"])
    assert ts.rerank({"old": 1.0, "new": 0.9}, ipc_era, metas, {})[1] == []


def test_gold_note_labels():
    from kanoon_bridge.drift.gold import note_labels, s2_changed

    assert note_labels("Corresponds to s.95 IPC. Wording retained.") == {"unchanged"}
    assert "punishment" in note_labels("Re-enacts s.282 IPC; the fine limit is raised to ₹5,000.")
    assert "substance" in note_labels("Re-enacts s.295A IPC and expressly adds 'through electronic means'. Punishment is unchanged.")
    assert note_labels("New provision in the BNS: false promise of marriage.") == {"new"}
    assert not s2_changed(note_labels("Merges ss.461-462 IPC. Punishments are unchanged."))


def test_eval_metrics():
    from kanoon_bridge.eval.drift_eval import auc, binary_metrics, cohen_kappa, mcnemar

    m = binary_metrics([1, 1, 0, 0], [1, 0, 0, 0])
    assert m["precision"] == 1 and m["recall"] == 0.5
    assert auc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert mcnemar([1, 1, 1], [1, 1, 1], [1, 1, 1]) == 1.0


def test_classifier_learns():
    import numpy as np

    from kanoon_bridge.drift.classify import cross_val_predict, fit

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.2 * rng.normal(size=200) > 0).astype(int)
    assert fit(X, y).coefficients(["a", "b", "c"])[0][0] == "a"
    assert ((cross_val_predict(X, y) >= 0.5) == y).mean() > 0.85


def test_committed_table_loads():
    from kanoon_bridge.config import project_path

    if not project_path("data/crosswalk/drift_sections.csv").exists():
        pytest.skip("drift table not built")
    t = DriftTable.load()
    assert t.sections["152"].drift_class == DriftClass.SUBSTANTIVE
    assert t.carry_over("ipc:302", "bns:103") > t.carry_over("ipc:124a", "bns:152")
