"""Jurisdiction-aware authority.  [D]"""

from datetime import date

from conftest import todo

from kanoon_bridge.index.facets import DocMeta
from kanoon_bridge.rank.authority import binding_status
from kanoon_bridge.rank.topk import net_score, top_k


def meta(court, states):
    return DocMeta("precedent", court, "", states, date(2015, 1, 1), "ipc", [])


def test_supreme_court_binds_everywhere():
    assert binding_status(meta("supreme_court", ["*"]), "kerala") == "binding"


def test_own_high_court_binds():
    assert binding_status(meta("high_court", ["delhi"]), "delhi") == "binding"


def test_other_high_court_is_persuasive():
    assert binding_status(meta("high_court", ["delhi"]), "maharashtra") == "persuasive"


def test_no_state_is_unknown():
    assert binding_status(meta("high_court", ["delhi"]), None) == "unknown"


def test_top_k_and_net_score():
    assert top_k({"a": 1.0, "b": 3.0, "c": 2.0}, 2) == [("b", 3.0), ("c", 2.0)]
    assert net_score(1.0, 0.5, 0.2) == 1.1


@todo
def test_state_changes_ranking():
    from kanoon_bridge.config import load_config
    from kanoon_bridge.rank.authority import Authority

    metas = {"delhi_hc": meta("high_court", ["delhi"]), "bom_hc": meta("high_court", ["maharashtra"])}
    a = Authority(cfg=load_config(), global_scores={"delhi_hc": 0.5, "bom_hc": 0.5}, metas=metas)
    assert a.score("delhi_hc", "delhi") > a.score("bom_hc", "delhi")
    assert a.score("bom_hc", "maharashtra") > a.score("delhi_hc", "maharashtra")
