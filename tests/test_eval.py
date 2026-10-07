"""Evaluation plumbing: test-set loading (IL-PCSR sample), set-specific metrics, tables, plots."""

from __future__ import annotations

import pytest

from kanoon_bridge.config import load_config, project_path
from kanoon_bridge.schema import Query, ScoredDoc

SAMPLE = "tests/data/ilpcsr_sample"


@pytest.fixture
def ev():
    return load_config("eval.yaml")


def test_e1_loads_sample_with_val_and_cap(monkeypatch, ev):
    if not project_path(SAMPLE).exists():
        pytest.skip("run tests/data/make_ilpcsr_sample.py")
    monkeypatch.setenv("KB_ILPCSR_DIR", SAMPLE)
    from kanoon_bridge.eval.run_eval import load_test_set

    ts = load_test_set("e1_ilpcsr", ev=ev)
    assert ts.queries and ts.target == "precedent" and ts.max_query_terms == 100
    assert ts.val is not None and ts.val.queries
    assert set(ts.qrels) <= {q.query_id for q in ts.queries}
    assert all(q.incident_date is not None for q in ts.queries)
    one = load_test_set("e1_ilpcsr", ev=ev, limit=1)
    assert len(one.queries) == 1 and set(one.qrels) <= {one.queries[0].query_id}


def test_hand_built_sets_load(ev):
    from kanoon_bridge.eval.run_eval import load_test_set

    ts = load_test_set("e2_collision", ev=ev)
    assert ts.queries and ts.rows[ts.queries[0].query_id].get("wrong_refs")


class _Bridge:
    statute_terms = {"S1": ["sec:ipc:302", "off:murder"], "bns:302": ["sec:bns:302"], "bns:103": ["sec:bns:103"]}


class _Engine:
    bridge = _Bridge()


def test_extra_metrics_e2_and_e7():
    from kanoon_bridge.eval.run_eval import TestSet, extra_metrics

    q1 = Query("punishment under section 302", query_id="a", incident_date="2023-05-10")
    q2 = Query("threat", query_id="b", incident_date="2025-02-01")
    e2 = TestSet("e2_collision", [q1, q2], {}, "statute",
                 rows={"a": {"wrong_refs": ["bns:302"]}, "b": {"wrong_refs": ["ipc:302"]}})
    run = {"a": [("S1", 2.0), ("bns:302", 1.0)], "b": [("bns:103", 1.0)]}
    assert extra_metrics(e2, run, _Engine())["wrong_hit@10"] == 0.5
    e7 = TestSet("e7_temporal", [q1, q2], {}, "statute")
    assert extra_metrics(e7, run, _Engine())["code_accuracy@1"] == 1.0


def test_write_table_and_plots(tmp_path):
    from kanoon_bridge.eval import plots
    from kanoon_bridge.eval.ablation import write_table

    write_table([{"step": "bm25", "MAP": 0.2, "F1@k_val": 0.1, "MRR": 0.3},
                 {"step": "+zones", "MAP": 0.25, "F1@k_val": 0.12, "MRR": 0.33}], tmp_path / "ablation_x.csv")
    write_table([{"mode": "all", "latency_ms_median": 5, "latency_ms_p95": 9, "recall@20_vs_exhaustive": 1.0},
                 {"mode": "tiers", "latency_ms_median": 2, "latency_ms_p95": 4, "recall@20_vs_exhaustive": 0.9}],
                tmp_path / "efficiency.csv")
    made = plots.make_all(tmp_path, tmp_path / "fig")
    assert {p.name for p in made} == {"ablation_x.png", "efficiency.png"}
    assert all(p.stat().st_size > 1000 for p in made)
