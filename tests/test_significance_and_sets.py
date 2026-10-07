"""Significance tests and the generated test sets."""

import importlib.util
import random
from pathlib import Path

import numpy as np

from kanoon_bridge.eval.significance import bootstrap_ci, compare, holm, per_query, randomization_test


def test_randomization_detects_a_real_difference():
    rng = np.random.default_rng(1)
    better = rng.uniform(0.05, 0.15, size=60)                  # B beats A on every query
    assert randomization_test(better, 2000) < 0.01
    noise = rng.normal(0, 0.1, size=60)
    assert randomization_test(noise - noise.mean(), 2000) > 0.5
    assert randomization_test(np.zeros(10)) == 1.0


def test_bootstrap_ci_contains_the_mean():
    d = np.array([0.1, 0.2, 0.0, 0.3, 0.1])
    lo, hi = bootstrap_ci(d, 2000)
    assert lo <= d.mean() <= hi


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert adj == [0.04, 0.09, 0.09, 0.5]


def test_compare_and_per_query():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {}}
    sa = per_query({"q1": ["x", "a"], "q2": ["b"]}, qrels)
    sb = per_query({"q1": ["a"], "q2": ["b"]}, qrels)
    assert set(sa["AP"]) == {"q1", "q2"}                       # q3 has no relevant docs: skipped
    r = compare(sa["AP"], sb["AP"], 500)
    assert r["diff"] == 0.25 and r["wins"] == 1 and r["losses"] == 0 and r["queries"] == 2


def _gen():
    spec = importlib.util.spec_from_file_location("gen", Path(__file__).parents[1] / "scripts" / "07_make_test_sets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generated_e2_and_e7_from_the_real_crosswalk():
    from kanoon_bridge.text.version_norm import VersionNormalizer

    gen = _gen()
    norm = VersionNormalizer.load()
    by_ref = {f"ipc:{n}": [f"I{n}"] for n in ("302", "506", "503", "507")} | {f"bns:{n}": [f"bns:{n}"] for n in ("103", "302", "351")}
    rows, qrels = gen.make_e2(norm, by_ref, None, random.Random(0))
    ids = {r["id"] for r in rows}
    assert {"E2-302-ipc-bare", "E2-302-bns-bare", "E2-302-ipc-context"} <= ids
    bare_ipc = next(r for r in rows if r["id"] == "E2-302-ipc-bare")
    assert bare_ipc["incident_date"] < "2024-07-01" and bare_ipc["wrong_refs"] == ["bns:302"]
    assert ("E2-302-bns-bare", "bns:302", 2) in qrels
    rows, qrels = gen.make_e7(norm, by_ref, None, random.Random(0))
    murder = [r for r in rows if r["offence"] == "off:murder_bns103"]
    assert {r["expected_code"] for r in murder} == {"ipc", "bns"}
    assert ("E7-murder_bns103-ipc", "I302", 2) in qrels and ("E7-murder_bns103-bns", "bns:103", 2) in qrels
