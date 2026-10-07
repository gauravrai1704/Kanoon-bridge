"""Cohen's kappa used for the E4/E6 second judge."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("second_judge", Path(__file__).parents[1] / "scripts/12_second_judge.py")
sj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sj)


def test_kappa_perfect_and_chance():
    assert sj.kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    # observed 0.5, expected 0.5 -> 0
    assert abs(sj.kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9


def test_kappa_textbook():
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no (Wikipedia example): kappa = 0.4
    a = [1] * 25 + [0] * 25
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert abs(sj.kappa(a, b) - 0.4) < 1e-9
