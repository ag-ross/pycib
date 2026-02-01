"""
Smoke tests for benchmark fixtures.

This test file is intended to ensure that benchmark datasets can be constructed
and exercised quickly in CI without asserting specific performance numbers.
"""

from __future__ import annotations

import numpy as np

from cib.benchmark_data import benchmark_matrix_b1, benchmark_matrix_b2
from cib.core import ConsistencyChecker, Scenario


def _random_scenario(matrix, seed: int) -> Scenario:
    rng = np.random.default_rng(int(seed))
    sdict = {}
    for d, states in matrix.descriptors.items():
        sdict[d] = states[int(rng.integers(0, len(states)))]
    return Scenario(sdict, matrix)


def test_b1_can_be_scored_and_checked() -> None:
    m = benchmark_matrix_b1()
    s = _random_scenario(m, seed=101)
    ok = ConsistencyChecker.check_consistency(s, m)
    assert isinstance(ok, bool)


def test_b2_can_be_scored_and_checked() -> None:
    m = benchmark_matrix_b2()
    s = _random_scenario(m, seed=202)
    ok = ConsistencyChecker.check_consistency(s, m)
    assert isinstance(ok, bool)

