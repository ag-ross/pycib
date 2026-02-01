"""
Unit tests for fast succession parity with the reference implementation.
"""

from __future__ import annotations

import numpy as np

from cib.benchmark_data import benchmark_matrix_b1
from cib.core import Scenario
from cib.fast_scoring import FastCIBScorer
from cib.sparse_scoring import SparseCIBScorer
from cib.fast_succession import run_to_attractor_indices
from cib.succession import GlobalSuccession, LocalSuccession


def _random_scenario(matrix, seed: int) -> Scenario:
    rng = np.random.default_rng(int(seed))
    sdict = {}
    for d, states in matrix.descriptors.items():
        sdict[d] = states[int(rng.integers(0, len(states)))]
    return Scenario(sdict, matrix)


def test_fast_global_attractor_matches_reference_for_fixed_point() -> None:
    m = benchmark_matrix_b1()
    scorer = FastCIBScorer.from_matrix(m)

    s0 = _random_scenario(m, seed=901)
    z0 = scorer.scenario_to_indices(s0)

    ref = GlobalSuccession().find_attractor(s0, m, max_iterations=200)
    fast = run_to_attractor_indices(
        scorer=scorer, initial_z_idx=z0, rule="global", max_iterations=200
    )

    if ref.is_cycle:
        assert fast.is_cycle is True
    else:
        assert fast.is_cycle is False
        assert tuple(int(x) for x in ref.attractor.to_indices()) == fast.attractor  # type: ignore[union-attr]


def test_sparse_global_attractor_matches_reference_for_fixed_point() -> None:
    m = benchmark_matrix_b1()
    scorer = SparseCIBScorer.from_matrix(m)

    s0 = _random_scenario(m, seed=903)
    z0 = scorer.scenario_to_indices(s0)

    ref = GlobalSuccession().find_attractor(s0, m, max_iterations=200)
    fast = run_to_attractor_indices(
        scorer=scorer, initial_z_idx=z0, rule="global", max_iterations=200
    )

    if ref.is_cycle:
        assert fast.is_cycle is True
    else:
        assert fast.is_cycle is False
        assert tuple(int(x) for x in ref.attractor.to_indices()) == fast.attractor  # type: ignore[union-attr]


def test_fast_local_attractor_matches_reference_for_fixed_point() -> None:
    m = benchmark_matrix_b1()
    scorer = FastCIBScorer.from_matrix(m)

    s0 = _random_scenario(m, seed=902)
    z0 = scorer.scenario_to_indices(s0)

    ref = LocalSuccession().find_attractor(s0, m, max_iterations=200)
    fast = run_to_attractor_indices(
        scorer=scorer, initial_z_idx=z0, rule="local", max_iterations=200
    )

    if ref.is_cycle:
        assert fast.is_cycle is True
    else:
        assert fast.is_cycle is False
        assert tuple(int(x) for x in ref.attractor.to_indices()) == fast.attractor  # type: ignore[union-attr]


def test_sparse_local_attractor_matches_reference_for_fixed_point() -> None:
    m = benchmark_matrix_b1()
    scorer = SparseCIBScorer.from_matrix(m)

    s0 = _random_scenario(m, seed=904)
    z0 = scorer.scenario_to_indices(s0)

    ref = LocalSuccession().find_attractor(s0, m, max_iterations=200)
    fast = run_to_attractor_indices(
        scorer=scorer, initial_z_idx=z0, rule="local", max_iterations=200
    )

    if ref.is_cycle:
        assert fast.is_cycle is True
    else:
        assert fast.is_cycle is False
        assert tuple(int(x) for x in ref.attractor.to_indices()) == fast.attractor  # type: ignore[union-attr]

