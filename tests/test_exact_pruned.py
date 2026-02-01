"""
Unit tests for the exact pruned solver.
"""

from __future__ import annotations

from cib.analysis import ScenarioAnalyzer
from cib.benchmark_data import benchmark_matrix_b1
from cib.solvers.config import ExactSolverConfig


def test_exact_pruned_matches_bruteforce_on_b1() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    brute = analyzer.find_all_consistent(max_scenarios=50_000)
    brute_set = {tuple(s.to_indices()) for s in brute}

    cfg = ExactSolverConfig(ordering="given", bound="safe_upper_bound_v1")
    res = analyzer.find_all_consistent_exact(config=cfg)
    got_set = {tuple(s.to_indices()) for s in res.scenarios}

    assert res.is_complete is True
    assert got_set == brute_set


def test_exact_pruned_without_bounds_matches_bruteforce_on_b1() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    brute = analyzer.find_all_consistent(max_scenarios=50_000)
    brute_set = {tuple(s.to_indices()) for s in brute}

    cfg = ExactSolverConfig(ordering="given", bound="none")
    res = analyzer.find_all_consistent_exact(config=cfg)
    got_set = {tuple(s.to_indices()) for s in res.scenarios}

    assert res.is_complete is True
    assert got_set == brute_set


def test_exact_pruned_time_limit_can_stop_early() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg = ExactSolverConfig(ordering="random", bound="safe_upper_bound_v1", time_limit_s=1e-9)
    res = analyzer.find_all_consistent_exact(config=cfg)

    assert res.is_complete is False
    assert res.status in {"timeout", "max_solutions"}

