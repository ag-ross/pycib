"""
Unit tests for Monte Carlo attractor discovery.
"""

from __future__ import annotations

from cib.analysis import ScenarioAnalyzer
from cib.benchmark_data import benchmark_matrix_b1
from cib.core import ConsistencyChecker, Scenario
from cib.solvers.config import MonteCarloAttractorConfig


def test_monte_carlo_attractor_reproducibility() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg = MonteCarloAttractorConfig(runs=200, seed=123, succession="global")
    r1 = analyzer.find_attractors_monte_carlo(config=cfg)
    r2 = analyzer.find_attractors_monte_carlo(config=cfg)

    assert r1.counts == r2.counts
    assert r1.attractor_keys_ranked == r2.attractor_keys_ranked

    cfg_sparse = MonteCarloAttractorConfig(
        runs=200, seed=123, succession="global", fast_backend="sparse"
    )
    s1 = analyzer.find_attractors_monte_carlo(config=cfg_sparse)
    s2 = analyzer.find_attractors_monte_carlo(config=cfg_sparse)
    assert s1.counts == s2.counts
    assert s1.attractor_keys_ranked == s2.attractor_keys_ranked


def test_monte_carlo_attractor_multiprocess_matches_single_process() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg1 = MonteCarloAttractorConfig(runs=200, seed=999, succession="global", n_jobs=1)
    cfg2 = MonteCarloAttractorConfig(runs=200, seed=999, succession="global", n_jobs=2)

    r1 = analyzer.find_attractors_monte_carlo(config=cfg1)
    r2 = analyzer.find_attractors_monte_carlo(config=cfg2)

    assert r1.counts == r2.counts
    assert r1.attractor_keys_ranked == r2.attractor_keys_ranked

    cfg1s = MonteCarloAttractorConfig(
        runs=200, seed=999, succession="global", n_jobs=1, fast_backend="sparse"
    )
    cfg2s = MonteCarloAttractorConfig(
        runs=200, seed=999, succession="global", n_jobs=2, fast_backend="sparse"
    )
    s1 = analyzer.find_attractors_monte_carlo(config=cfg1s)
    s2 = analyzer.find_attractors_monte_carlo(config=cfg2s)
    assert s1.counts == s2.counts
    assert s1.attractor_keys_ranked == s2.attractor_keys_ranked


def test_monte_carlo_attractor_keys_are_valid_fixed_points() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg = MonteCarloAttractorConfig(runs=200, seed=456, succession="global")
    res = analyzer.find_attractors_monte_carlo(config=cfg)

    # At least one attractor should be discovered for this benchmark.
    assert len(res.counts) >= 1

    # Fixed-point keys should correspond to consistent scenarios.
    for key in res.attractor_keys_ranked[:5]:
        if key.kind != "fixed":
            continue
        v = key.value
        assert isinstance(v, tuple)
        s = Scenario(list(v), m)
        assert ConsistencyChecker.check_consistency(s, m) is True


def test_monte_carlo_cycle_key_policy_rotate_min_has_cycle_structure() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg = MonteCarloAttractorConfig(
        runs=200,
        seed=321,
        succession="global",
        cycle_mode="keep_cycle",
        cycle_key_policy="rotate_min",
    )
    res = analyzer.find_attractors_monte_carlo(config=cfg)
    assert len(res.counts) >= 1

    for key in res.attractor_keys_ranked:
        if key.kind != "cycle":
            continue
        v = key.value
        assert isinstance(v, tuple)
        assert len(v) >= 1
        assert isinstance(v[0], tuple)
        break


def test_monte_carlo_cycle_mode_representative_first_is_distinct_from_keep_cycle() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg_keep = MonteCarloAttractorConfig(
        runs=200, seed=222, succession="global", cycle_mode="keep_cycle"
    )
    cfg_rep = MonteCarloAttractorConfig(
        runs=200, seed=222, succession="global", cycle_mode="representative_first"
    )
    r_keep = analyzer.find_attractors_monte_carlo(config=cfg_keep)
    r_rep = analyzer.find_attractors_monte_carlo(config=cfg_rep)

    assert r_keep.diagnostics["n_completed_runs"] == r_rep.diagnostics["n_completed_runs"]
    assert r_keep.cycles is not None
    assert r_rep.cycles is None


def test_monte_carlo_cycle_mode_representative_random_is_reproducible_across_jobs() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)

    cfg1 = MonteCarloAttractorConfig(
        runs=300,
        seed=777,
        succession="global",
        cycle_mode="representative_random",
        n_jobs=1,
    )
    cfg2 = MonteCarloAttractorConfig(
        runs=300,
        seed=777,
        succession="global",
        cycle_mode="representative_random",
        n_jobs=2,
    )
    r1 = analyzer.find_attractors_monte_carlo(config=cfg1)
    r2 = analyzer.find_attractors_monte_carlo(config=cfg2)
    assert r1.counts == r2.counts
