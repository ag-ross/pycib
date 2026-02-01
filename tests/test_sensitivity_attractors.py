from __future__ import annotations

from cib.analysis import ScenarioAnalyzer
from cib.benchmark_data import benchmark_matrix_b1
from cib.sensitivity import compute_global_sensitivity_attractors
from cib.solvers.config import MonteCarloAttractorConfig


def test_global_sensitivity_attractors_summary_runs() -> None:
    m = benchmark_matrix_b1()
    analyzer = ScenarioAnalyzer(m)
    cfg = MonteCarloAttractorConfig(runs=200, seed=123, succession="global")
    res = analyzer.find_attractors_monte_carlo(config=cfg)

    rep = compute_global_sensitivity_attractors(res, top_k=5)
    assert rep.n_runs > 0
    assert len(rep.outcome_names) >= 1

