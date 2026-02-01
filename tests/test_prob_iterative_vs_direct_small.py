from cib.prob import DiagnosticsReport
from cib.prob.model import ProbabilisticCIAModel
from cib.prob.types import FactorSpec


def test_iterative_method_runs_and_is_reasonable_on_two_by_two() -> None:
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.6, "a1": 0.4}, "B": {"b0": 0.7, "b1": 0.3}}

    p_a1_b1 = 0.6
    p_a1_b0 = (0.4 - 0.3 * p_a1_b1) / 0.7

    multipliers = {
        (("A", "a1"), ("B", "b1")): p_a1_b1 / marginals["A"]["a1"],
        (("A", "a0"), ("B", "b1")): (1.0 - p_a1_b1) / marginals["A"]["a0"],
        (("A", "a1"), ("B", "b0")): p_a1_b0 / marginals["A"]["a1"],
        (("A", "a0"), ("B", "b0")): (1.0 - p_a1_b0) / marginals["A"]["a0"],
    }

    model = ProbabilisticCIAModel(factors=factors, marginals=marginals, multipliers=multipliers)
    dense = model.fit_joint(method="direct", kl_weight=1e-8, solver_maxiter=3000)
    approx = model.fit_joint(
        method="iterative",
        random_seed=123,
        iterative_burn_in_sweeps=1500,
        iterative_n_samples=20000,
        iterative_thinning=2,
        with_report=True,
    )

    # The method is expected to produce a usable distribution object and diagnostics.
    rep_dense = DiagnosticsReport.from_distribution(dense, marginals=marginals, multipliers=multipliers)
    rep_approx = DiagnosticsReport.from_distribution(approx, marginals=marginals, multipliers=multipliers)
    assert rep_dense.sum_to_one_error < 1e-9
    assert rep_approx.sum_to_one_error < 1e-8

    # Conditional agreement is expected to be approximate.
    c_dense = dense.conditional(("A", "a1"), ("B", "b1"))
    c_approx = approx.conditional(("A", "a1"), ("B", "b1"))
    assert abs(float(c_dense) - float(c_approx)) < 0.25

