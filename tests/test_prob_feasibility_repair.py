import pytest

from cib.prob.model import ProbabilisticCIAModel
from cib.prob.types import FactorSpec


def test_strict_feasibility_rejects_frechet_violation() -> None:
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.5, "a1": 0.5}, "B": {"b0": 0.5, "b1": 0.5}}

    # Target would be P(A=a1,B=b1) = 5 * 0.5 * 0.5 = 1.25 (infeasible).
    multipliers = {(("A", "a1"), ("B", "b1")): 5.0}

    with pytest.raises(ValueError):
        ProbabilisticCIAModel(
            factors=factors,
            marginals=marginals,
            multipliers=multipliers,
            feasibility_mode="strict",
        )


def test_repair_feasibility_clips_targets_and_reports_adjustments() -> None:
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.5, "a1": 0.5}, "B": {"b0": 0.5, "b1": 0.5}}

    multipliers = {(("A", "a1"), ("B", "b1")): 5.0}
    model = ProbabilisticCIAModel(
        factors=factors,
        marginals=marginals,
        multipliers=multipliers,
        feasibility_mode="repair",
    )
    dist = model.fit_joint(method="direct", kl_weight=1e-8, solver_maxiter=3000, with_report=True)

    assert dist.fit_report is not None
    assert dist.fit_report.feasibility_mode == "repair"
    assert len(dist.fit_report.feasibility_adjustments) >= 1
    adj = dist.fit_report.feasibility_adjustments[0]
    assert adj.original_value > adj.frechet_upper
    assert adj.adjusted_value == adj.frechet_upper

