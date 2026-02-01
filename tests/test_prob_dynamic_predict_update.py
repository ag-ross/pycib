import numpy as np

from cib.prob.dynamic import DynamicProbabilisticCIA
from cib.prob.model import ProbabilisticCIAModel
from cib.prob.types import FactorSpec


def _multipliers_for_a_given_b(marginals, *, p_a1_b0: float, p_a1_b1: float):
    """
    Return a complete multiplier set for A given B for both outcomes of A and B.
    """
    p_a0_b0 = 1.0 - float(p_a1_b0)
    p_a0_b1 = 1.0 - float(p_a1_b1)
    return {
        (("A", "a1"), ("B", "b0")): float(p_a1_b0) / float(marginals["A"]["a1"]),
        (("A", "a0"), ("B", "b0")): float(p_a0_b0) / float(marginals["A"]["a0"]),
        (("A", "a1"), ("B", "b1")): float(p_a1_b1) / float(marginals["A"]["a1"]),
        (("A", "a0"), ("B", "b1")): float(p_a0_b1) / float(marginals["A"]["a0"]),
    }


def test_predict_update_is_more_conservative_than_refit_when_kl_is_large() -> None:
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.6, "a1": 0.4}, "B": {"b0": 0.7, "b1": 0.3}}

    # Period 1 encourages A=a1 when B=b1.
    multipliers_t1 = _multipliers_for_a_given_b(marginals, p_a1_b0=0.3142857142857143, p_a1_b1=0.9)
    # Period 2 encourages the opposite.
    multipliers_t2 = _multipliers_for_a_given_b(marginals, p_a1_b0=0.3142857142857143, p_a1_b1=0.1)

    model1 = ProbabilisticCIAModel(factors=factors, marginals=marginals, multipliers=multipliers_t1)
    model2 = ProbabilisticCIAModel(factors=factors, marginals=marginals, multipliers=multipliers_t2)

    dyn = DynamicProbabilisticCIA(periods=[2025, 2030], models_by_period={2025: model1, 2030: model2})

    refit = dyn.fit_distributions(
        mode="refit",
        method="direct",
        kl_weight=0.0,
        weight_by_target=False,
        solver_maxiter=4000,
    )
    pu = dyn.fit_distributions(
        mode="predict-update",
        method="direct",
        kl_weight=50.0,
        weight_by_target=False,
        solver_maxiter=4000,
    )

    p1 = refit[2025].p
    p2_refit = refit[2030].p
    p2_pu = pu[2030].p

    # The predict–update fit is expected to remain closer to the previous period distribution.
    d_refit = float(np.sum(np.abs(p2_refit - p1)))
    d_pu = float(np.sum(np.abs(p2_pu - p1)))
    assert d_pu < d_refit

