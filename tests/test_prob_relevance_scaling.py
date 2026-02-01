import numpy as np

from cib.prob import RelevanceSpec
from cib.prob.model import ProbabilisticCIAModel
from cib.prob.types import FactorSpec


def _multipliers_for_a_given_x(marginals, *, given_factor: str, p_a1_given0: float, p_a1_given1: float):
    p_a0_g0 = 1.0 - float(p_a1_given0)
    p_a0_g1 = 1.0 - float(p_a1_given1)
    return {
        (("A", "a1"), (given_factor, "x0")): float(p_a1_given0) / float(marginals["A"]["a1"]),
        (("A", "a0"), (given_factor, "x0")): float(p_a0_g0) / float(marginals["A"]["a0"]),
        (("A", "a1"), (given_factor, "x1")): float(p_a1_given1) / float(marginals["A"]["a1"]),
        (("A", "a0"), (given_factor, "x1")): float(p_a0_g1) / float(marginals["A"]["a0"]),
    }


def test_relevance_scaling_can_disable_multiplier_constraints() -> None:
    factors = [
        FactorSpec("A", ["a0", "a1"]),
        FactorSpec("B", ["x0", "x1"]),
        FactorSpec("C", ["x0", "x1"]),
    ]
    marginals = {
        "A": {"a0": 0.6, "a1": 0.4},
        "B": {"x0": 0.5, "x1": 0.5},
        "C": {"x0": 0.5, "x1": 0.5},
    }

    multipliers_ab = _multipliers_for_a_given_x(marginals, given_factor="B", p_a1_given0=0.2, p_a1_given1=0.8)
    # An extreme multiplier set that would otherwise dominate the fit.
    # Values are chosen to remain coherent with P(A=a1)=0.4 under P(C)=uniform:
    # 0.5 * p + 0.5 * q = 0.4 -> p + q = 0.8.
    multipliers_ac = _multipliers_for_a_given_x(marginals, given_factor="C", p_a1_given0=0.75, p_a1_given1=0.05)

    model_full = ProbabilisticCIAModel(factors=factors, marginals=marginals, multipliers={**multipliers_ab, **multipliers_ac})
    model_ab_only = ProbabilisticCIAModel(factors=factors, marginals=marginals, multipliers=multipliers_ab)

    rel = RelevanceSpec(parents={"A": {"B"}})
    dist_rel = model_full.fit_joint(method="direct", relevance=rel, relevance_default_weight=0.0, weight_by_target=False, solver_maxiter=4000)
    dist_ab = model_ab_only.fit_joint(method="direct", weight_by_target=False, solver_maxiter=4000)

    # With C excluded by relevance, the fit should be close to the A<-B-only fit.
    l1 = float(np.sum(np.abs(dist_rel.p - dist_ab.p)))
    assert l1 < 5e-3

