from cib.prob import event_probability_bounds
from cib.prob.types import FactorSpec


def test_risk_bounds_match_frechet_bounds_for_pair_event_under_marginals_only() -> None:
    factors = [FactorSpec("A", ["a0", "a1"]), FactorSpec("B", ["b0", "b1"])]
    marginals = {"A": {"a0": 0.6, "a1": 0.4}, "B": {"b0": 0.7, "b1": 0.3}}

    res = event_probability_bounds(
        factors=factors,
        marginals=marginals,
        multipliers={},
        event={"A": "a1", "B": "b1"},
        include_pairwise_targets=False,
    )
    assert res.status == "ok"

    pi = marginals["A"]["a1"]
    pj = marginals["B"]["b1"]
    lo = max(0.0, pi + pj - 1.0)
    hi = min(pi, pj)

    assert abs(res.lower - lo) < 1e-8
    assert abs(res.upper - hi) < 1e-8

