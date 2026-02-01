from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

from cib.prob.constraints import (
    Marginals,
    Multipliers,
    multiplier_pairwise_targets,
    pairwise_target_frechet_violations,
    project_pairwise_targets_to_frechet_bounds,
    validate_marginals,
)
from cib.prob.types import AssignmentLike, FactorSpec, ScenarioIndex


@dataclass(frozen=True)
class RiskBoundsResult:
    """
    Linear-programming bounds on an event probability under a constraint set.

    The returned bounds are identification bounds under the configured constraints
    and are not to be interpreted as sampling uncertainty intervals.
    """

    lower: float
    upper: float
    status: str
    message: str


def _build_marginal_constraints(index: ScenarioIndex, marginals: Marginals) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build equality constraints matrix C p = d for:
      - sum_x p_x = 1
      - per-factor marginals (k-1 outcomes per factor, last implied)
    """
    rows = []
    rhs = []

    rows.append(np.ones(index.size, dtype=float))
    rhs.append(1.0)

    for pos, factor in enumerate(index.factors):
        outcomes = list(factor.outcomes)
        for outcome in outcomes[:-1]:
            row = np.zeros(index.size, dtype=float)
            for idx in range(index.size):
                scen = index.scenario_at(idx)
                if scen.assignment[pos] == outcome:
                    row[idx] = 1.0
            rows.append(row)
            rhs.append(float(marginals[factor.name][outcome]))

    C = np.vstack(rows)
    d = np.array(rhs, dtype=float)
    return C, d


def _event_indicator(index: ScenarioIndex, event: Mapping[str, str]) -> np.ndarray:
    """
    Return an indicator vector for a partial assignment event.

    The event is specified as a mapping {factor_name: outcome}.
    """
    event = {str(k): str(v) for k, v in event.items()}
    for k in event:
        if k not in index.factor_names:
            raise ValueError(f"Unknown factor in event specification: {k!r}")
    pos_by_name = {n: i for i, n in enumerate(index.factor_names)}

    ind = np.zeros(index.size, dtype=float)
    for idx in range(index.size):
        scen = index.scenario_at(idx)
        ok = True
        for fname, outcome in event.items():
            if scen.assignment[pos_by_name[fname]] != outcome:
                ok = False
                break
        if ok:
            ind[idx] = 1.0
    return ind


def event_probability_bounds(
    *,
    factors: Sequence[FactorSpec],
    marginals: Marginals,
    multipliers: Optional[Multipliers] = None,
    event: Mapping[str, str],
    include_pairwise_targets: bool = True,
    feasibility_mode: str = "strict",
    feasibility_tol: float = 1e-9,
    max_scenarios: int = 50_000,
) -> RiskBoundsResult:
    """
    Compute bounds on P(event) under a constraint set using linear programming.

    The constraint set is defined by:
    - p >= 0
    - sum(p) = 1
    - exact marginals
    - optional exact pairwise target constraints implied by multipliers

    This routine is intended for small scenario spaces where dense enumeration is feasible.
    """
    multipliers = multipliers or {}
    feasibility_mode = str(feasibility_mode).strip().lower()
    if feasibility_mode not in {"strict", "repair"}:
        raise ValueError(f"Unknown feasibility_mode: {feasibility_mode!r}")

    index = ScenarioIndex(tuple(factors))
    if int(index.size) > int(max_scenarios):
        raise ValueError(
            f"Scenario space too large for risk bounds (size={index.size}, max_scenarios={int(max_scenarios)})."
        )

    validate_marginals(index.factors, marginals)
    C, d = _build_marginal_constraints(index, marginals)

    A_eq = C
    b_eq = d

    if include_pairwise_targets and multipliers:
        targets_raw = multiplier_pairwise_targets(marginals, multipliers)
        if feasibility_mode == "strict":
            violations = pairwise_target_frechet_violations(marginals, targets_raw, tol=float(feasibility_tol))
            if violations:
                k = next(iter(violations.keys()))
                v, lo, hi = violations[k]
                raise ValueError(
                    "Multiplier-implied pairwise target violates Fréchet bounds: "
                    f"{k!r} has {v}, but bounds are [{lo}, {hi}]"
                )
            targets = dict(targets_raw)
        else:
            targets, _adjustments = project_pairwise_targets_to_frechet_bounds(
                marginals, targets_raw, tol=float(feasibility_tol)
            )

        rows = []
        rhs = []
        fname_to_pos = {n: i for i, n in enumerate(index.factor_names)}
        for (i, a, j, b), t in targets.items():
            pi = fname_to_pos[str(i)]
            pj = fname_to_pos[str(j)]
            row = np.zeros(index.size, dtype=float)
            for idx in range(index.size):
                scen = index.scenario_at(idx)
                if scen.assignment[pi] == str(a) and scen.assignment[pj] == str(b):
                    row[idx] = 1.0
            rows.append(row)
            rhs.append(float(t))

        if rows:
            A_pair = np.vstack(rows)
            b_pair = np.array(rhs, dtype=float)
            A_eq = np.vstack([A_eq, A_pair])
            b_eq = np.concatenate([b_eq, b_pair])

    ind = _event_indicator(index, event)
    bounds = [(0.0, 1.0) for _ in range(index.size)]

    # Lower bound: minimise indicator mass.
    res_lo = linprog(c=ind, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res_lo.success:
        # Covers infeasible, unbounded, or iteration-limit; res_lo.message gives detail.
        return RiskBoundsResult(
            lower=float("nan"),
            upper=float("nan"),
            status="failed",
            message=f"Lower-bound optimisation failed: {res_lo.message}",
        )
    lo = float(res_lo.fun)

    # Upper bound: maximise indicator mass by minimising negative.
    res_hi = linprog(c=-ind, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res_hi.success:
        # Covers infeasible, unbounded, or iteration-limit; res_hi.message gives detail.
        return RiskBoundsResult(
            lower=float("nan"),
            upper=float("nan"),
            status="failed",
            message=f"Upper-bound optimisation failed: {res_hi.message}",
        )
    hi = float(-res_hi.fun)

    return RiskBoundsResult(lower=float(lo), upper=float(hi), status="ok", message="ok")

