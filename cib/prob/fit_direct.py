from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Union, List

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from cib.prob.constraints import (
    Marginals,
    Multipliers,
    multiplier_pairwise_targets,
    pairwise_target_frechet_violations,
    project_pairwise_targets_to_frechet_bounds,
    validate_marginals,
)
from cib.prob.fit_report import FitReport
from cib.prob.types import ScenarioIndex


def _product_baseline(index: ScenarioIndex, marginals: Marginals) -> np.ndarray:
    """
    Baseline q(x) = Π_i P(X_i = x_i) implied by marginals (independence baseline).
    """
    q = np.zeros(index.size, dtype=float)
    for idx in range(index.size):
        scen = index.scenario_at(idx)
        prod = 1.0
        for fname, outcome in zip(index.factor_names, scen.assignment):
            prod *= float(marginals[fname][outcome])
        q[idx] = float(prod)
    s = float(np.sum(q))
    if s <= 0.0:
        raise ValueError("Baseline distribution is degenerate (sum <= 0)")
    q = q / s
    return q


def _build_linear_constraints(index: ScenarioIndex, marginals: Marginals) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build equality constraints matrix C p = d for:
      - sum_x p_x = 1
      - per-factor marginals (k-1 outcomes per factor, last implied)
    """
    rows = []
    rhs = []

    # The sum-to-one constraint is enforced.
    rows.append(np.ones(index.size, dtype=float))
    rhs.append(1.0)

    # Only (k-1) outcomes per factor are used for marginals to avoid dependent equalities.
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


def _build_pairwise_matrix(
    index: ScenarioIndex,
    targets: Mapping[Tuple[str, str, str, str], float],
) -> Tuple[List[Tuple[str, str, str, str]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Build A p ~ t for pairwise targets P(i=a, j=b).

    Returns:
      A: (K, M)
      t: (K,)
      w: (K,) weights (default 1.0)
    """
    items = list(targets.items())
    keys = [(str(i), str(a), str(j), str(b)) for (i, a, j, b), _v in items]
    A = np.zeros((len(keys), index.size), dtype=float)
    t = np.array([float(v) for (_k, v) in items], dtype=float)

    fname_to_pos = {n: i for i, n in enumerate(index.factor_names)}
    for r, (i, a, j, b) in enumerate(keys):
        pi = fname_to_pos[str(i)]
        pj = fname_to_pos[str(j)]
        a = str(a)
        b = str(b)
        for idx in range(index.size):
            scen = index.scenario_at(idx)
            if scen.assignment[pi] == a and scen.assignment[pj] == b:
                A[r, idx] = 1.0
    w = np.ones(len(keys), dtype=float)
    return keys, A, t, w


def fit_joint_direct(
    *,
    index: ScenarioIndex,
    marginals: Marginals,
    multipliers: Optional[Multipliers] = None,
    kl_weight: float = 0.0,
    kl_baseline: Optional[np.ndarray] = None,
    kl_baseline_eps: float = 0.0,
    feasibility_mode: str = "strict",
    feasibility_tol: float = 1e-9,
    relevance_weights: Optional[Mapping[Tuple[str, str], float]] = None,
    weight_by_target: bool = True,
    random_seed: Optional[int] = None,
    solver_maxiter: int = 2_000,
    return_report: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, FitReport]]:
    """
    Fit a dense joint distribution p over all scenarios.

    Objective (default):
      minimize Σ_k w_k (A_k p - t_k)^2  + kl_weight * KL(p || q)

    subject to:
      p >= 0, sum(p)=1, and exact marginal matching.
    """
    multipliers = multipliers or {}
    validate_marginals(index.factors, marginals)

    # Fast path: if there are no multiplier constraints, the independent baseline implied by marginals is already a valid joint distribution.
    if not multipliers:
        return _product_baseline(index, marginals)

    feasibility_mode = str(feasibility_mode).strip().lower()
    if feasibility_mode not in {"strict", "repair"}:
        raise ValueError(f"Unknown feasibility_mode: {feasibility_mode!r}")

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
        feasibility_adjustments = tuple()
    else:
        targets, feasibility_adjustments = project_pairwise_targets_to_frechet_bounds(
            marginals, targets_raw, tol=float(feasibility_tol)
        )
    keys, A, t, w = _build_pairwise_matrix(index, targets)
    if weight_by_target and len(w) > 0:
        # Relative errors are weighted more evenly across different scales.
        w = 1.0 / np.maximum(1e-12, np.abs(t))
        # Extreme weights are capped to avoid numerical overflow when targets are tiny.
        w = np.minimum(w, 1e8)

    if relevance_weights is not None and len(w) > 0:
        # Directed relevance weights are applied at the factor-pair level.
        rw = np.array([float(relevance_weights.get((i, j), 1.0)) for (i, _a, j, _b) in keys], dtype=float)
        w = w * rw

    C, d = _build_linear_constraints(index, marginals)
    lin = LinearConstraint(C, d, d)

    kl_weight = float(kl_weight)
    use_kl = kl_weight > 0.0
    if use_kl:
        if kl_baseline is None:
            q = _product_baseline(index, marginals)
        else:
            q = np.asarray(kl_baseline, dtype=float)
            if q.ndim != 1 or int(q.shape[0]) != int(index.size):
                raise ValueError("KL baseline has wrong shape")
            if float(kl_baseline_eps) > 0.0:
                q = np.maximum(q, float(kl_baseline_eps))
            qs = float(np.sum(q))
            if qs <= 0.0:
                raise ValueError("KL baseline is degenerate (sum <= 0)")
            q = q / qs
        if np.any(q <= 0.0):
            raise ValueError("KL baseline has zeros; cannot use KL regularisation")
        eps = 1e-12
        bounds = Bounds(eps * np.ones(index.size), np.ones(index.size))
    else:
        q = np.ones(index.size, dtype=float) / float(index.size)
        bounds = Bounds(np.zeros(index.size), np.ones(index.size))

    rng = np.random.default_rng(int(random_seed) if random_seed is not None else 0)

    # Feasible initialization: baseline q generally matches marginals by construction.
    # The product of marginals is used so marginals match exactly; sum-to-one is also satisfied.
    x0 = _product_baseline(index, marginals)
    if not use_kl:
        # A small random perturbation is applied to help trust-constr avoid some degenerate Hessians.
        noise = rng.normal(scale=1e-6, size=index.size)
        x0 = np.clip(x0 + noise, 0.0, None)
        x0 = x0 / float(np.sum(x0))

    w = np.asarray(w, dtype=float)
    t = np.asarray(t, dtype=float)
    A = np.asarray(A, dtype=float)

    def fun(p: np.ndarray) -> float:
        p = np.asarray(p, dtype=float)
        # Weighted least squares are used in residual form to avoid forming large quadratic matrices.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            r = A @ p - t
        if not np.all(np.isfinite(r)):
            return 1e100
        val = float(np.sum(w * r * r))
        if use_kl:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
                kl = float(np.sum(p * (np.log(p) - np.log(q))))
            if not np.isfinite(kl):
                return 1e100
            val += float(kl_weight) * kl
        return float(val)

    def jac(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            r = A @ p - t
        if not np.all(np.isfinite(r)):
            return np.zeros(index.size, dtype=float)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            grad = (2.0 * (A.T @ (w * r))).astype(float)
        if use_kl:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
                grad = grad + float(kl_weight) * (np.log(p) - np.log(q) + 1.0)
        return grad

    solver_name = "scipy.optimize.minimize(trust-constr)"
    try:
        res = minimize(
            fun,
            x0,
            method="trust-constr",
            jac=jac,
            constraints=[lin],
            bounds=bounds,
            options={
                "maxiter": int(solver_maxiter),
                "verbose": 0,
                "gtol": 1e-10,
                "xtol": 1e-12,
            },
        )
    except ValueError:
        # A fallback is used for rare numerical failures inside trust-constr.
        # The fallback is intended to be slower but more forgiving for some ill-conditioned problems.
        solver_name = "scipy.optimize.minimize(SLSQP)"
        lb = float(bounds.lb[0]) if np.ndim(bounds.lb) else float(bounds.lb)
        ub = float(bounds.ub[0]) if np.ndim(bounds.ub) else float(bounds.ub)
        bnds = [(lb, ub) for _ in range(index.size)]
        # C and d are captured by reference in the lambdas and are not modified in this scope.
        eq = {
            "type": "eq",
            "fun": lambda p: (C @ np.asarray(p, dtype=float) - d),
            "jac": lambda _p: C,
        }
        res = minimize(
            fun,
            x0,
            method="SLSQP",
            jac=jac,
            constraints=[eq],
            bounds=bnds,
            options={"maxiter": int(solver_maxiter), "ftol": 1e-12, "disp": False},
        )

    if not res.success:
        raise RuntimeError(f"Direct fit failed: {res.message}")

    p = np.asarray(res.x, dtype=float)
    # Numerical cleanliness is enforced.
    p[p < 0.0] = 0.0
    s = float(np.sum(p))
    if s <= 0.0:
        raise RuntimeError("Direct fit returned degenerate distribution")
    p = p / s
    if not return_report:
        return p

    # Fit report values are computed from the returned distribution.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        r_final = A @ p - t
    wls_val = float(np.sum(w * r_final * r_final)) if r_final.size else 0.0
    if use_kl:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            kl_val = float(np.sum(p * (np.log(p) - np.log(q))))
        if not np.isfinite(kl_val):
            kl_val = float("nan")
    else:
        kl_val = 0.0

    # Linear constraint residuals are computed (marginals + sum-to-one).
    lin_res = C @ p - d
    max_abs_marg_res = float(np.max(np.abs(lin_res))) if lin_res.size else 0.0
    max_abs_pair_res = float(np.max(np.abs(r_final))) if r_final.size else 0.0

    report = FitReport(
        method="direct",
        solver=str(solver_name),
        success=bool(res.success),
        message=str(res.message),
        n_iterations=int(getattr(res, "nit", 0)) if getattr(res, "nit", None) is not None else None,
        solver_status=int(getattr(res, "status", 0)) if getattr(res, "status", None) is not None else None,
        objective_value=float(wls_val + float(kl_weight) * float(kl_val)),
        wls_value=float(wls_val),
        kl_value=float(kl_val),
        kl_weight=float(kl_weight),
        max_abs_marginal_residual=float(max_abs_marg_res),
        max_abs_pairwise_residual=float(max_abs_pair_res),
        weight_by_target=bool(weight_by_target),
        feasibility_mode=str(feasibility_mode),
        feasibility_adjustments=tuple(feasibility_adjustments),
    )
    return p, report

