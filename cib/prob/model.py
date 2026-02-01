from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from cib.prob.constraints import (
    Marginals,
    Multipliers,
    multiplier_normalization_issues,
    multiplier_pairwise_targets,
    pairwise_target_frechet_violations,
    validate_marginals,
)
from cib.prob.approx import ApproxJointDistribution
from cib.prob.fit_direct import fit_joint_direct
from cib.prob.fit_iterative import fit_joint_iterative
from cib.prob.fit_report import FitReport
from cib.prob.graph import RelevanceSpec, relevance_weight
from cib.prob.risk_bounds import RiskBoundsResult, event_probability_bounds
from cib.prob.types import AssignmentLike, FactorSpec, ScenarioIndex


@dataclass(frozen=True)
class JointDistribution:
    """
    Dense joint distribution over all scenarios (small scenario spaces).
    """

    index: ScenarioIndex
    p: np.ndarray  # shape (index.size,)
    fit_report: Optional[FitReport] = None

    def __post_init__(self) -> None:
        if self.p.ndim != 1:
            raise ValueError("p must be 1D")
        if int(self.p.shape[0]) != int(self.index.size):
            raise ValueError("p has wrong length")

    def scenario_prob(self, assignment: AssignmentLike) -> float:
        return float(self.p[self.index.index_of(assignment)])

    def marginal(self, factor: str) -> Dict[str, float]:
        factor = str(factor)
        if factor not in self.index.factor_names:
            raise ValueError(f"Unknown factor: {factor!r}")
        pos = list(self.index.factor_names).index(factor)
        outs = list(self.index.factors[pos].outcomes)
        out: Dict[str, float] = {o: 0.0 for o in outs}
        for idx in range(self.index.size):
            scen = self.index.scenario_at(idx)
            out[scen.assignment[pos]] += float(self.p[idx])
        return out

    def pairwise_marginal(self, i: str, a: str, j: str, b: str) -> float:
        i = str(i)
        j = str(j)
        pos_i = list(self.index.factor_names).index(i)
        pos_j = list(self.index.factor_names).index(j)
        total = 0.0
        for idx in range(self.index.size):
            scen = self.index.scenario_at(idx)
            if scen.assignment[pos_i] == str(a) and scen.assignment[pos_j] == str(b):
                total += float(self.p[idx])
        return float(total)

    def conditional(self, target: Tuple[str, str], given: Tuple[str, str], *, eps: float = 1e-15) -> float:
        """
        Return P(target_factor=target_outcome | given_factor=given_outcome).
        """
        (i, a) = (str(target[0]), str(target[1]))
        (j, b) = (str(given[0]), str(given[1]))
        num = self.pairwise_marginal(i, a, j, b)
        denom = self.marginal(j).get(b, 0.0)
        if denom <= float(eps):
            return float("nan")
        return float(num / denom)


class ProbabilisticCIAModel:
    """
    Static joint-distribution probabilistic CIA model (point marginals + multipliers).
    """

    def __init__(
        self,
        *,
        factors: Sequence[FactorSpec],
        marginals: Marginals,
        multipliers: Optional[Multipliers] = None,
        feasibility_mode: str = "strict",
        feasibility_tol: float = 1e-9,
        enforce_multiplier_normalisation: bool = False,
    ) -> None:
        if not factors:
            raise ValueError("factors cannot be empty")
        self.factors = tuple(factors)
        self.index = ScenarioIndex(self.factors)
        self.marginals: Marginals = marginals
        self.multipliers: Multipliers = multipliers or {}
        self.feasibility_mode = str(feasibility_mode).strip().lower()
        self.feasibility_tol = float(feasibility_tol)
        self.enforce_multiplier_normalisation = bool(enforce_multiplier_normalisation)

        validate_marginals(self.factors, self.marginals)
        if self.feasibility_mode not in {"strict", "repair"}:
            raise ValueError(f"Unknown feasibility_mode: {self.feasibility_mode!r}")

        if self.feasibility_mode == "strict":
            # A strict pre-check is performed: Fréchet violations indicate incoherent constraints.
            targets = multiplier_pairwise_targets(self.marginals, self.multipliers)
            violations = pairwise_target_frechet_violations(
                self.marginals, targets, tol=float(self.feasibility_tol)
            )
            if violations:
                k = next(iter(violations.keys()))
                v, lo, hi = violations[k]
                raise ValueError(
                    "Multiplier-implied pairwise target violates Fréchet bounds: "
                    f"{k!r} has {v}, but bounds are [{lo}, {hi}]"
                )

        if self.enforce_multiplier_normalisation:
            issues = multiplier_normalization_issues(self.factors, self.marginals, self.multipliers)
            if issues:
                first = issues[0]
                raise ValueError(
                    "Multiplier normalisation constraint is violated for at least one context: "
                    f"(i={first.i!r}, j={first.j!r}, given_outcome={first.given_outcome!r})"
                )

    def fit_joint(
        self,
        *,
        method: str = "direct",
        max_scenarios: int = 20_000,
        kl_weight: float = 0.0,
        kl_baseline: Optional[np.ndarray] = None,
        kl_baseline_eps: float = 0.0,
        relevance: Optional[RelevanceSpec] = None,
        relevance_default_weight: float = 0.0,
        weight_by_target: bool = True,
        random_seed: Optional[int] = None,
        solver_maxiter: int = 2_000,
        iterative_burn_in_sweeps: int = 2_000,
        iterative_n_samples: int = 10_000,
        iterative_thinning: int = 5,
        iterative_eps: float = 1e-15,
        with_report: bool = False,
    ) -> Union[JointDistribution, ApproxJointDistribution]:
        method = str(method).strip().lower()
        if method not in {"direct", "iterative"}:
            raise ValueError(f"Unknown method: {method!r}")

        if method == "direct" and self.index.size > int(max_scenarios):
            raise ValueError(
                f"Scenario space too large for dense fit (size={self.index.size}, "
                f"max_scenarios={int(max_scenarios)})."
            )

        rel_map: Optional[Dict[Tuple[str, str], float]] = None
        if relevance is not None:
            rel_map = {}
            for (i_a, j_b), _m in self.multipliers.items():
                (i, _a) = i_a
                (j, _b) = j_b
                rel_map[(str(i), str(j))] = float(
                    relevance_weight(
                        child=str(i),
                        parent=str(j),
                        spec=relevance,
                        default_weight=float(relevance_default_weight),
                    )
                )

        if method == "iterative":
            if with_report:
                dist, _report = fit_joint_iterative(
                    index=self.index,
                    marginals=self.marginals,
                    multipliers=self.multipliers,
                    relevance_weights=rel_map,
                    random_seed=random_seed,
                    burn_in_sweeps=int(iterative_burn_in_sweeps),
                    n_samples=int(iterative_n_samples),
                    thinning=int(iterative_thinning),
                    eps=float(iterative_eps),
                    with_report=True,
                )
                # fit_joint_iterative returns the report separately; the distribution is already populated.
                return dist

            return fit_joint_iterative(
                index=self.index,
                marginals=self.marginals,
                multipliers=self.multipliers,
                relevance_weights=rel_map,
                random_seed=random_seed,
                burn_in_sweeps=int(iterative_burn_in_sweeps),
                n_samples=int(iterative_n_samples),
                thinning=int(iterative_thinning),
                eps=float(iterative_eps),
                with_report=False,
            )

        if with_report:
            p, report = fit_joint_direct(
                index=self.index,
                marginals=self.marginals,
                multipliers=self.multipliers,
                kl_weight=float(kl_weight),
                kl_baseline=kl_baseline,
                kl_baseline_eps=float(kl_baseline_eps),
                feasibility_mode=self.feasibility_mode,
                feasibility_tol=float(self.feasibility_tol),
                relevance_weights=rel_map,
                weight_by_target=bool(weight_by_target),
                random_seed=random_seed,
                solver_maxiter=int(solver_maxiter),
                return_report=True,
            )
            return JointDistribution(index=self.index, p=p, fit_report=report)

        p = fit_joint_direct(
            index=self.index,
            marginals=self.marginals,
            multipliers=self.multipliers,
            kl_weight=float(kl_weight),
            kl_baseline=kl_baseline,
            kl_baseline_eps=float(kl_baseline_eps),
            feasibility_mode=self.feasibility_mode,
            feasibility_tol=float(self.feasibility_tol),
            relevance_weights=rel_map,
            weight_by_target=bool(weight_by_target),
            random_seed=random_seed,
            solver_maxiter=int(solver_maxiter),
            return_report=False,
        )
        return JointDistribution(index=self.index, p=p, fit_report=None)

    def event_probability_bounds(
        self,
        *,
        event: Mapping[str, str],
        include_pairwise_targets: bool = True,
        max_scenarios: int = 50_000,
    ) -> RiskBoundsResult:
        """
        Return identification bounds on P(event) under the configured constraint set.

        This method is intended for small scenario spaces and uses dense linear programming.
        """
        return event_probability_bounds(
            factors=self.factors,
            marginals=self.marginals,
            multipliers=self.multipliers,
            event=event,
            include_pairwise_targets=bool(include_pairwise_targets),
            feasibility_mode=self.feasibility_mode,
            feasibility_tol=float(self.feasibility_tol),
            max_scenarios=int(max_scenarios),
        )

