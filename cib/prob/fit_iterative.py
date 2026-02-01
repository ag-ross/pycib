from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from cib.prob.approx import ApproxJointDistribution
from cib.prob.constraints import Marginals, Multipliers, validate_marginals
from cib.prob.fit_report import FitReport
from cib.prob.types import FactorSpec, ScenarioIndex


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = float(np.max(logits))
    z = logits - m
    e = np.exp(z)
    s = float(np.sum(e))
    if s <= 0.0 or not np.isfinite(s):
        return np.ones_like(logits) / float(logits.size)
    return e / s


def fit_joint_iterative(
    *,
    index: ScenarioIndex,
    marginals: Marginals,
    multipliers: Optional[Multipliers] = None,
    relevance_weights: Optional[Mapping[Tuple[str, str], float]] = None,
    random_seed: Optional[int] = None,
    burn_in_sweeps: int = 2_000,
    n_samples: int = 10_000,
    thinning: int = 5,
    eps: float = 1e-15,
    with_report: bool = False,
) -> Union[ApproxJointDistribution, Tuple[ApproxJointDistribution, FitReport]]:
    """
    Fit an approximate distribution using a Gibbs sampler over a dependency-network conditional model.

    This method is intended for large scenario spaces where dense enumeration is not feasible.
    The stationary distribution of the implied Gibbs chain is an approximation; marginal agreement
    with the input marginals is not guaranteed and should be checked via diagnostics.
    """
    multipliers = multipliers or {}
    validate_marginals(index.factors, marginals)
    burn_in_sweeps = int(burn_in_sweeps)
    n_samples = int(n_samples)
    thinning = int(thinning)
    eps = float(eps)

    rng = np.random.default_rng(int(random_seed) if random_seed is not None else 0)

    n_factors = len(index.factors)
    outcomes_by_factor = [list(f.outcomes) for f in index.factors]
    n_outcomes = [len(o) for o in outcomes_by_factor]

    # Baseline log-probabilities are derived from marginals.
    log_p0: list[np.ndarray] = []
    for f in index.factors:
        vals = np.array([float(marginals[f.name][o]) for o in f.outcomes], dtype=float)
        vals = np.maximum(vals, eps)
        vals = vals / float(np.sum(vals))
        log_p0.append(np.log(vals))

    # Multiplier contributions are encoded as log-multipliers for (i <- j=b) contexts.
    # For a fixed i and current outcomes of neighbours, a conditional is formed as:
    #   log P(i=a | rest) ∝ log P0_i(a) + sum_j w(i,j) * log m_{(i=a)<-(j=b_j)}.
    contrib: Dict[Tuple[int, int, str], np.ndarray] = {}
    pos_by_name = {n: i for i, n in enumerate(index.factor_names)}

    for (i_a, j_b), m in multipliers.items():
        (i_name, a_name) = i_a
        (j_name, b_name) = j_b
        i_pos = pos_by_name[str(i_name)]
        j_pos = pos_by_name[str(j_name)]
        b_name = str(b_name)

        w = 1.0
        if relevance_weights is not None:
            w = float(relevance_weights.get((str(i_name), str(j_name)), 1.0))
        if w == 0.0:
            continue

        # Vector over outcomes of i.
        vec = contrib.get((i_pos, j_pos, b_name))
        if vec is None:
            vec = np.zeros(n_outcomes[i_pos], dtype=float)
        try:
            a_idx = outcomes_by_factor[i_pos].index(str(a_name))
        except ValueError:
            continue
        m = float(m)
        if m <= 0.0:
            # Non-positive multipliers are treated as invalid and are ignored here.
            continue
        vec[a_idx] = vec[a_idx] + w * float(np.log(m))
        contrib[(i_pos, j_pos, b_name)] = vec

    # Initial assignment is sampled from marginals.
    state = [0] * n_factors
    for i in range(n_factors):
        probs = _softmax(log_p0[i])
        state[i] = int(rng.choice(np.arange(n_outcomes[i]), p=probs))

    counts: Dict[int, int] = {}
    collected = 0
    sweeps = 0

    def sweep() -> None:
        for i in range(n_factors):
            scores = np.array(log_p0[i], copy=True)
            for j in range(n_factors):
                if j == i:
                    continue
                b = outcomes_by_factor[j][state[j]]
                vec = contrib.get((i, j, str(b)))
                if vec is not None:
                    scores = scores + vec
            probs = _softmax(scores)
            state[i] = int(rng.choice(np.arange(n_outcomes[i]), p=probs))

    while sweeps < burn_in_sweeps:
        sweep()
        sweeps += 1

    while collected < n_samples:
        for _ in range(max(1, thinning)):
            sweep()
            sweeps += 1
        assignment = [outcomes_by_factor[i][state[i]] for i in range(n_factors)]
        idx = int(index.index_of(assignment))
        counts[idx] = int(counts.get(idx, 0) + 1)
        collected += 1

    support_indices = np.array(list(counts.keys()), dtype=int)
    support_counts = np.array([float(counts[int(i)]) for i in support_indices], dtype=float)
    support_prob = support_counts / float(np.sum(support_counts))

    dist = ApproxJointDistribution(
        index=index,
        support_indices=support_indices,
        support_probabilities=support_prob,
        n_samples=int(n_samples),
        fit_report=None,
    )

    if not with_report:
        return dist

    report = FitReport(
        method="iterative",
        solver="gibbs_dependency_network",
        success=True,
        message=f"Samples collected: n_samples={int(n_samples)}, burn_in_sweeps={int(burn_in_sweeps)}, thinning={int(thinning)}",
        n_iterations=int(sweeps),
        solver_status=None,
        objective_value=float("nan"),
        wls_value=float("nan"),
        kl_value=float("nan"),
        kl_weight=0.0,
        max_abs_marginal_residual=float("nan"),
        max_abs_pairwise_residual=float("nan"),
        weight_by_target=False,
        feasibility_mode="approximate",
        feasibility_adjustments=tuple(),
    )
    dist = ApproxJointDistribution(
        index=index,
        support_indices=support_indices,
        support_probabilities=support_prob,
        n_samples=int(n_samples),
        fit_report=report,
    )
    return dist, report

