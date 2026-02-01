from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from cib.prob.fit_report import FitReport
from cib.prob.types import AssignmentLike, ScenarioIndex


@dataclass(frozen=True)
class ApproxJointDistribution:
    """
    Approximate joint distribution represented by a finite sampled support.

    The distribution is represented as a sparse probability mass function over
    the visited scenario indices. Unvisited scenarios are treated as having
    probability zero.
    """

    index: ScenarioIndex
    support_indices: np.ndarray  # shape (K,), integer indices into ScenarioIndex
    support_probabilities: np.ndarray  # shape (K,), probabilities summing to 1
    n_samples: int
    fit_report: Optional[FitReport] = None

    def __post_init__(self) -> None:
        if self.support_indices.ndim != 1:
            raise ValueError("support_indices must be 1D")
        if self.support_probabilities.ndim != 1:
            raise ValueError("support_probabilities must be 1D")
        if int(self.support_indices.shape[0]) != int(self.support_probabilities.shape[0]):
            raise ValueError("support arrays must have the same length")
        if np.any(self.support_probabilities < 0.0):
            raise ValueError("support_probabilities must be non-negative")
        s = float(np.sum(self.support_probabilities))
        if not np.isfinite(s) or abs(s - 1.0) > 1e-8:
            raise ValueError("support_probabilities must sum to 1")

    def scenario_prob(self, assignment: AssignmentLike) -> float:
        idx = int(self.index.index_of(assignment))
        # A linear scan is used. Support sizes are expected to remain modest relative to the
        # full scenario space, but this may be optimised if required.
        for i, p in zip(self.support_indices, self.support_probabilities):
            if int(i) == idx:
                return float(p)
        return 0.0

    def marginal(self, factor: str) -> Dict[str, float]:
        factor = str(factor)
        if factor not in self.index.factor_names:
            raise ValueError(f"Unknown factor: {factor!r}")
        pos = list(self.index.factor_names).index(factor)
        outs = list(self.index.factors[pos].outcomes)
        out: Dict[str, float] = {o: 0.0 for o in outs}
        for idx, p in zip(self.support_indices, self.support_probabilities):
            scen = self.index.scenario_at(int(idx))
            out[scen.assignment[pos]] += float(p)
        return out

    def pairwise_marginal(self, i: str, a: str, j: str, b: str) -> float:
        i = str(i)
        j = str(j)
        pos_i = list(self.index.factor_names).index(i)
        pos_j = list(self.index.factor_names).index(j)
        total = 0.0
        for idx, p in zip(self.support_indices, self.support_probabilities):
            scen = self.index.scenario_at(int(idx))
            if scen.assignment[pos_i] == str(a) and scen.assignment[pos_j] == str(b):
                total += float(p)
        return float(total)

    def conditional(self, target: Tuple[str, str], given: Tuple[str, str], *, eps: float = 1e-15) -> float:
        (ti, ta) = (str(target[0]), str(target[1]))
        (gj, gb) = (str(given[0]), str(given[1]))
        num = self.pairwise_marginal(ti, ta, gj, gb)
        denom = self.marginal(gj).get(gb, 0.0)
        if denom <= float(eps):
            return float("nan")
        return float(num / denom)

    def top_scenarios(self, n: int = 10) -> List[Tuple[float, Mapping[str, str]]]:
        """
        Return a list of (probability, assignment_dict) pairs for the support top-n.
        """
        n = int(n)
        pairs = [(float(p), int(i)) for i, p in zip(self.support_indices, self.support_probabilities)]
        pairs.sort(key=lambda x: x[0], reverse=True)
        out: List[Tuple[float, Mapping[str, str]]] = []
        for p, idx in pairs[: max(0, n)]:
            out.append((float(p), self.index.scenario_at(int(idx)).to_dict()))
        return out

