"""
Scenario analysis tools for CIB systems.

This module provides utilities for enumerating scenarios, finding consistent
scenarios, and ranking scenarios by various metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cib.core import CIBMatrix, ConsistencyChecker, Scenario
from cib.constraints import ConstraintIndex, ConstraintSpec

# Imports for Monte Carlo are attempted.
try:
    from cib.uncertainty import UncertainCIBMatrix
except ImportError:
    UncertainCIBMatrix = None

try:
    from cib.bayesian import GaussianCIBMatrix
except ImportError:
    GaussianCIBMatrix = None

from cib.example_data import seeds_for_run


class ScenarioAnalyzer:
    """
    Analyzes scenarios in a CIB system.

    Provides methods to enumerate all possible scenarios, find consistent
    scenarios, and filter or rank scenarios based on consistency.
    """

    def __init__(self, matrix: CIBMatrix) -> None:
        """
        Initialize analyzer with a CIB matrix.

        Args:
            matrix: CIB matrix to analyze.
        """
        self.matrix = matrix

    def enumerate_scenarios(self) -> List[Scenario]:
        """
        Generate all possible scenarios for the matrix.

        Returns:
            List of all possible scenario combinations.

        Note:
            The number of scenarios grows exponentially with the number
            of descriptors and states. Use with caution for large systems.
        """
        descriptor_names = list(self.matrix.descriptors.keys())
        state_lists = [
            self.matrix.descriptors[desc] for desc in descriptor_names
        ]

        scenarios: List[Scenario] = []
        for state_combination in product(*state_lists):
            state_dict = dict(zip(descriptor_names, state_combination))
            scenario = Scenario(state_dict, self.matrix)
            scenarios.append(scenario)

        return scenarios

    def find_all_consistent(
        self,
        max_scenarios: Optional[int] = None,
        n_restarts: int = 200,
        seed: Optional[int] = None,
        max_iterations: int = 1000,
        constraints: Optional[Sequence[ConstraintSpec]] = None,
    ) -> List[Scenario]:
        """
        Find all consistent scenarios in the system.

        For small systems (R ≤ 50k), uses exhaustive enumeration.
        For larger systems, uses succession-based random restarts to find a
        shortlist of consistent attractors.

        Args:
            max_scenarios: Maximum number of scenarios to enumerate.
                If None, enumerates all scenarios (only for small systems).
            n_restarts: Number of random initial scenarios to use when the
                system is too large to enumerate.
            seed: Random seed for reproducibility of random restarts.
            max_iterations: Maximum iterations per succession run.

        Returns:
            List of all consistent scenarios found.
        """
        total_scenarios = 1
        for count in self.matrix.state_counts:
            total_scenarios *= count

        if max_scenarios is not None and max_scenarios <= 0:
            raise ValueError("max_scenarios must be positive if specified")
        if n_restarts <= 0:
            raise ValueError("n_restarts must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        can_enumerate = total_scenarios <= 50000
        if max_scenarios is not None:
            can_enumerate = can_enumerate and total_scenarios <= max_scenarios

        if can_enumerate:
            all_scenarios = self.enumerate_scenarios()
            return self.filter_consistent(all_scenarios, constraints=constraints)

        return self.find_consistent_via_random_restarts(
            n_restarts=n_restarts,
            seed=seed,
            max_iterations=max_iterations,
            constraints=constraints,
        )

    def find_consistent_via_random_restarts(
        self,
        n_restarts: int = 200,
        seed: Optional[int] = None,
        max_iterations: int = 1000,
        constraints: Optional[Sequence[ConstraintSpec]] = None,
    ) -> List[Scenario]:
        """
        Find a shortlist of consistent scenarios via succession random restarts.

        This is the recommended workflow for medium/large systems where full
        enumeration is infeasible.
        """
        from cib.succession import GlobalSuccession

        results = self.find_attractors_via_random_restarts(
            n_restarts=n_restarts,
            seed=seed,
            max_iterations=max_iterations,
            succession_operator=GlobalSuccession(),
        )

        consistent: List[Scenario] = []
        seen: set[Scenario] = set()
        cidx = ConstraintIndex.from_specs(self.matrix, constraints)
        for res in results:
            if res.is_cycle:
                continue
            attractor = res.attractor
            if not isinstance(attractor, Scenario):
                continue
            if attractor in seen:
                continue
            if cidx is not None and not bool(cidx.is_full_valid(attractor.to_indices())):
                continue
            if ConsistencyChecker.check_consistency(attractor, self.matrix):
                consistent.append(attractor)
                seen.add(attractor)

        return consistent

    def find_attractors_via_random_restarts(
        self,
        n_restarts: int = 200,
        seed: Optional[int] = None,
        max_iterations: int = 1000,
        succession_operator=None,
    ):
        """
        Find attractors (fixed points or cycles) via random restarts.

        Returns:
            List of AttractorResult objects.
        """
        from cib.succession import AttractorResult, AttractorFinder, GlobalSuccession

        if succession_operator is None:
            succession_operator = GlobalSuccession()

        rng = np.random.default_rng(seed)
        initial_scenarios: List[Scenario] = []
        descriptor_names = list(self.matrix.descriptors.keys())
        for _ in range(n_restarts):
            state_dict: Dict[str, str] = {}
            for desc in descriptor_names:
                states = self.matrix.descriptors[desc]
                state_dict[desc] = states[int(rng.integers(0, len(states)))]
            initial_scenarios.append(Scenario(state_dict, self.matrix))

        finder = AttractorFinder(self.matrix)
        results: List[AttractorResult] = []
        for initial in initial_scenarios:
            results.append(
                succession_operator.find_attractor(
                    initial, self.matrix, max_iterations=max_iterations
                )
            )
        # Results are de-duplicated by attractor identity (fixed point or first element of cycle).
        unique: dict[Scenario, AttractorResult] = {}
        for res in results:
            if isinstance(res.attractor, Scenario):
                key = res.attractor
            else:
                cycle = res.attractor
                key = min(cycle, key=lambda s: tuple(s.to_indices()))
            unique.setdefault(key, res)
        return list(unique.values())

    def find_attractors_monte_carlo(
        self,
        *,
        config=None,
    ):
        """
        Find attractors and estimate weights via Monte Carlo sampling.

        This method is intended for large scenario spaces where complete
        enumeration is infeasible.
        """
        from cib.solvers.config import MonteCarloAttractorConfig
        from cib.solvers.monte_carlo_attractors import (
            MonteCarloAttractorResult,
            find_attractors_monte_carlo,
        )

        cfg = config if config is not None else MonteCarloAttractorConfig()
        if not isinstance(cfg, MonteCarloAttractorConfig):
            raise ValueError("config must be a MonteCarloAttractorConfig")
        res: MonteCarloAttractorResult = find_attractors_monte_carlo(
            matrix=self.matrix, config=cfg
        )
        return res

    def find_all_consistent_exact(
        self,
        *,
        config=None,
    ):
        """
        Enumerate all consistent scenarios using a pruned exact solver.
        """
        from cib.solvers.config import ExactSolverConfig
        from cib.solvers.exact_pruned import ExactSolverResult, find_all_consistent_exact

        cfg = config if config is not None else ExactSolverConfig()
        if not isinstance(cfg, ExactSolverConfig):
            raise ValueError("config must be an ExactSolverConfig")
        res: ExactSolverResult = find_all_consistent_exact(matrix=self.matrix, config=cfg)
        return res

    def filter_consistent(
        self,
        candidates: List[Scenario],
        *,
        constraints: Optional[Sequence[ConstraintSpec]] = None,
    ) -> List[Scenario]:
        """
        Filter consistent scenarios from a candidate list.

        Args:
            candidates: List of scenarios to check.

        Returns:
            List of scenarios that are consistent.
        """
        consistent: List[Scenario] = []
        cidx = ConstraintIndex.from_specs(self.matrix, constraints)
        for scenario in candidates:
            if cidx is not None and not bool(cidx.is_full_valid(scenario.to_indices())):
                continue
            if ConsistencyChecker.check_consistency(scenario, self.matrix):
                consistent.append(scenario)

        return consistent

    def rank_scenarios(
        self, scenarios: List[Scenario]
    ) -> List[Tuple[Scenario, float]]:
        """
        Rank scenarios by consistency strength.

        Consistency strength is measured as the minimum gap between the
        chosen state's impact score and all other states' impact scores
        for each descriptor. Higher values indicate stronger consistency.

        Args:
            scenarios: List of scenarios to rank.

        Returns:
            List of (scenario, strength) tuples, sorted by strength
            in descending order.
        """
        from cib.core import ImpactBalance
        import numpy as np

        ranked: List[Tuple[Scenario, float]] = []

        for scenario in scenarios:
            min_gap = float("inf")

            for descriptor in self.matrix.descriptors:
                current_state = scenario.get_state(descriptor)
                balance = ImpactBalance(scenario, self.matrix)
                current_score = balance.get_score(descriptor, current_state)

                for state in self.matrix.descriptors[descriptor]:
                    if state != current_state:
                        score = balance.get_score(descriptor, state)
                        gap = current_score - score
                        if gap < min_gap:
                            min_gap = gap

            ranked.append((scenario, min_gap))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


@dataclass
class MonteCarloResults:
    """
    Results from Monte Carlo consistency probability estimation.

    Attributes:
        scenario_probabilities: Dictionary mapping scenarios to their
            estimated consistency probabilities.
        confidence_intervals: Dictionary mapping scenarios to (lower, upper)
            confidence interval tuples.
        n_samples: Number of Monte Carlo samples used.
    """

    scenario_probabilities: Dict[Scenario, float]
    confidence_intervals: Dict[Scenario, Tuple[float, float]]
    n_samples: int


class MonteCarloAnalyzer:
    """
    Monte Carlo analyzer for probabilistic consistency estimation.

    Estimates P(consistent | z) for scenarios by sampling uncertain CIB
    matrices and checking consistency in each sample.
    """

    def __init__(
        self,
        matrix: object,
        n_samples: int = 10000,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize Monte Carlo analyzer.

        Args:
            matrix: UncertainCIBMatrix with confidence codes.
            n_samples: Number of Monte Carlo samples to use.
            seed: Random seed for reproducibility. If None, uses random seed.

        Raises:
            ValueError: If matrix is not an UncertainCIBMatrix or if
                n_samples is non-positive.
        """
        if not hasattr(matrix, "sample_matrix"):
            raise ValueError(
                "Matrix must provide a sample_matrix(seed: int) method"
            )
        # Backwards-compatible type checks are performed for common implementations.
        if UncertainCIBMatrix is not None and isinstance(matrix, UncertainCIBMatrix):
            pass
        elif GaussianCIBMatrix is not None and isinstance(matrix, GaussianCIBMatrix):
            pass
        else:
            # Other sampling-capable matrices are accepted (duck-typing).
            pass
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        self.matrix = matrix  # type: ignore[assignment]
        self.n_samples = n_samples
        self.base_seed = seed if seed is not None else np.random.randint(0, 2**31)

    def estimate_consistency_probability(
        self, scenario: Scenario
    ) -> float:
        """
        Estimate P(consistent | z) for a scenario.

        Args:
            scenario: Scenario to evaluate.

        Returns:
            Estimated probability of consistency (between 0 and 1).
        """
        consistent_count = self._estimate_consistency_count(scenario)
        return consistent_count / self.n_samples

    def _estimate_consistency_count(self, scenario: Scenario) -> int:
        """
        Estimate the number of Monte Carlo draws where the scenario is consistent.
        """
        consistent_count = 0
        for sample_idx in range(self.n_samples):
            if seeds_for_run is not None:
                seeds = seeds_for_run(self.base_seed, sample_idx)
                sample_seed = seeds["judgment_uncertainty_seed"]
            else:
                sample_seed = self.base_seed + sample_idx

            sampled_matrix = self.matrix.sample_matrix(sample_seed)
            if ConsistencyChecker.check_consistency(scenario, sampled_matrix):
                consistent_count += 1
        return consistent_count

    def score_candidates(
        self, candidates: List[Scenario]
    ) -> MonteCarloResults:
        """
        Score multiple candidate scenarios.

        Args:
            candidates: List of scenarios to evaluate.

        Returns:
            MonteCarloResults containing probabilities and confidence intervals.
        """
        scenario_probabilities: Dict[Scenario, float] = {}
        confidence_intervals: Dict[Scenario, Tuple[float, float]] = {}

        for scenario in candidates:
            k = self._estimate_consistency_count(scenario)
            prob = k / self.n_samples
            scenario_probabilities[scenario] = prob
            confidence_intervals[scenario] = self._wilson_interval_from_count(
                k, self.n_samples, level=0.95
            )

        return MonteCarloResults(
            scenario_probabilities=scenario_probabilities,
            confidence_intervals=confidence_intervals,
            n_samples=self.n_samples,
        )

    def get_confidence_intervals(
        self, scenario: Scenario, level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for consistency probability.

        Uses Wilson score interval for binomial proportion.

        Args:
            scenario: Scenario to evaluate.
            level: Confidence level (default 0.95 for 95% interval).

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        k = self._estimate_consistency_count(scenario)
        return self._wilson_interval_from_count(k, self.n_samples, level=level)

    @staticmethod
    def _wilson_interval_from_count(
        k: int, n: int, level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Wilson score interval for a binomial proportion given success count k.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        if k < 0 or k > n:
            raise ValueError("k must be in [0, n]")
        if not (0.0 < float(level) < 1.0):
            raise ValueError("level must be between 0 and 1")

        # Scipy is preferred if available for arbitrary confidence levels.
        z: float
        try:
            from scipy.stats import norm  # type: ignore

            z = float(norm.ppf(0.5 + float(level) / 2.0))
        except Exception:
            # A fallback is used for common levels; otherwise approximation is performed with 95%.
            if np.isclose(level, 0.95):
                z = 1.96
            elif np.isclose(level, 0.99):
                z = 2.576
            elif np.isclose(level, 0.90):
                z = 1.645
            else:
                z = 1.96

        p = k / n
        denominator = 1 + (z**2 / n)
        center = (p + (z**2 / (2 * n))) / denominator
        margin = (z / denominator) * np.sqrt(
            (p * (1 - p) / n) + (z**2 / (4 * n**2))
        )
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        return (float(lower), float(upper))
