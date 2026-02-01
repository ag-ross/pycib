"""
Configuration objects for solver modes.

In this module, configuration dataclasses are provided as a stable, typed
surface for user-selectable solver modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Mapping, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from cib.constraints import ConstraintSpec


SolverMode = Literal[
    "exact",
    "monte_carlo_attractors",
    "branching_hybrid",
    "dynamic_monte_carlo",
]


@dataclass(frozen=True)
class MonteCarloAttractorConfig:
    """
    Configuration for Monte Carlo attractor discovery.
    """

    runs: int = 10_000
    succession: Literal["global", "local"] = "global"
    max_iterations: int = 1000
    seed: int = 123
    bitgen: Literal["PCG64"] = "PCG64"

    sampler: Literal["uniform", "weighted"] = "uniform"
    sampler_weights: Optional[Dict[str, Dict[str, float]]] = None

    n_jobs: int = 1
    cycle_mode: Literal[
        "keep_cycle",
        "representative_first",
        "representative_random",
    ] = "keep_cycle"
    cycle_key_policy: Literal["min_state", "rotate_min"] = "min_state"

    use_fast_scoring: bool = True
    strict_fast: bool = False
    fast_backend: Literal["dense", "sparse"] = "dense"

    result_storage: Literal["counts_only", "topk_scenarios"] = "counts_only"
    top_k: int = 50

    float_atol: float = 1e-08
    float_rtol: float = 1e-05

    def validate(self) -> None:
        """
        Configuration validation is performed.
        """
        if int(self.runs) <= 0:
            raise ValueError("runs must be positive")
        if int(self.max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        if int(self.n_jobs) <= 0:
            raise ValueError("n_jobs must be positive")
        if int(self.top_k) <= 0:
            raise ValueError("top_k must be positive")
        if float(self.float_atol) < 0.0 or float(self.float_rtol) < 0.0:
            raise ValueError("float_atol and float_rtol must be non-negative")

        if str(self.bitgen) != "PCG64":
            raise ValueError("bitgen is not recognised")
        if str(self.cycle_mode) not in {
            "keep_cycle",
            "representative_first",
            "representative_random",
        }:
            raise ValueError("cycle_mode is not recognised")
        if str(self.cycle_key_policy) not in {"min_state", "rotate_min"}:
            raise ValueError("cycle_key_policy is not recognised")

        if str(self.fast_backend) not in {"dense", "sparse"}:
            raise ValueError("fast_backend is not recognised")

        if self.sampler == "weighted":
            if not self.sampler_weights:
                raise ValueError(
                    "sampler_weights must be provided when sampler is 'weighted'"
                )


@dataclass(frozen=True)
class ExactSolverConfig:
    """
    Configuration for exact consistency enumeration.
    """

    ordering: Literal["given", "connectivity", "random"] = "given"
    bound: Literal["none", "safe_upper_bound_v1"] = "safe_upper_bound_v1"

    max_solutions: Optional[int] = None
    time_limit_s: Optional[float] = None

    use_fast_scoring: bool = True
    strict_fast: bool = False

    float_atol: float = 1e-08
    float_rtol: float = 1e-05

    seed: int = 123
    constraints: Optional[Sequence["ConstraintSpec"]] = None

    def validate(self) -> None:
        """
        Configuration validation is performed.
        """
        if self.max_solutions is not None and int(self.max_solutions) <= 0:
            raise ValueError("max_solutions must be positive when provided")
        if self.time_limit_s is not None and float(self.time_limit_s) <= 0.0:
            raise ValueError("time_limit_s must be positive when provided")
        if float(self.float_atol) < 0.0 or float(self.float_rtol) < 0.0:
            raise ValueError("float_atol and float_rtol must be non-negative")


@dataclass(frozen=True)
class BranchingHybridConfig:
    """
    Configuration for hybrid branching.

    Note: `BranchingPathwayBuilder` already exposes most relevant parameters.
    This config is provided for future unification of solver entrypoints.
    """

    prune_policy: Literal["incoming_mass", "per_parent_topk", "min_edge_weight"] = (
        "incoming_mass"
    )
    min_edge_weight: Optional[float] = None
    n_jobs: int = 1
    use_fast_scoring: bool = True
    float_atol: float = 1e-08
    float_rtol: float = 1e-05

    def validate(self) -> None:
        if int(self.n_jobs) <= 0:
            raise ValueError("n_jobs must be positive")
        if self.min_edge_weight is not None and float(self.min_edge_weight) < 0.0:
            raise ValueError("min_edge_weight must be non-negative when provided")

