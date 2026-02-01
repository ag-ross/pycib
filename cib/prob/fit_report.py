from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class FeasibilityAdjustment:
    """
    Record of a repair adjustment applied to a multiplier-implied pairwise target.
    """

    i: str
    a: str
    j: str
    b: str
    original_value: float
    adjusted_value: float
    frechet_lower: float
    frechet_upper: float


@dataclass(frozen=True)
class FitReport:
    """
    Fit report intended for research use and auditability.

    The report is treated as a deterministic description of a single fit attempt,
    including objective decomposition and basic constraint residual summaries.
    """

    method: str
    solver: str
    success: bool
    message: str
    n_iterations: Optional[int]
    solver_status: Optional[int]

    objective_value: float
    wls_value: float
    kl_value: float
    kl_weight: float

    max_abs_marginal_residual: float
    max_abs_pairwise_residual: float

    weight_by_target: bool
    feasibility_mode: str
    feasibility_adjustments: Tuple[FeasibilityAdjustment, ...]

