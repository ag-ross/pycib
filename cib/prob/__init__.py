"""
Joint-distribution probabilistic cross-impact analysis (CIA).

This subpackage is intentionally separate from the existing `cib` probabilistic
outputs (which are empirical Monte Carlo / branching frequencies from the CIB
simulation model). Here, "probability" means an explicit joint distribution
over factor outcomes constrained by marginals and probabilistic cross-impact
multipliers.
"""

from cib.prob.types import FactorSpec, ProbScenario, ScenarioIndex
from cib.prob.model import ProbabilisticCIAModel, JointDistribution
from cib.prob.approx import ApproxJointDistribution
from cib.prob.diagnostics import DiagnosticsReport
from cib.prob.dynamic import DynamicProbabilisticCIA
from cib.prob.fit_report import FeasibilityAdjustment, FitReport
from cib.prob.graph import RelevanceSpec
from cib.prob.risk_bounds import RiskBoundsResult, event_probability_bounds

__all__ = [
    "FactorSpec",
    "ProbScenario",
    "ScenarioIndex",
    "ProbabilisticCIAModel",
    "JointDistribution",
    "ApproxJointDistribution",
    "DiagnosticsReport",
    "DynamicProbabilisticCIA",
    "FeasibilityAdjustment",
    "FitReport",
    "RelevanceSpec",
    "RiskBoundsResult",
    "event_probability_bounds",
]

