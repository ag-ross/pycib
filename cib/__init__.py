"""
Core CIB analysis modules.

This package provides the fundamental components for Cross-Impact Balance
analysis, including matrix structures, scenario representation, consistency
checking, and succession operators.
"""

from cib.analysis import MonteCarloAnalyzer, ScenarioAnalyzer
from cib.bayesian import ExpertAggregator, GaussianCIBMatrix
from cib.cyclic import CyclicDescriptor
from cib.core import (
    CIBMatrix,
    ConsistencyChecker,
    ImpactBalance,
    Scenario,
)
from cib.dynamic import DynamicCIB
from cib.branching import BranchingPathwayBuilder, BranchingResult
from cib.scoring import (
    ScenarioDiagnostics,
    impact_label,
    judgment_section_labels,
    scenario_diagnostics,
)
from cib.pathway import (
    TransformationPathway,
    numeric_quantile_timelines,
    pathway_frequencies,
    state_probability_timelines,
)
from cib.shocks import RobustnessTester, ShockModel
from cib.succession import (
    AttractorFinder,
    GlobalSuccession,
    LocalSuccession,
    SuccessionOperator,
)
from cib.uncertainty import ConfidenceMapper, UncertainCIBMatrix
from cib.reduction import reduce_matrix, bin_states, map_scenario_to_reduced
from cib.utils import (
    load_from_csv,
    load_from_json,
    save_to_csv,
    save_to_json,
)
from cib.visualization import (
    DynamicVisualizer,
    MatrixVisualizer,
    ScenarioVisualizer,
    ShockVisualizer,
    UncertaintyVisualizer,
)
from cib.threshold import ThresholdRule
from cib.network_analysis import (
    ImpactPathwayAnalyzer,
    NetworkAnalyzer,
    NetworkGraphBuilder,
)
from cib.transformation_matrix import (
    PerturbationInfo,
    TransformationMatrix,
    TransformationMatrixBuilder,
)
from cib.attribution import (
    Contribution,
    DescriptorAttribution,
    FlipCandidate,
    ScenarioAttribution,
    attribute_scenario,
    flip_candidates_for_descriptor,
)
from cib.rare_events import (
    BinomialInterval,
    EventRateDiagnostics,
    event_rate_diagnostics,
    min_switch_margin,
    near_miss_rate,
    wilson_interval_from_count,
)
from cib.sensitivity import (
    DriverSpec,
    GlobalSensitivityReport,
    ImportanceSummary,
    OutcomeSensitivity,
    OutcomeSpec,
    compute_global_sensitivity_attractors,
    compute_global_sensitivity_dynamic,
)

__all__ = [
    "CIBMatrix",
    "GaussianCIBMatrix",
    "Scenario",
    "ConsistencyChecker",
    "ImpactBalance",
    "SuccessionOperator",
    "GlobalSuccession",
    "LocalSuccession",
    "AttractorFinder",
    "ScenarioAnalyzer",
    "DynamicCIB",
    "BranchingPathwayBuilder",
    "BranchingResult",
    "ScenarioDiagnostics",
    "scenario_diagnostics",
    "impact_label",
    "judgment_section_labels",
    "TransformationPathway",
    "pathway_frequencies",
    "state_probability_timelines",
    "numeric_quantile_timelines",
    "ThresholdRule",
    "CyclicDescriptor",
    "UncertainCIBMatrix",
    "ConfidenceMapper",
    "MonteCarloAnalyzer",
    "ExpertAggregator",
    "ShockModel",
    "RobustnessTester",
    "MatrixVisualizer",
    "ScenarioVisualizer",
    "UncertaintyVisualizer",
    "ShockVisualizer",
    "DynamicVisualizer",
    "load_from_csv",
    "save_to_csv",
    "load_from_json",
    "save_to_json",
    "NetworkGraphBuilder",
    "NetworkAnalyzer",
    "ImpactPathwayAnalyzer",
    "TransformationMatrix",
    "TransformationMatrixBuilder",
    "PerturbationInfo",
    "reduce_matrix",
    "bin_states",
    "map_scenario_to_reduced",
    "Contribution",
    "DescriptorAttribution",
    "FlipCandidate",
    "ScenarioAttribution",
    "attribute_scenario",
    "flip_candidates_for_descriptor",
    "BinomialInterval",
    "EventRateDiagnostics",
    "wilson_interval_from_count",
    "event_rate_diagnostics",
    "min_switch_margin",
    "near_miss_rate",
    "DriverSpec",
    "OutcomeSpec",
    "ImportanceSummary",
    "OutcomeSensitivity",
    "GlobalSensitivityReport",
    "compute_global_sensitivity_dynamic",
    "compute_global_sensitivity_attractors",
]
