"""
Scenario diagnostics and scoring utilities for CIB.

This module provides a small, explicit “scenario quality” surface:
  - consistency (binary)
  - detailed inconsistency diagnostics
  - margin-to-inconsistency (brink detector)
  - total impact score (sum of chosen-state impact scores)
  - qualitative labels for numeric impacts (hindering/promoting)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from cib.core import CIBMatrix, ConsistencyChecker, ImpactBalance, Scenario


@dataclass(frozen=True)
class ScenarioDiagnostics:
    is_consistent: bool
    chosen_states: Dict[str, str]
    balances: Dict[str, Dict[str, float]]
    inconsistencies: List[Dict[str, object]]
    # Minimum over descriptors of (chosen_score - best_alternative_score).
    # Negative values indicate inconsistency; 0 indicates a tie at the maximum.
    consistency_margin: float
    # Sum of impact scores of chosen states across descriptors.
    total_impact_score: float

    def brink_descriptors(self, threshold: float = 0.0) -> List[str]:
        """
        Descriptors at or below the margin threshold are returned.

        This is useful for identifying descriptors that are on the brink of switching.
        """
        threshold = float(threshold)
        brink: List[str] = []
        for d, bal in self.balances.items():
            chosen_state = self.chosen_states.get(d)
            if chosen_state is None:
                continue
            chosen = bal.get(chosen_state)
            if chosen is None:
                continue
            # The best alternative is used (chosen state is excluded).
            best_alt = float("-inf")
            for s, v in bal.items():
                if str(s) == str(chosen_state):
                    continue
                best_alt = max(best_alt, float(v))
            margin = 0.0 if best_alt == float("-inf") else float(chosen) - float(best_alt)
            if margin <= threshold:
                brink.append(d)
        return brink


def scenario_diagnostics(
    scenario: Scenario, matrix: CIBMatrix
) -> ScenarioDiagnostics:
    """
    Standard CIB diagnostics for a scenario are computed.

    Args:
        scenario: Scenario to analyze.
        matrix: CIB matrix containing impact relationships.

    Returns:
        ScenarioDiagnostics object containing consistency status, impact
        balances, margins, and total impact score.

    Raises:
        ValueError: If scenario or matrix contain invalid descriptor or
            state references.
    """
    detail = ConsistencyChecker.check_consistency_detailed(scenario, matrix)
    balances = detail["balances"]
    inconsistencies = detail["inconsistencies"]
    is_consistent = bool(detail["is_consistent"])

    ib = ImpactBalance(scenario, matrix)
    total = 0.0
    margins: List[float] = []
    chosen_states = {}
    for d, states in matrix.descriptors.items():
        chosen_state = scenario.get_state(d)
        chosen_states[d] = chosen_state
        chosen_score = float(ib.get_score(d, chosen_state))
        total += chosen_score
        # Margin to switching is defined against the best alternative state.
        # This makes the margin strictly positive for strongly-consistent descriptors,
        # 0.0 for ties at the maximum, and negative for inconsistent descriptors.
        best_alt = float("-inf")
        for s in states:
            if s == chosen_state:
                continue
            best_alt = max(best_alt, float(ib.get_score(d, s)))
        if best_alt == float("-inf"):
            # Degenerate descriptor (single state): no alternative exists.
            margins.append(0.0)
        else:
            margins.append(chosen_score - best_alt)

    margin = float(min(margins)) if margins else 0.0
    diag = ScenarioDiagnostics(
        is_consistent=is_consistent,
        chosen_states=chosen_states,
        balances={k: {s: float(v) for s, v in row.items()} for k, row in balances.items()},
        inconsistencies=list(inconsistencies),
        consistency_margin=margin,
        total_impact_score=float(total),
    )
    return diag


def impact_label(
    value: float,
    *,
    weak_threshold: float = 0.5,
    strong_threshold: float = 1.5,
) -> str:
    """
    A numeric impact value is mapped to a qualitative label.

    Args:
        value: Numeric impact value to label.
        weak_threshold: Threshold for weak impact classification.
        strong_threshold: Threshold for strong impact classification.

    Returns:
        Qualitative label: "strongly_hindering", "hindering", "neutral",
        "promoting", or "strongly_promoting".

    Raises:
        ValueError: If weak_threshold <= 0 or strong_threshold <= weak_threshold.
    """
    v = float(value)
    weak = float(weak_threshold)
    strong = float(strong_threshold)
    if strong <= weak or weak <= 0:
        raise ValueError("Require 0 < weak_threshold < strong_threshold")

    if v <= -strong:
        return "strongly_hindering"
    if v <= -weak:
        return "hindering"
    if v < weak:
        return "neutral"
    if v < strong:
        return "promoting"
    return "strongly_promoting"


def judgment_section_labels(
    matrix: CIBMatrix,
    *,
    src_desc: str,
    tgt_desc: str,
    weak_threshold: float = 0.5,
    strong_threshold: float = 1.5,
) -> Dict[Tuple[str, str], str]:
    """
    A judgement section (src_desc -> tgt_desc) is labelled with qualitative labels.

    Args:
        matrix: CIB matrix containing impact relationships.
        src_desc: Source descriptor name.
        tgt_desc: Target descriptor name.
        weak_threshold: Threshold for weak impact classification.
        strong_threshold: Threshold for strong impact classification.

    Returns:
        Dictionary mapping (src_state, tgt_state) tuples to qualitative
        labels.

    Raises:
        ValueError: If src_desc == tgt_desc (diagonal sections omitted), or
            if threshold parameters are invalid.
    """
    if src_desc == tgt_desc:
        raise ValueError("Diagonal sections are omitted by convention")
    out: Dict[Tuple[str, str], str] = {}
    for src_state in matrix.descriptors[src_desc]:
        for tgt_state in matrix.descriptors[tgt_desc]:
            v = matrix.get_impact(src_desc, src_state, tgt_desc, tgt_state)
            out[(src_state, tgt_state)] = impact_label(
                v, weak_threshold=weak_threshold, strong_threshold=strong_threshold
            )
    return out

