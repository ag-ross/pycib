"""
Attribution and sensitivity utilities for CIB scenarios.

Local explanation utilities are provided:
  - a per-descriptor margin to switching (chosen versus best alternative)
  - a per-source contribution decomposition of that margin
  - simple single-cell “flip candidate” suggestions (bounded perturbations)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Mapping, Optional, Tuple

from cib.core import CIBMatrix, Scenario


@dataclass(frozen=True)
class Contribution:
    """
    One source descriptor's contribution to a target descriptor's margin is stored.
    """

    src_descriptor: str
    src_state: str
    # Contribution to chosen and alternative state scores:
    impact_to_chosen: float
    impact_to_alternative: float
    # Delta = (impact_to_chosen - impact_to_alternative); sums to the margin.
    delta: float


@dataclass(frozen=True)
class FlipCandidate:
    """
    A single-cell perturbation suggestion that could flip a descriptor is stored.
    """

    target_descriptor: str
    src_descriptor: str
    src_state: str
    chosen_state: str
    alternative_state: str
    action: Literal["decrease_chosen_cell", "increase_alternative_cell"]
    required_change: float
    available_change: float
    feasible_under_clip: bool


@dataclass(frozen=True)
class DescriptorAttribution:
    target_descriptor: str
    chosen_state: str
    alternative_state: str
    chosen_score: float
    alternative_score: float
    margin_to_switch: float
    contributions: Tuple[Contribution, ...]

    def top_supporting(self, k: int = 10) -> Tuple[Contribution, ...]:
        k = max(0, int(k))
        items = [c for c in self.contributions if float(c.delta) > 0.0]
        items.sort(key=lambda c: float(c.delta), reverse=True)
        return tuple(items[:k])

    def top_undermining(self, k: int = 10) -> Tuple[Contribution, ...]:
        k = max(0, int(k))
        items = [c for c in self.contributions if float(c.delta) < 0.0]
        items.sort(key=lambda c: float(c.delta))
        return tuple(items[:k])


@dataclass(frozen=True)
class ScenarioAttribution:
    scenario: Scenario
    per_descriptor: Tuple[DescriptorAttribution, ...]
    # The overall scenario "distance to switching" is the minimum margin.
    min_margin_to_switch: float

    def by_descriptor(self) -> Dict[str, DescriptorAttribution]:
        return {a.target_descriptor: a for a in self.per_descriptor}


def _best_alternative_state(
    *,
    descriptor: str,
    chosen_state: str,
    scores: Mapping[str, float],
) -> str:
    best_state: Optional[str] = None
    best_score = float("-inf")
    for s, sc in scores.items():
        if s == chosen_state:
            continue
        if float(sc) > best_score:
            best_score = float(sc)
            best_state = str(s)
    # Degenerate descriptor: no alternative.
    return best_state if best_state is not None else str(chosen_state)


def attribute_scenario(
    scenario: Scenario,
    matrix: CIBMatrix,
    *,
    top_k_sources: Optional[int] = None,
) -> ScenarioAttribution:
    """
    A scenario is attributed under a fixed matrix.

    Per-descriptor margin decomposition against the best alternative state is returned.
    """
    per_desc: List[DescriptorAttribution] = []
    min_margin = float("inf")

    for tgt_desc, tgt_states in matrix.descriptors.items():
        chosen = scenario.get_state(tgt_desc)

        # Compute scores for all candidate states for this target descriptor.
        scores: Dict[str, float] = {
            s: float(matrix.calculate_impact_score(scenario, tgt_desc, s)) for s in tgt_states
        }
        alt = _best_alternative_state(
            descriptor=str(tgt_desc), chosen_state=str(chosen), scores=scores
        )
        chosen_score = float(scores[str(chosen)])
        alt_score = float(scores[str(alt)])
        margin = float(chosen_score - alt_score) if str(alt) != str(chosen) else 0.0

        contribs: List[Contribution] = []
        for src_desc in matrix.descriptors.keys():
            if src_desc == tgt_desc:
                continue
            src_state = scenario.get_state(src_desc)
            ic = float(matrix.get_impact(src_desc, src_state, tgt_desc, chosen))
            ia = float(matrix.get_impact(src_desc, src_state, tgt_desc, alt))
            contribs.append(
                Contribution(
                    src_descriptor=str(src_desc),
                    src_state=str(src_state),
                    impact_to_chosen=float(ic),
                    impact_to_alternative=float(ia),
                    delta=float(ic - ia),
                )
            )

        # Optionally cap contribution list to top-K by |delta| for readability.
        if top_k_sources is not None:
            kk = max(0, int(top_k_sources))
            contribs.sort(key=lambda c: abs(float(c.delta)), reverse=True)
            contribs = contribs[:kk]

        per_desc.append(
            DescriptorAttribution(
                target_descriptor=str(tgt_desc),
                chosen_state=str(chosen),
                alternative_state=str(alt),
                chosen_score=float(chosen_score),
                alternative_score=float(alt_score),
                margin_to_switch=float(margin),
                contributions=tuple(contribs),
            )
        )
        min_margin = min(min_margin, float(margin))

    if min_margin == float("inf"):
        min_margin = 0.0

    return ScenarioAttribution(
        scenario=scenario,
        per_descriptor=tuple(per_desc),
        min_margin_to_switch=float(min_margin),
    )


def flip_candidates_for_descriptor(
    attribution: DescriptorAttribution,
    *,
    clip_lo: float = -3.0,
    clip_hi: float = 3.0,
    eps: float = 1e-12,
    k: int = 10,
) -> Tuple[FlipCandidate, ...]:
    """
    Single-cell perturbations that could flip the target descriptor from the chosen
    state to the alternative state are suggested.

    It is assumed that impacts are clipped to [clip_lo, clip_hi] (as used by the
    uncertainty/shock machinery). A local heuristic is used; a global minimal-change
    guarantee is not provided.
    """
    clip_lo = float(clip_lo)
    clip_hi = float(clip_hi)
    eps = float(eps)
    k = max(0, int(k))

    # If margin <= 0, a tie or inconsistency is already present.
    margin = float(attribution.margin_to_switch)
    required = max(0.0, margin + eps)

    out: List[FlipCandidate] = []
    for c in attribution.contributions:
        # Option 1: decrease chosen cell (reduces chosen_score).
        avail_dec = float(c.impact_to_chosen) - float(clip_lo)
        feas_dec = avail_dec >= required and required > 0.0
        out.append(
            FlipCandidate(
                target_descriptor=str(attribution.target_descriptor),
                src_descriptor=str(c.src_descriptor),
                src_state=str(c.src_state),
                chosen_state=str(attribution.chosen_state),
                alternative_state=str(attribution.alternative_state),
                action="decrease_chosen_cell",
                required_change=float(required),
                available_change=float(max(0.0, avail_dec)),
                feasible_under_clip=bool(feas_dec),
            )
        )

        # Option 2: increase alternative cell (increases alt_score).
        avail_inc = float(clip_hi) - float(c.impact_to_alternative)
        feas_inc = avail_inc >= required and required > 0.0
        out.append(
            FlipCandidate(
                target_descriptor=str(attribution.target_descriptor),
                src_descriptor=str(c.src_descriptor),
                src_state=str(c.src_state),
                chosen_state=str(attribution.chosen_state),
                alternative_state=str(attribution.alternative_state),
                action="increase_alternative_cell",
                required_change=float(required),
                available_change=float(max(0.0, avail_inc)),
                feasible_under_clip=bool(feas_inc),
            )
        )

    # Rank by feasibility, then by smallest required/available ratio.
    def _rank(fc: FlipCandidate) -> Tuple[int, float, float]:
        feas_rank = 0 if fc.feasible_under_clip else 1
        denom = max(1e-12, float(fc.available_change))
        ratio = float(fc.required_change) / denom
        return (feas_rank, ratio, -float(fc.available_change))

    out.sort(key=_rank)
    return tuple(out[:k])

