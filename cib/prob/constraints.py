from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

from cib.prob.types import FactorSpec
from cib.prob.fit_report import FeasibilityAdjustment


Marginals = Mapping[str, Mapping[str, float]]
MultiplierKey = Tuple[Tuple[str, str], Tuple[str, str]]  # ((i, a), (j, b))
Multipliers = Mapping[MultiplierKey, float]


def validate_marginals(factors: Iterable[FactorSpec], marginals: Marginals, *, tol: float = 1e-9) -> None:
    """
    Validate that marginals cover all factors/outcomes and sum to 1 per factor.
    """
    tol = float(tol)
    for f in factors:
        if f.name not in marginals:
            raise ValueError(f"Missing marginals for factor {f.name!r}")
        probs = marginals[f.name]
        missing = [o for o in f.outcomes if o not in probs]
        if missing:
            raise ValueError(f"Missing marginal outcomes for factor {f.name!r}: {missing!r}")
        vals = [float(probs[o]) for o in f.outcomes]
        if any(v < -tol for v in vals):
            raise ValueError(f"Negative marginal probability for factor {f.name!r}")
        s = sum(vals)
        if abs(s - 1.0) > tol:
            raise ValueError(f"Marginals for factor {f.name!r} must sum to 1 (got {s})")


def frechet_bounds(pi: float, pj: float) -> Tuple[float, float]:
    """
    Fréchet bounds for a pairwise probability P(A,B) given marginals pi, pj.
    """
    pi = float(pi)
    pj = float(pj)
    lo = max(0.0, pi + pj - 1.0)
    hi = min(pi, pj)
    return float(lo), float(hi)


@dataclass(frozen=True)
class MultiplierFeasibilityIssue:
    i: str
    j: str
    given_outcome: str
    implied_sum: float


def multiplier_normalization_issues(
    factors: Iterable[FactorSpec],
    marginals: Marginals,
    multipliers: Multipliers,
    *,
    tol: float = 1e-6,
) -> list[MultiplierFeasibilityIssue]:
    """
    Check the implied normalization constraint:

      For fixed (i, j=b): sum_a m_{(i=a)<-(j=b)} * P(i=a) = 1

    Only checks contexts where multipliers are provided for *all* outcomes of i.
    """
    tol = float(tol)
    outcomes_by_factor = {f.name: list(f.outcomes) for f in factors}
    issues: list[MultiplierFeasibilityIssue] = []

    # Keys are grouped by (i, j, b) for processing.
    grouped: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for (i_a, j_b), m in multipliers.items():
        (i, a) = i_a
        (j, b) = j_b
        grouped.setdefault((i, j, b), {})[a] = float(m)

    for (i, j, b), by_a in grouped.items():
        if i not in outcomes_by_factor or j not in outcomes_by_factor:
            continue
        needed = outcomes_by_factor[i]
        if any(a not in by_a for a in needed):
            continue
        implied = 0.0
        for a in needed:
            implied += float(by_a[a]) * float(marginals[i][a])
        if abs(float(implied) - 1.0) > tol:
            issues.append(
                MultiplierFeasibilityIssue(
                    i=str(i),
                    j=str(j),
                    given_outcome=str(b),
                    implied_sum=float(implied),
                )
            )
    return issues


def multiplier_pairwise_targets(
    marginals: Marginals,
    multipliers: Multipliers,
) -> Dict[Tuple[str, str, str, str], float]:
    """
    Convert multipliers into pairwise-marginal targets using fixed marginals:

      P(i=a, j=b) = m_{(i=a)<-(j=b)} * P(i=a) * P(j=b)
    """
    targets: Dict[Tuple[str, str, str, str], float] = {}
    for (i_a, j_b), m in multipliers.items():
        (i, a) = i_a
        (j, b) = j_b
        pi = float(marginals[i][a])
        pj = float(marginals[j][b])
        targets[(i, a, j, b)] = float(m) * pi * pj
    return targets


def pairwise_target_frechet_violations(
    marginals: Marginals,
    targets: Mapping[Tuple[str, str, str, str], float],
    *,
    tol: float = 1e-9,
) -> Dict[Tuple[str, str, str, str], Tuple[float, float, float]]:
    """
    Return any targets that violate Fréchet bounds by more than tol.
    """
    tol = float(tol)
    out: Dict[Tuple[str, str, str, str], Tuple[float, float, float]] = {}
    for (i, a, j, b), pij in targets.items():
        pi = float(marginals[i][a])
        pj = float(marginals[j][b])
        lo, hi = frechet_bounds(pi, pj)
        v = float(pij)
        if v < lo - tol or v > hi + tol:
            out[(i, a, j, b)] = (float(v), float(lo), float(hi))
    return out


def project_pairwise_targets_to_frechet_bounds(
    marginals: Marginals,
    targets: Mapping[Tuple[str, str, str, str], float],
    *,
    tol: float = 1e-9,
) -> Tuple[Dict[Tuple[str, str, str, str], float], Tuple[FeasibilityAdjustment, ...]]:
    """
    Project pairwise targets into Fréchet-feasible intervals.

    This helper is intended for a "repair" feasibility mode, where incoherent
    multiplier-implied targets are adjusted (clipped) rather than rejected.

    Adjustments are returned so that the applied changes can be audited.
    """
    tol = float(tol)
    adjusted: Dict[Tuple[str, str, str, str], float] = {}
    adjustments: list[FeasibilityAdjustment] = []

    for (i, a, j, b), pij in targets.items():
        pi = float(marginals[i][a])
        pj = float(marginals[j][b])
        lo, hi = frechet_bounds(pi, pj)
        v = float(pij)
        if v < lo - tol:
            v_adj = float(lo)
            adjustments.append(
                FeasibilityAdjustment(
                    i=str(i),
                    a=str(a),
                    j=str(j),
                    b=str(b),
                    original_value=float(v),
                    adjusted_value=float(v_adj),
                    frechet_lower=float(lo),
                    frechet_upper=float(hi),
                )
            )
            adjusted[(str(i), str(a), str(j), str(b))] = float(v_adj)
        elif v > hi + tol:
            v_adj = float(hi)
            adjustments.append(
                FeasibilityAdjustment(
                    i=str(i),
                    a=str(a),
                    j=str(j),
                    b=str(b),
                    original_value=float(v),
                    adjusted_value=float(v_adj),
                    frechet_lower=float(lo),
                    frechet_upper=float(hi),
                )
            )
            adjusted[(str(i), str(a), str(j), str(b))] = float(v_adj)
        else:
            adjusted[(str(i), str(a), str(j), str(b))] = float(v)

    return adjusted, tuple(adjustments)

