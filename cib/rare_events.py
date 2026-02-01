"""
Rare-event reliability utilities.

The following items are provided:
  - binomial Wilson confidence intervals for event rates
  - simple undersampling warnings based on expected hits n*p
  - “near-miss” rates using CIB switch margins (chosen versus best alternative)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from cib.core import CIBMatrix, Scenario


@dataclass(frozen=True)
class BinomialInterval:
    level: float
    lower: float
    upper: float


@dataclass(frozen=True)
class EventRateDiagnostics:
    n: int
    k: int
    p_hat: float
    interval: BinomialInterval
    expected_hits: float
    is_under_sampled: bool


def wilson_interval_from_count(k: int, n: int, *, level: float = 0.95) -> BinomialInterval:
    """
    Wilson score interval for a binomial proportion given success count k.
    """
    n = int(n)
    k = int(k)
    level = float(level)
    if n <= 0:
        raise ValueError("n must be positive")
    if k < 0 or k > n:
        raise ValueError("k must be in [0, n]")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be between 0 and 1")

    # SciPy preferred; fallback supports common levels.
    try:
        from scipy.stats import norm  # type: ignore

        z = float(norm.ppf(0.5 + level / 2.0))
    except Exception:
        if np.isclose(level, 0.95):
            z = 1.96
        elif np.isclose(level, 0.99):
            z = 2.576
        elif np.isclose(level, 0.90):
            z = 1.645
        else:
            z = 1.96

    p = k / n
    denominator = 1.0 + (z**2 / n)
    center = (p + (z**2 / (2.0 * n))) / denominator
    margin = (z / denominator) * np.sqrt((p * (1.0 - p) / n) + (z**2 / (4.0 * n**2)))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return BinomialInterval(level=level, lower=float(lower), upper=float(upper))


def event_rate_diagnostics(
    *,
    k: int,
    n: int,
    level: float = 0.95,
    min_expected_hits: float = 20.0,
) -> EventRateDiagnostics:
    """
    An event-rate estimate, confidence interval, and simple undersampling flag are computed.

    The undersampling heuristic uses expected hits = n * p_hat.
    """
    n = int(n)
    k = int(k)
    if n <= 0:
        raise ValueError("n must be positive")
    if k < 0 or k > n:
        raise ValueError("k must be in [0, n]")
    p_hat = float(k) / float(n)
    interval = wilson_interval_from_count(k, n, level=float(level))
    expected = float(n) * float(p_hat)
    under = expected < float(min_expected_hits)
    return EventRateDiagnostics(
        n=n,
        k=k,
        p_hat=float(p_hat),
        interval=interval,
        expected_hits=float(expected),
        is_under_sampled=bool(under),
    )


def min_switch_margin(scenario: Scenario, matrix: CIBMatrix) -> float:
    """
    Minimum over descriptors of (chosen_score - best_alternative_score).
    """
    m = float("inf")
    for d, states in matrix.descriptors.items():
        chosen = scenario.get_state(d)
        chosen_score = float(matrix.calculate_impact_score(scenario, d, chosen))
        best_alt = float("-inf")
        for s in states:
            if s == chosen:
                continue
            best_alt = max(best_alt, float(matrix.calculate_impact_score(scenario, d, s)))
        if best_alt == float("-inf"):
            margin = 0.0
        else:
            margin = float(chosen_score - best_alt)
        m = min(m, float(margin))
    return 0.0 if m == float("inf") else float(m)


def near_miss_rate(
    scenarios: Sequence[Scenario],
    matrix: CIBMatrix,
    *,
    epsilon: float = 0.25,
) -> float:
    """
    Fraction of scenarios whose min switch margin is <= epsilon.

    Interpretation: how often the system is close to switching under the given matrix,
    with margins used as a proxy.
    """
    if not scenarios:
        raise ValueError("scenarios cannot be empty")
    eps = float(epsilon)
    hits = 0
    for s in scenarios:
        if float(min_switch_margin(s, matrix)) <= eps:
            hits += 1
    return float(hits) / float(len(scenarios))

