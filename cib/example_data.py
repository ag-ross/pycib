"""
Self-contained, workshop-aligned example data for Cross-Impact Balance (CIB).

This module provides:
  - DATASET_B5: Canonical demo dataset (Energy Transition, 5 descriptors × 5 states),
    deterministically expanded from a compact influence spec at import time.
  - DATASET_C10: Workshop-scale example (Energy Transition, 10 descriptors × 3 states),
    aligned with published CIB practice (typically 8-12 descriptors with 2-4 states).
  - Reference seeded generators for:
      - judgment-uncertainty sampling,
      - structural shocks,
      - AR(1) dynamic shocks (macro-style persistence).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

ImpactKey = Tuple[str, str, str, str]  # (src_descriptor, src_state, tgt_descriptor, tgt_state)


# A default set of period labels is provided for multi-period examples.
DEFAULT_PERIODS: Tuple[int, ...] = (2025, 2030, 2035, 2040, 2045)


# ---------------------------
# Confidence and seed policy are defined.
# ---------------------------

CONFIDENCE_TO_SIGMA: Dict[int, float] = {5: 0.2, 4: 0.5, 3: 0.8, 2: 1.2, 1: 1.5}


def sigma_from_confidence(c: int) -> float:
    """Map confidence code (1..5) to sigma."""
    try:
        return CONFIDENCE_TO_SIGMA[int(c)]
    except Exception as e:  # noqa: BLE001 - keep dependency-light
        raise ValueError(
            f"Invalid confidence code {c!r}; expected one of {sorted(CONFIDENCE_TO_SIGMA)}"
        ) from e


def seeds_for_run(base_seed: int, m: int) -> Dict[str, int]:
    """
    Derive non-overlapping seeds for each randomness source.

    This avoids accidental coupling between uncertainty/shock draws.
    """
    base_seed = int(base_seed)
    m = int(m)
    return {
        "judgment_uncertainty_seed": base_seed + 10_000 + m,
        "structural_shock_seed": base_seed + 20_000 + m,
        "dynamic_shock_seed": base_seed + 30_000 + m,
    }


def clip_impact(x: float, lo: float = -3.0, hi: float = +3.0) -> float:
    """Clip to the conventional CIB impact range."""
    return float(np.clip(x, lo, hi))


# -----------------------------------
# Reference stochastic data generators are provided.
# -----------------------------------

def sample_uncertain_cim(
    *,
    impacts: Mapping[ImpactKey, float],
    confidence: Mapping[ImpactKey, int],
    seed: int,
    sigma_scale: float = 1.0,
    clip_lo: float = -3.0,
    clip_hi: float = +3.0,
) -> Dict[ImpactKey, float]:
    """
    Sample a noisy CIM around point impacts using confidence-derived sigmas.

    This is an epistemic uncertainty model: it represents "noise around workshop
    point judgments".
    """
    rng = np.random.default_rng(int(seed))
    sigma_scale = float(sigma_scale)
    if sigma_scale <= 0:
        raise ValueError("sigma_scale must be positive")
    out: Dict[ImpactKey, float] = {}
    for key, mu in impacts.items():
        sigma = sigma_from_confidence(confidence[key]) * sigma_scale
        out[key] = float(
            np.clip(rng.normal(loc=float(mu), scale=float(sigma)), clip_lo, clip_hi)
        )
    return out


def _draw_eps(
    rng: np.random.Generator,
    *,
    dist: str,
    scale: float,
    df: Optional[float] = None,
    jump_prob: float = 0.0,
    jump_scale: Optional[float] = None,
) -> float:
    """
    Draw a single disturbance with optional fat tails / jump process.

    dist:
      - "normal": Gaussian N(0, scale)
      - "student_t": scaled Student-t with df degrees of freedom (fat tails)
      - "normal_jump": mixture of normal + rare jumps

    Jump process (if jump_prob > 0): with probability jump_prob, add an
    independent N(0, jump_scale) jump to the base draw.
    """
    dist = str(dist)
    scale = float(scale)
    if scale < 0:
        raise ValueError("scale must be non-negative")

    if dist == "normal":
        x = float(rng.normal(loc=0.0, scale=scale))
    elif dist == "student_t":
        nu = 5.0 if df is None else float(df)
        if nu <= 2:
            raise ValueError("student_t df must be > 2 for finite variance")
        # Standard t has variance nu/(nu-2). Scaling is performed to have variance = scale^2.
        z = float(rng.standard_t(df=nu))
        x = z * (scale * float(np.sqrt((nu - 2.0) / nu)))
    elif dist == "normal_jump":
        x = float(rng.normal(loc=0.0, scale=scale))
    else:
        raise ValueError(f"Unknown dist {dist!r}")

    jp = float(jump_prob)
    if jp > 0:
        if not (0.0 <= jp <= 1.0):
            raise ValueError("jump_prob must be in [0, 1]")
        js = float(jump_scale if jump_scale is not None else (3.0 * scale))
        if float(rng.random()) < jp:
            x += float(rng.normal(loc=0.0, scale=js))

    return float(x)


def sample_structural_shock(
    *,
    impacts_keys: Iterable[ImpactKey],
    structural_sigma: float,
    seed: int,
    dist: str = "normal",
    df: Optional[float] = None,
    jump_prob: float = 0.0,
    jump_scale: Optional[float] = None,
) -> Dict[ImpactKey, float]:
    """
    Structural shock epsilon for each impact cell (additive on CIM).

    Interpretation: a within-run structural disturbance to the impact relationships.
    """
    rng = np.random.default_rng(int(seed))
    structural_sigma = float(structural_sigma)
    return {
        key: _draw_eps(
            rng,
            dist=dist,
            scale=structural_sigma,
            df=df,
            jump_prob=jump_prob,
            jump_scale=jump_scale,
        )
        for key in impacts_keys
    }


def sample_dynamic_shocks_ar1(
    *,
    descriptors: Mapping[str, Sequence[str]],
    periods: Sequence[int],
    dynamic_tau: float,
    rho: float,
    seed: int,
    innovation_dist: str = "normal",
    innovation_df: Optional[float] = None,
    jump_prob: float = 0.0,
    jump_scale: Optional[float] = None,
) -> Dict[int, Dict[Tuple[str, str], float]]:
    """
    AR(1) dynamic shocks at the (descriptor, candidate_state) level, per period.

    Intended application: perturb impact balances during within-period
    succession:
        theta'[d,state] = theta[d,state] + eta_t[d,state]
    """
    rng = np.random.default_rng(int(seed))
    dynamic_tau = float(dynamic_tau)
    rho = float(rho)
    innovation_sigma = float(np.sqrt(max(0.0, 1.0 - rho**2)) * dynamic_tau)

    shocks: Dict[int, Dict[Tuple[str, str], float]] = {int(t): {} for t in periods}
    eta_prev: Dict[Tuple[str, str], float] = {
        (d, s): 0.0 for d, states in descriptors.items() for s in states
    }

    for t in periods:
        t = int(t)
        for d, states in descriptors.items():
            for s in states:
                u = _draw_eps(
                    rng,
                    dist=innovation_dist if innovation_dist != "normal_jump" else "normal_jump",
                    scale=innovation_sigma,
                    df=innovation_df,
                    jump_prob=jump_prob,
                    jump_scale=jump_scale,
                )
                eta = rho * eta_prev[(d, s)] + u
                shocks[t][(d, s)] = float(eta)
                eta_prev[(d, s)] = float(eta)
    return shocks


def sample_dynamic_shocks_ar1_time_varying(
    *,
    descriptors: Mapping[str, Sequence[str]],
    periods: Sequence[int],
    tau_by_period: Mapping[int, float],
    rho: float,
    seed: int,
    innovation_dist: str = "normal",
    innovation_df: Optional[float] = None,
    jump_prob: float = 0.0,
    jump_scale: Optional[float] = None,
) -> Dict[int, Dict[Tuple[str, str], float]]:
    """
    AR(1) dynamic shocks with a time-varying tau(t).

    This is useful when you want uncertainty to increase toward the longer-term
    future in scenario exercises (a common workshop reporting assumption).

    Semantics:
      eta_t = rho * eta_{t-1} + u_t
      u_t ~ Normal(0, sqrt(1-rho^2) * tau(t))
    """
    rng = np.random.default_rng(int(seed))
    rho = float(rho)
    if rho < -1.0 or rho > 1.0:
        raise ValueError("rho must be in [-1, 1]")

    periods_int = [int(t) for t in periods]
    for t in periods_int:
        if t not in tau_by_period:
            raise ValueError(f"tau_by_period missing entry for period {t}")
        if float(tau_by_period[t]) <= 0:
            raise ValueError("All tau_by_period values must be positive")

    shocks: Dict[int, Dict[Tuple[str, str], float]] = {int(t): {} for t in periods_int}
    eta_prev: Dict[Tuple[str, str], float] = {
        (d, s): 0.0 for d, states in descriptors.items() for s in states
    }

    for t in periods_int:
        tau_t = float(tau_by_period[t])
        innovation_sigma = float(np.sqrt(max(0.0, 1.0 - rho**2)) * tau_t)
        for d, states in descriptors.items():
            for s in states:
                u = _draw_eps(
                    rng,
                    dist=innovation_dist if innovation_dist != "normal_jump" else "normal_jump",
                    scale=innovation_sigma,
                    df=innovation_df,
                    jump_prob=jump_prob,
                    jump_scale=jump_scale,
                )
                eta = rho * eta_prev[(d, s)] + u
                shocks[t][(d, s)] = float(eta)
                eta_prev[(d, s)] = float(eta)

    return shocks


def _vec_for_source_state_ordered(
    *,
    source_state: str,
    source_states: Sequence[str],
    target_states: Sequence[str],
    direction: str,
    strength: int,
) -> List[int]:
    """
    Produce an integer impact vector aligned to an ordered target descriptor.

    This generalizes `_vec_for_source_state_3way` to N-way ordered descriptors.
    Semantics: "up" means positive association (low->low, high->high), "down" means
    negative association (low->high, high->low).

    The output is centered (sums approximately to 0) and scaled so that the maximum
    absolute value is approximately `strength` (clipped to [0..2] for consistency
    with the compact workshop spec).
    """
    s = max(0, min(int(strength), 2))
    n_src = len(source_states)
    n_tgt = len(target_states)
    if s == 0:
        return [0 for _ in range(n_tgt)]
    if n_src < 2 or n_tgt < 2:
        raise ValueError("source_states and target_states must have at least 2 states")

    idx_src = list(source_states).index(source_state)
    # Source position is mapped into target index space.
    mapped = int(round(idx_src * (n_tgt - 1) / (n_src - 1)))
    if direction == "down":
        mapped = (n_tgt - 1) - mapped
    elif direction != "up":
        raise ValueError(f"Unexpected direction {direction!r}; expected 'up' or 'down'.")

    raw = [-abs(j - mapped) for j in range(n_tgt)]
    mean = sum(raw) / float(n_tgt)
    centered = [r - mean for r in raw]
    max_abs = max(abs(x) for x in centered) or 1.0
    scaled = [(x / max_abs) * float(s) for x in centered]
    return [int(round(v)) for v in scaled]


def expand_to_full_table(
    *,
    descriptors: Dict[str, List[str]],
    influence_spec: Sequence[Tuple[str, str, str, int, int]],
    default_impact: int = 0,
    default_confidence: int = 3,
) -> Tuple[Dict[ImpactKey, int], Dict[ImpactKey, int]]:
    """
    Deterministically expand a compact influence spec to a full off-diagonal table.

    Returns:
        (impacts, confidence) dictionaries containing all keys where src != tgt.
    """
    impacts: Dict[ImpactKey, int] = {}
    confidence: Dict[ImpactKey, int] = {}

    for (src, tgt, direction, strength, conf) in influence_spec:
        for src_state in descriptors[src]:
            vec = _vec_for_source_state_ordered(
                source_state=src_state,
                source_states=descriptors[src],
                target_states=descriptors[tgt],
                direction=direction,
                strength=strength,
            )
            for tgt_state, val in zip(descriptors[tgt], vec):
                k = (src, src_state, tgt, tgt_state)
                impacts[k] = int(val)
                confidence[k] = int(conf)

    for src in descriptors:
        for tgt in descriptors:
            if src == tgt:
                continue
            for src_state in descriptors[src]:
                for tgt_state in descriptors[tgt]:
                    k = (src, src_state, tgt, tgt_state)
                    impacts.setdefault(k, int(default_impact))
                    confidence.setdefault(k, int(default_confidence))

    return impacts, confidence


# -----------------------------------------
# Dataset B (5-state variant for dynamics)
# -----------------------------------------
#
# Rationale: 3-state systems often produce flat medians and visually "jumpy" paths.
# A 5-state ordered variant is better for demonstrating gradual drift
# (workshop structure) with occasional shock-driven switching or tipping.

DATASET_B5_DESCRIPTORS: Dict[str, List[str]] = {
    "Policy_Stringency": ["Very Low", "Low", "Medium", "High", "Very High"],
    "Fossil_Price_Level": ["Very Low", "Low", "Medium", "High", "Very High"],
    "Renewables_Deployment": ["Very Slow", "Slow", "Moderate", "Fast", "Very Fast"],
    "Grid_Flexibility": ["Very Low", "Low", "Medium", "High", "Very High"],
    "Electrification_Demand": ["Very Low", "Low", "Medium", "High", "Very High"],
}

# Compact workshop relationships (direction/strength/confidence) are adjusted slightly
# to better demonstrate variability and long-horizon uncertainty in a 5-state setting.
#
# Key intent is described as follows:
# - A broadly "linear" drift story is kept from workshop structure (policy/renewables/grid reinforce electrification).
# - Epistemic uncertainty is increased (lower confidence) for some links that are often contested in workshops.
# - A couple of weak feedback links are added to make regime-switching or tipping plausible when near ties.
DATASET_B5_INFLUENCE_SPEC: List[Tuple[str, str, str, int, int]] = [
    # Policy impacts are defined.
    ("Policy_Stringency", "Renewables_Deployment", "up", 2, 4),
    ("Policy_Stringency", "Grid_Flexibility", "up", 2, 3),
    ("Policy_Stringency", "Electrification_Demand", "up", 2, 1),  # uncertain magnitude in practice
    ("Policy_Stringency", "Fossil_Price_Level", "down", 1, 2),  # long-run demand/price dampening

    # Fossil price impacts are defined.
    ("Fossil_Price_Level", "Renewables_Deployment", "up", 1, 3),
    ("Fossil_Price_Level", "Electrification_Demand", "up", 2, 3),
    ("Fossil_Price_Level", "Policy_Stringency", "up", 1, 2),  # price shocks can tighten policy

    # Renewables build-out impacts are defined.
    ("Renewables_Deployment", "Grid_Flexibility", "up", 2, 4),
    ("Renewables_Deployment", "Fossil_Price_Level", "down", 1, 2),
    ("Renewables_Deployment", "Electrification_Demand", "up", 2, 2),

    # Grid flexibility impacts are defined.
    ("Grid_Flexibility", "Renewables_Deployment", "up", 1, 3),
    ("Grid_Flexibility", "Electrification_Demand", "up", 2, 2),

    # Electrification demand impacts are defined.
    ("Electrification_Demand", "Grid_Flexibility", "up", 2, 4),
    ("Electrification_Demand", "Policy_Stringency", "up", 1, 2),
]

DATASET_B5_IMPACTS, DATASET_B5_CONFIDENCE = expand_to_full_table(
    descriptors=DATASET_B5_DESCRIPTORS,
    influence_spec=DATASET_B5_INFLUENCE_SPEC,
    default_impact=0,
    default_confidence=3,
)

DATASET_B5_INITIAL_SCENARIO: Dict[str, str] = {
    "Policy_Stringency": "Medium",
    "Fossil_Price_Level": "Medium",
    "Renewables_Deployment": "Moderate",
    "Grid_Flexibility": "Medium",
    "Electrification_Demand": "Medium",
}

DATASET_B5_THRESHOLD_RULE = {
    "name": "Fast_Permitting_Regime_5State",
    "condition": lambda z: (
        # A slightly easier trigger than the 3-state version is used so the regime switch is visible
        # in a 5-state demo (still requires at least Medium policy and decent grid).
        z.get("Policy_Stringency") in ("Medium", "High", "Very High")
        and z.get("Grid_Flexibility") in ("Medium", "High", "Very High")
    ),
    "cim_modifier": "boost_policy_to_renewables",
}

# An optional numeric mapping is provided for fan charts and expected values (explicit and linear).
DATASET_B5_NUMERIC_MAPPING = {
    "Electrification_Demand": {
        "Very Low": 0.20,
        "Low": 0.35,
        "Medium": 0.50,
        "High": 0.65,
        "Very High": 0.80,
    }
}


def _ordered_markov_transition(
    states: Sequence[str],
    *,
    stay: float,
    step: float,
    step2: float = 0.0,
    up_bias: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """
    Build an inertia transition matrix for ordered states with optional upward bias.

    - Mostly stays put (`stay`)
    - Some probability of moving one step up/down (`step`)
    - Optional tiny probability of moving two steps (`step2`)
    - `up_bias` shifts mass from down to up moves (clipped so probabilities remain valid)
    """
    n = len(states)
    out: Dict[str, Dict[str, float]] = {}
    for i, s in enumerate(states):
        row = {t: 0.0 for t in states}
        row[s] = float(stay)
        if i - 1 >= 0:
            row[states[i - 1]] += float(step)
        if i + 1 < n:
            row[states[i + 1]] += float(step)
        if step2 > 0:
            if i - 2 >= 0:
                row[states[i - 2]] += float(step2)
            if i + 2 < n:
                row[states[i + 2]] += float(step2)

        # Upward bias is applied by shifting probability mass from down to up where possible.
        if up_bias != 0.0:
            if i - 1 >= 0 and i + 1 < n:
                shift = min(abs(float(up_bias)), row[states[i - 1]])
                if up_bias > 0:
                    row[states[i - 1]] -= shift
                    row[states[i + 1]] += shift
                else:
                    shift = min(abs(float(up_bias)), row[states[i + 1]])
                    row[states[i + 1]] -= shift
                    row[states[i - 1]] += shift

        # Renormalization is performed (edges have fewer moves).
        total = sum(row.values())
        out[s] = {k: (v / total) for k, v in row.items()}
    return out


DATASET_B5_FOSSIL_PRICE_TRANSITION = _ordered_markov_transition(
    DATASET_B5_DESCRIPTORS["Fossil_Price_Level"],
    stay=0.60,
    step=0.20,
    step2=0.00,
    up_bias=0.0,
)

# Upward drift is applied for policy (transparent scenario assumption).
DATASET_B5_POLICY_TRANSITION = _ordered_markov_transition(
    DATASET_B5_DESCRIPTORS["Policy_Stringency"],
    stay=0.62,
    step=0.18,
    step2=0.00,
    up_bias=0.12,
)

DATASET_B5_RENEWABLES_TRANSITION = _ordered_markov_transition(
    DATASET_B5_DESCRIPTORS["Renewables_Deployment"],
    stay=0.70,
    step=0.15,
    step2=0.00,
    up_bias=0.08,
)

DATASET_B5_GRID_TRANSITION = _ordered_markov_transition(
    DATASET_B5_DESCRIPTORS["Grid_Flexibility"],
    stay=0.72,
    step=0.14,
    step2=0.00,
    up_bias=0.03,
)

# NOTE: electrification is treated as *endogenous* in the demo (solved by CIB).
# This transition is kept as an optional alternative if electrification is desired to be
# exogenous or inertial instead.
DATASET_B5_ELECTRIFICATION_TRANSITION = _ordered_markov_transition(
    DATASET_B5_DESCRIPTORS["Electrification_Demand"],
    stay=0.78,
    step=0.11,
    step2=0.00,
    up_bias=0.03,
)


def dataset_b5_cyclic_descriptors():
    """
    Cyclic descriptors for Dataset B 5-state variant (inertia + mild drift).
    """
    from cib.cyclic import CyclicDescriptor

    return [
        CyclicDescriptor("Fossil_Price_Level", DATASET_B5_FOSSIL_PRICE_TRANSITION),
        CyclicDescriptor("Policy_Stringency", DATASET_B5_POLICY_TRANSITION),
        CyclicDescriptor("Renewables_Deployment", DATASET_B5_RENEWABLES_TRANSITION),
        CyclicDescriptor("Grid_Flexibility", DATASET_B5_GRID_TRANSITION),
    ]


def dataset_b5_threshold_rule_fast_permitting():
    """
    Construct a ThresholdRule matching DATASET_B5_THRESHOLD_RULE.
    """
    from cib.threshold import ThresholdRule

    return ThresholdRule(
        name=DATASET_B5_THRESHOLD_RULE["name"],
        condition=lambda s: DATASET_B5_THRESHOLD_RULE["condition"](s.to_dict()),
        modifier=dataset_b_modifier_boost_policy_to_renewables,
    )


def dataset_b_modifier_boost_policy_to_renewables(matrix):
    """
    Regime-switch modifier for Dataset B.

    Intuition: under a "fast permitting / strong policy regime", the coupling from
    policy and grid to renewables and electrification strengthens.
    """
    import numpy as _np
    from cib.core import CIBMatrix

    if not isinstance(matrix, CIBMatrix):
        raise ValueError("matrix must be a CIBMatrix")

    out = CIBMatrix(matrix.descriptors)
    out.set_impacts(dict(matrix.iter_impacts()))

    def _bump(src_desc: str, src_state: str, tgt_desc: str, delta: float) -> None:
        for tgt_state in out.descriptors[tgt_desc]:
            base = out.get_impact(src_desc, src_state, tgt_desc, tgt_state)
            states = out.descriptors[tgt_desc]
            hi = states[-1]
            lo = states[0]
            # Mass is smoothly biased upward for N-state targets:
            # - top (and second-top if present) are boosted
            # - bottom (and second-bottom if present) are penalized
            if tgt_state == hi:
                new = base + abs(delta)
            elif len(states) >= 4 and tgt_state == states[-2]:
                new = base + 0.5 * abs(delta)
            elif tgt_state == lo:
                new = base - abs(delta)
            elif len(states) >= 4 and tgt_state == states[1]:
                new = base - 0.5 * abs(delta)
            else:
                new = base
            clipped = float(_np.clip(new, -3.0, 3.0))
            if _np.isclose(clipped, float(base)):
                continue
            out.set_impact(src_desc, src_state, tgt_desc, tgt_state, clipped)

    # Stronger policy and higher grid flexibility push towards faster renewables and higher electrification.
    # The bump is applied at the highest (and, for >=4 states, second-highest) source levels,
    # so this works cleanly for both 3-state and 5-state variants.
    def _apply_at_top(src_desc: str, tgt_desc: str, delta: float) -> None:
        src_states = out.descriptors[src_desc]
        tops = [src_states[-1]]
        if len(src_states) >= 4:
            tops.append(src_states[-2])
        for s in tops:
            _bump(src_desc, s, tgt_desc, delta=delta)

    _apply_at_top("Policy_Stringency", "Renewables_Deployment", delta=2.0)
    _apply_at_top("Policy_Stringency", "Electrification_Demand", delta=2.0)
    _apply_at_top("Grid_Flexibility", "Renewables_Deployment", delta=1.5)
    _apply_at_top("Grid_Flexibility", "Electrification_Demand", delta=1.5)
    _apply_at_top("Renewables_Deployment", "Electrification_Demand", delta=2.0)

    return out


def dataset_b_threshold_rule_fast_permitting():
    """
    Construct a ThresholdRule matching DATASET_B_THRESHOLD_RULE.
    """
    from cib.threshold import ThresholdRule

    raise RuntimeError(
        "DATASET_B_* has been removed. Use dataset_b5_threshold_rule_fast_permitting()."
    )


# -----------------------------------------
# Dataset C10 (10 descriptors × 3 states - workshop-scale example)
# -----------------------------------------
#
# Rationale: A workshop-scale example aligned with published CIB practice
# (typically 8-12 descriptors with 2-4 states) is provided. Scaling to a larger system
# whilst interpretability is maintained is demonstrated.
#
# Total scenario space: 3^10 = 59,049 scenarios (manageable with Monte Carlo methods).

DATASET_C10_DESCRIPTORS: Dict[str, List[str]] = {
    "Policy_Stringency": ["Low", "Medium", "High"],
    "Regulatory_Framework": ["Weak", "Moderate", "Strong"],
    "Renewables_Deployment": ["Slow", "Moderate", "Fast"],
    "Grid_Flexibility": ["Low", "Medium", "High"],
    "Storage_Capacity": ["Limited", "Moderate", "Extensive"],
    "Fossil_Price_Level": ["Low", "Medium", "High"],
    "Investment_Level": ["Low", "Medium", "High"],
    "Technology_Costs": ["High", "Moderate", "Low"],
    "Electrification_Demand": ["Low", "Medium", "High"],
    "Public_Acceptance": ["Low", "Medium", "High"],
}

# Compact workshop relationships (direction/strength/confidence) are defined.
# Not all descriptor pairs have direct relationships (a sparse structure is used).
DATASET_C10_INFLUENCE_SPEC: List[Tuple[str, str, str, int, int]] = [
    # Policy impacts are defined.
    ("Policy_Stringency", "Renewables_Deployment", "up", 2, 4),
    ("Policy_Stringency", "Grid_Flexibility", "up", 2, 3),
    ("Policy_Stringency", "Regulatory_Framework", "up", 2, 4),
    ("Policy_Stringency", "Electrification_Demand", "up", 1, 2),  # uncertain magnitude in practice
    ("Policy_Stringency", "Fossil_Price_Level", "down", 1, 2),  # long-run demand/price dampening

    # Regulatory framework impacts are defined.
    ("Regulatory_Framework", "Renewables_Deployment", "up", 2, 4),
    ("Regulatory_Framework", "Investment_Level", "up", 1, 3),

    # Renewables deployment impacts are defined.
    ("Renewables_Deployment", "Grid_Flexibility", "up", 2, 4),
    ("Renewables_Deployment", "Storage_Capacity", "up", 1, 3),
    ("Renewables_Deployment", "Fossil_Price_Level", "down", 1, 2),
    ("Renewables_Deployment", "Technology_Costs", "down", 1, 3),  # learning curve effects
    ("Renewables_Deployment", "Electrification_Demand", "up", 2, 3),

    # Grid flexibility impacts are defined.
    ("Grid_Flexibility", "Renewables_Deployment", "up", 1, 3),  # feedback
    ("Grid_Flexibility", "Electrification_Demand", "up", 2, 3),
    ("Grid_Flexibility", "Storage_Capacity", "up", 1, 2),

    # Storage capacity impacts are defined.
    ("Storage_Capacity", "Grid_Flexibility", "up", 2, 4),
    ("Storage_Capacity", "Renewables_Deployment", "up", 1, 3),

    # Fossil price impacts are defined.
    ("Fossil_Price_Level", "Renewables_Deployment", "up", 2, 3),
    ("Fossil_Price_Level", "Electrification_Demand", "up", 2, 3),
    ("Fossil_Price_Level", "Policy_Stringency", "up", 1, 2),  # price shocks can tighten policy

    # Investment level impacts are defined.
    ("Investment_Level", "Renewables_Deployment", "up", 2, 4),
    ("Investment_Level", "Grid_Flexibility", "up", 1, 3),
    ("Investment_Level", "Storage_Capacity", "up", 2, 3),
    ("Investment_Level", "Technology_Costs", "down", 1, 2),  # scale effects

    # Technology costs impacts are defined.
    ("Technology_Costs", "Renewables_Deployment", "down", 2, 3),
    ("Technology_Costs", "Investment_Level", "down", 1, 2),

    # Electrification demand impacts are defined.
    ("Electrification_Demand", "Grid_Flexibility", "up", 2, 4),
    ("Electrification_Demand", "Policy_Stringency", "up", 1, 2),
    ("Electrification_Demand", "Renewables_Deployment", "up", 1, 2),

    # Public acceptance impacts are defined.
    ("Public_Acceptance", "Policy_Stringency", "up", 1, 2),
    ("Public_Acceptance", "Renewables_Deployment", "up", 1, 2),
]

DATASET_C10_IMPACTS, DATASET_C10_CONFIDENCE = expand_to_full_table(
    descriptors=DATASET_C10_DESCRIPTORS,
    influence_spec=DATASET_C10_INFLUENCE_SPEC,
    default_impact=0,
    default_confidence=3,
)

DATASET_C10_INITIAL_SCENARIO: Dict[str, str] = {
    "Policy_Stringency": "Medium",
    "Regulatory_Framework": "Moderate",
    "Renewables_Deployment": "Moderate",
    "Grid_Flexibility": "Medium",
    "Storage_Capacity": "Moderate",
    "Fossil_Price_Level": "Medium",
    "Investment_Level": "Medium",
    "Technology_Costs": "Moderate",
    "Electrification_Demand": "Medium",
    "Public_Acceptance": "Medium",
}


# -----------------------------------------
# Dataset C15 (15 descriptors × 4 states - heavier workshop-scale example)
# -----------------------------------------
#
# Rationale: A heavier workshop-scale dataset is provided so that rare events,
# regime switching, and computational scaling can be exercised more visibly.
#
# Total scenario space: 4^15 = 1,073,741,824 scenarios (enumeration is not feasible).

DATASET_C15_DESCRIPTORS: Dict[str, List[str]] = {
    "Policy_Stringency": ["Low", "Medium", "High", "Very High"],
    "Regulatory_Framework": ["Weak", "Moderate", "Strong", "Very Strong"],
    "Renewables_Deployment": ["Slow", "Moderate", "Fast", "Very Fast"],
    "Grid_Flexibility": ["Low", "Medium", "High", "Very High"],
    "Storage_Capacity": ["Limited", "Moderate", "Extensive", "Very Extensive"],
    "Fossil_Price_Level": ["Low", "Medium", "High", "Very High"],
    "Investment_Level": ["Low", "Medium", "High", "Very High"],
    "Technology_Costs": ["High", "Moderate", "Low", "Very Low"],
    "Electrification_Demand": ["Low", "Medium", "High", "Very High"],
    "Public_Acceptance": ["Low", "Medium", "High", "Very High"],
    "Supply_Chain_Constraints": ["Severe", "Moderate", "Mild", "None"],
    "Permitting_Speed": ["Slow", "Moderate", "Fast", "Very Fast"],
    "Industrial_Demand": ["Low", "Medium", "High", "Very High"],
    "Hydrogen_Deployment": ["Low", "Medium", "High", "Very High"],
    "CCS_Deployment": ["Low", "Medium", "High", "Very High"],
}

# A sparse compact influence spec is defined (direction/strength/confidence).
DATASET_C15_INFLUENCE_SPEC: List[Tuple[str, str, str, int, int]] = [
    ("Policy_Stringency", "Regulatory_Framework", "up", 2, 4),
    ("Policy_Stringency", "Permitting_Speed", "up", 2, 3),
    ("Policy_Stringency", "Investment_Level", "up", 2, 3),
    ("Policy_Stringency", "Renewables_Deployment", "up", 2, 4),
    ("Policy_Stringency", "Electrification_Demand", "up", 2, 2),
    ("Policy_Stringency", "Fossil_Price_Level", "down", 1, 2),

    ("Regulatory_Framework", "Renewables_Deployment", "up", 2, 4),
    ("Regulatory_Framework", "Grid_Flexibility", "up", 1, 3),

    ("Permitting_Speed", "Renewables_Deployment", "up", 2, 3),
    ("Permitting_Speed", "Storage_Capacity", "up", 1, 2),

    ("Investment_Level", "Renewables_Deployment", "up", 2, 4),
    ("Investment_Level", "Grid_Flexibility", "up", 2, 3),
    ("Investment_Level", "Storage_Capacity", "up", 2, 3),
    ("Investment_Level", "Technology_Costs", "down", 1, 2),
    ("Investment_Level", "Hydrogen_Deployment", "up", 1, 2),
    ("Investment_Level", "CCS_Deployment", "up", 1, 2),

    ("Technology_Costs", "Renewables_Deployment", "down", 2, 3),
    ("Technology_Costs", "Electrification_Demand", "down", 1, 2),
    ("Technology_Costs", "Hydrogen_Deployment", "down", 1, 2),

    ("Fossil_Price_Level", "Policy_Stringency", "up", 1, 2),
    ("Fossil_Price_Level", "Electrification_Demand", "up", 2, 3),
    ("Fossil_Price_Level", "Renewables_Deployment", "up", 1, 3),

    ("Renewables_Deployment", "Grid_Flexibility", "up", 2, 4),
    ("Renewables_Deployment", "Storage_Capacity", "up", 1, 3),
    ("Renewables_Deployment", "Technology_Costs", "down", 1, 3),
    ("Renewables_Deployment", "Electrification_Demand", "up", 2, 3),

    ("Grid_Flexibility", "Electrification_Demand", "up", 2, 3),
    ("Grid_Flexibility", "Renewables_Deployment", "up", 1, 3),
    ("Grid_Flexibility", "Storage_Capacity", "up", 1, 2),

    ("Storage_Capacity", "Grid_Flexibility", "up", 2, 4),
    ("Storage_Capacity", "Renewables_Deployment", "up", 1, 3),

    ("Supply_Chain_Constraints", "Renewables_Deployment", "down", 2, 2),
    ("Supply_Chain_Constraints", "Storage_Capacity", "down", 1, 2),
    ("Supply_Chain_Constraints", "Technology_Costs", "up", 1, 2),

    ("Public_Acceptance", "Policy_Stringency", "up", 1, 2),
    ("Public_Acceptance", "Permitting_Speed", "up", 1, 2),
    ("Public_Acceptance", "Renewables_Deployment", "up", 1, 2),

    ("Industrial_Demand", "Electrification_Demand", "up", 2, 3),
    ("Hydrogen_Deployment", "Electrification_Demand", "down", 1, 2),
    ("CCS_Deployment", "Policy_Stringency", "down", 1, 2),
]

DATASET_C15_IMPACTS, DATASET_C15_CONFIDENCE = expand_to_full_table(
    descriptors=DATASET_C15_DESCRIPTORS,
    influence_spec=DATASET_C15_INFLUENCE_SPEC,
    default_impact=0,
    default_confidence=3,
)

DATASET_C15_INITIAL_SCENARIO: Dict[str, str] = {
    "Policy_Stringency": "Medium",
    "Regulatory_Framework": "Moderate",
    "Renewables_Deployment": "Moderate",
    "Grid_Flexibility": "Medium",
    "Storage_Capacity": "Moderate",
    "Fossil_Price_Level": "Medium",
    "Investment_Level": "Medium",
    "Technology_Costs": "Moderate",
    "Electrification_Demand": "Medium",
    "Public_Acceptance": "Medium",
    "Supply_Chain_Constraints": "Moderate",
    "Permitting_Speed": "Moderate",
    "Industrial_Demand": "Medium",
    "Hydrogen_Deployment": "Medium",
    "CCS_Deployment": "Medium",
}

DATASET_C15_THRESHOLD_RULE = {
    "name": "Accelerated_Transition_Regime_C15",
    "condition": lambda z: (
        z.get("Policy_Stringency") in ("High", "Very High")
        and z.get("Regulatory_Framework") in ("Strong", "Very Strong")
        and z.get("Permitting_Speed") in ("Fast", "Very Fast")
        and z.get("Investment_Level") in ("High", "Very High")
    ),
    "cim_modifier": "boost_accelerated_transition_c15",
}

DATASET_C15_NUMERIC_MAPPING = {
    "Electrification_Demand": {
        "Low": 0.25,
        "Medium": 0.50,
        "High": 0.70,
        "Very High": 0.85,
    },
}


DATASET_C15_POLICY_TRANSITION = _ordered_markov_transition(
    DATASET_C15_DESCRIPTORS["Policy_Stringency"],
    stay=0.62,
    step=0.16,
    step2=0.02,
    up_bias=0.08,
)
DATASET_C15_FOSSIL_PRICE_TRANSITION = _ordered_markov_transition(
    DATASET_C15_DESCRIPTORS["Fossil_Price_Level"],
    stay=0.58,
    step=0.18,
    step2=0.02,
    up_bias=0.0,
)
DATASET_C15_SUPPLY_CHAIN_TRANSITION = _ordered_markov_transition(
    DATASET_C15_DESCRIPTORS["Supply_Chain_Constraints"],
    stay=0.70,
    step=0.14,
    step2=0.02,
    up_bias=-0.05,
)


def dataset_c15_cyclic_descriptors():
    """
    Cyclic descriptors for Dataset C15 are returned.
    """
    from cib.cyclic import CyclicDescriptor

    return [
        CyclicDescriptor("Policy_Stringency", DATASET_C15_POLICY_TRANSITION),
        CyclicDescriptor("Fossil_Price_Level", DATASET_C15_FOSSIL_PRICE_TRANSITION),
        CyclicDescriptor("Supply_Chain_Constraints", DATASET_C15_SUPPLY_CHAIN_TRANSITION),
    ]


def dataset_c15_threshold_rule_accelerated_transition():
    """
    A ThresholdRule matching DATASET_C15_THRESHOLD_RULE is constructed.
    """
    from cib.threshold import ThresholdRule

    return ThresholdRule(
        name=DATASET_C15_THRESHOLD_RULE["name"],
        condition=lambda s: DATASET_C15_THRESHOLD_RULE["condition"](s.to_dict()),
        modifier=dataset_c15_modifier_boost_accelerated_transition,
    )


def dataset_c15_modifier_boost_accelerated_transition(matrix):
    """
    Regime-switch modifier for Dataset C15.

    Intuition: under an accelerated transition regime, coupling to renewables,
    grid, storage, and electrification is strengthened.
    """
    import numpy as _np
    from cib.core import CIBMatrix

    if not isinstance(matrix, CIBMatrix):
        raise ValueError("matrix must be a CIBMatrix")

    out = CIBMatrix(matrix.descriptors)
    out.set_impacts(dict(matrix.iter_impacts()))

    def _bump(src_desc: str, src_state: str, tgt_desc: str, delta: float) -> None:
        for tgt_state in out.descriptors[tgt_desc]:
            base = out.get_impact(src_desc, src_state, tgt_desc, tgt_state)
            states = out.descriptors[tgt_desc]
            hi = states[-1]
            lo = states[0]
            if tgt_state == hi:
                new = base + abs(delta)
            elif len(states) >= 4 and tgt_state == states[-2]:
                new = base + 0.5 * abs(delta)
            elif tgt_state == lo:
                new = base - abs(delta)
            elif len(states) >= 4 and tgt_state == states[1]:
                new = base - 0.5 * abs(delta)
            else:
                new = base
            clipped = float(_np.clip(new, -3.0, 3.0))
            if _np.isclose(clipped, float(base)):
                continue
            out.set_impact(src_desc, src_state, tgt_desc, tgt_state, clipped)

    def _apply_at_top(src_desc: str, tgt_desc: str, delta: float) -> None:
        src_states = out.descriptors[src_desc]
        tops = [src_states[-1]]
        if len(src_states) >= 4:
            tops.append(src_states[-2])
        for s in tops:
            _bump(src_desc, s, tgt_desc, delta=delta)

    _apply_at_top("Policy_Stringency", "Renewables_Deployment", delta=2.0)
    _apply_at_top("Policy_Stringency", "Permitting_Speed", delta=1.0)
    _apply_at_top("Regulatory_Framework", "Renewables_Deployment", delta=1.5)
    _apply_at_top("Investment_Level", "Renewables_Deployment", delta=2.0)
    _apply_at_top("Investment_Level", "Grid_Flexibility", delta=1.5)
    _apply_at_top("Investment_Level", "Storage_Capacity", delta=1.5)
    _apply_at_top("Renewables_Deployment", "Electrification_Demand", delta=2.0)
    _apply_at_top("Grid_Flexibility", "Electrification_Demand", delta=1.5)

    return out

DATASET_C10_THRESHOLD_RULE = {
    "name": "Accelerated_Transition_Regime",
    "condition": lambda z: (
        # An accelerated transition regime is triggered when strong policy, regulatory support,
        # and investment are present (still requires High policy and at least Moderate regulatory).
        z.get("Policy_Stringency") == "High"
        and z.get("Regulatory_Framework") in ("Moderate", "Strong")
        and z.get("Investment_Level") in ("Medium", "High")
    ),
    "cim_modifier": "boost_accelerated_transition",
}

# An optional numeric mapping is provided for fan charts and expected values (explicit and linear).
DATASET_C10_NUMERIC_MAPPING = {
    "Electrification_Demand": {
        "Low": 0.30,
        "Medium": 0.50,
        "High": 0.70,
    },
    "Renewables_Deployment": {
        "Slow": 0.20,
        "Moderate": 0.50,
        "Fast": 0.80,
    },
}


DATASET_C10_POLICY_TRANSITION = _ordered_markov_transition(
    DATASET_C10_DESCRIPTORS["Policy_Stringency"],
    stay=0.65,
    step=0.20,
    step2=0.00,
    up_bias=0.10,
)

DATASET_C10_FOSSIL_PRICE_TRANSITION = _ordered_markov_transition(
    DATASET_C10_DESCRIPTORS["Fossil_Price_Level"],
    stay=0.60,
    step=0.25,
    step2=0.00,
    up_bias=0.0,
)

DATASET_C10_RENEWABLES_TRANSITION = _ordered_markov_transition(
    DATASET_C10_DESCRIPTORS["Renewables_Deployment"],
    stay=0.70,
    step=0.20,
    step2=0.00,
    up_bias=0.08,
)

DATASET_C10_INVESTMENT_TRANSITION = _ordered_markov_transition(
    DATASET_C10_DESCRIPTORS["Investment_Level"],
    stay=0.68,
    step=0.22,
    step2=0.00,
    up_bias=0.05,
)


def dataset_c10_cyclic_descriptors():
    """
    Cyclic descriptors for Dataset C10 (inertia + mild drift) are returned.
    """
    from cib.cyclic import CyclicDescriptor

    return [
        CyclicDescriptor("Policy_Stringency", DATASET_C10_POLICY_TRANSITION),
        CyclicDescriptor("Fossil_Price_Level", DATASET_C10_FOSSIL_PRICE_TRANSITION),
        CyclicDescriptor("Renewables_Deployment", DATASET_C10_RENEWABLES_TRANSITION),
        CyclicDescriptor("Investment_Level", DATASET_C10_INVESTMENT_TRANSITION),
    ]


def dataset_c10_threshold_rule_accelerated_transition():
    """
    A ThresholdRule matching DATASET_C10_THRESHOLD_RULE is constructed.
    """
    from cib.threshold import ThresholdRule

    return ThresholdRule(
        name=DATASET_C10_THRESHOLD_RULE["name"],
        condition=lambda s: DATASET_C10_THRESHOLD_RULE["condition"](s.to_dict()),
        modifier=dataset_c10_modifier_boost_accelerated_transition,
    )


def dataset_c10_modifier_boost_accelerated_transition(matrix):
    """
    Regime-switch modifier for Dataset C10.

    Intuition: under an "accelerated transition regime", the coupling from
    policy, investment, and renewables to grid, storage, and electrification strengthens.
    """
    import numpy as _np
    from cib.core import CIBMatrix

    if not isinstance(matrix, CIBMatrix):
        raise ValueError("matrix must be a CIBMatrix")

    out = CIBMatrix(matrix.descriptors)
    out.set_impacts(dict(matrix.iter_impacts()))

    def _bump(src_desc: str, src_state: str, tgt_desc: str, delta: float) -> None:
        for tgt_state in out.descriptors[tgt_desc]:
            base = out.get_impact(src_desc, src_state, tgt_desc, tgt_state)
            states = out.descriptors[tgt_desc]
            hi = states[-1]
            lo = states[0]
            # Mass is smoothly biased upward for 3-state targets:
            # - top state is boosted
            # - bottom state is penalised
            if tgt_state == hi:
                new = base + abs(delta)
            elif tgt_state == lo:
                new = base - abs(delta)
            else:
                new = base
            clipped = float(_np.clip(new, -3.0, 3.0))
            if _np.isclose(clipped, float(base)):
                continue
            out.set_impact(src_desc, src_state, tgt_desc, tgt_state, clipped)

    # Stronger policy, investment, and renewables push towards faster grid, storage, and electrification.
    # The bump is applied at the highest source levels for 3-state variants.
    def _apply_at_top(src_desc: str, tgt_desc: str, delta: float) -> None:
        src_states = out.descriptors[src_desc]
        top = src_states[-1]
        _bump(src_desc, top, tgt_desc, delta=delta)

    _apply_at_top("Policy_Stringency", "Renewables_Deployment", delta=1.5)
    _apply_at_top("Policy_Stringency", "Grid_Flexibility", delta=1.5)
    _apply_at_top("Policy_Stringency", "Storage_Capacity", delta=1.0)
    _apply_at_top("Investment_Level", "Renewables_Deployment", delta=1.5)
    _apply_at_top("Investment_Level", "Grid_Flexibility", delta=1.0)
    _apply_at_top("Investment_Level", "Storage_Capacity", delta=1.5)
    _apply_at_top("Renewables_Deployment", "Grid_Flexibility", delta=1.0)
    _apply_at_top("Renewables_Deployment", "Storage_Capacity", delta=1.0)
    _apply_at_top("Renewables_Deployment", "Electrification_Demand", delta=1.5)

    return out


def _save_cim_to_text_file(
    descriptors: Dict[str, List[str]],
    impacts: Dict[ImpactKey, int],
    confidence: Dict[ImpactKey, int],
    dataset_name: str,
    output_path: str,
) -> None:
    """
    Save CIM in standard CIB text format.

    Generates a text file containing the complete Cross-Impact Matrix
    with all impacts and confidence codes.
    """
    from pathlib import Path
    
    # Descriptor names are obtained in order.
    descriptor_names = list(descriptors.keys())
    
    # Output is formatted as standard CIB text matrix.
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("CROSS-IMPACT MATRIX (CIM)")
    output_lines.append(f"Dataset: {dataset_name}")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Descriptors and states are written.
    output_lines.append("DESCRIPTORS AND STATES:")
    output_lines.append("-" * 80)
    for desc_name, states in descriptors.items():
        states_str = " | ".join(states)
        output_lines.append(f"  {desc_name:25s}: {states_str}")
    output_lines.append("")
    output_lines.append("")
    
    # Impact matrix sections are written.
    output_lines.append("=" * 80)
    output_lines.append("IMPACT MATRIX SECTIONS")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("Format: Source_Descriptor[Source_State] -> Target_Descriptor[Target_State] = Impact(Confidence)")
    output_lines.append("")
    
    # Impacts are grouped by source descriptor.
    for src_desc in descriptor_names:
        output_lines.append("-" * 80)
        output_lines.append(f"SOURCE: {src_desc}")
        output_lines.append("-" * 80)
        
        # Grouping is performed by target descriptor.
        for tgt_desc in descriptor_names:
            if src_desc == tgt_desc:
                continue  # Skip diagonal
            
            output_lines.append(f"  -> TARGET: {tgt_desc}")
            
            # All states for source and target are obtained.
            src_states = descriptors[src_desc]
            tgt_states = descriptors[tgt_desc]
            
            # A matrix table is created.
            # Header row is formatted.
            header = " " * 30
            for tgt_state in tgt_states:
                header += f"{tgt_state:>12s}"
            output_lines.append(header)
            
            # Data rows are formatted.
            for src_state in src_states:
                row = f"  {src_desc}[{src_state:15s}]"
                for tgt_state in tgt_states:
                    key = (src_desc, src_state, tgt_desc, tgt_state)
                    impact = impacts.get(key, 0)
                    conf = confidence.get(key, 3)
                    # Formatting: impact (confidence).
                    cell = f"{impact:>4.0f}({conf})"
                    row += cell
                output_lines.append(row)
            
            output_lines.append("")
    
    # Confidence legend is added.
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("CONFIDENCE CODES")
    output_lines.append("=" * 80)
    output_lines.append("  5 = Very High Confidence (sigma = 0.2)")
    output_lines.append("  4 = High Confidence (sigma = 0.5)")
    output_lines.append("  3 = Medium Confidence (sigma = 0.8)")
    output_lines.append("  2 = Low Confidence (sigma = 1.2)")
    output_lines.append("  1 = Very Low Confidence (sigma = 1.5)")
    output_lines.append("")
    output_lines.append("Impact values range from -3 (strongly hindering) to +3 (strongly promoting)")
    output_lines.append("")
    
    # Output directory is determined (package root / results).
    try:
        # An attempt is made to get the package directory.
        import cib.example_data as mod
        package_dir = Path(mod.__file__).parent.parent
        results_dir = package_dir / "results"
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / output_path
    except Exception:
        # Fallback: current directory is used.
        output_file = Path(output_path)
    
    # Output is written to file (silently fails if file cannot be written, e.g., read-only filesystem).
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
    except Exception:  # noqa: BLE001 - keep dependency-light
        pass


def _save_scenario_scoring_to_file(
    descriptors: Dict[str, List[str]],
    impacts: Dict[ImpactKey, int],
    initial_scenario: Dict[str, str],
    output_path: str,
) -> None:
    """
    Save initial scenario scoring in standard format.

    Generates a text file containing scenario diagnostics for the initial
    scenario, including consistency, margins, impact balances, and brink descriptors.
    """
    from pathlib import Path

    from cib.core import CIBMatrix, Scenario
    from cib.scoring import scenario_diagnostics

    # Matrix and scenario are created.
    matrix = CIBMatrix(descriptors)
    matrix.set_impacts(impacts)
    scenario = Scenario(initial_scenario, matrix)

    # Diagnostics are computed.
    diag = scenario_diagnostics(scenario, matrix)

    # Output is formatted.
    output_lines = []
    output_lines.append("=" * 60)
    output_lines.append("SCENARIO SCORING OUTPUT")
    output_lines.append("=" * 60)
    output_lines.append("")

    output_lines.append("INITIAL SCENARIO SCORING:")
    output_lines.append(f"  Consistent: {diag.is_consistent}")
    output_lines.append(f"  Consistency Margin: {diag.consistency_margin:.3f}")
    output_lines.append(f"  Total Impact Score: {diag.total_impact_score:.3f}")
    output_lines.append(f"  Brink Descriptors (threshold=0.5): {diag.brink_descriptors(threshold=0.5)}")
    output_lines.append(f"  Chosen States: {diag.chosen_states}")
    output_lines.append("")

    output_lines.append("  Impact Balances:")
    for desc, balances in diag.balances.items():
        output_lines.append(f"    {desc}:")
        for state, score in balances.items():
            marker = " <-- CHOSEN" if state == diag.chosen_states.get(desc) else ""
            output_lines.append(f"      {state}: {score:6.3f}{marker}")
    output_lines.append("")

    # Output directory is determined (package root / results).
    try:
        import cib.example_data as mod
        package_dir = Path(mod.__file__).parent.parent
        results_dir = package_dir / "results"
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / output_path
    except Exception:
        output_file = Path(output_path)

    # Output is written to file (silently fails if file cannot be written, e.g., read-only filesystem).
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
    except Exception:  # noqa: BLE001 - keep dependency-light
        pass


def write_example_dataset_artifacts(*, dataset: str = "B5") -> None:
    """
    Example dataset artefacts are written to `results/`.

    This helper is provided so that import-time side effects are avoided.
    """
    key = str(dataset).strip().upper()
    if key == "B5":
        _save_cim_to_text_file(
            descriptors=DATASET_B5_DESCRIPTORS,
            impacts=DATASET_B5_IMPACTS,
            confidence=DATASET_B5_CONFIDENCE,
            dataset_name="DATASET_B5 (Energy Transition - 5 descriptors × 5 states)",
            output_path="dataset_b5_cim.txt",
        )
        _save_scenario_scoring_to_file(
            descriptors=DATASET_B5_DESCRIPTORS,
            impacts=DATASET_B5_IMPACTS,
            initial_scenario=DATASET_B5_INITIAL_SCENARIO,
            output_path="scenario_scoring_output.txt",
        )
        return

    if key == "C10":
        _save_cim_to_text_file(
            descriptors=DATASET_C10_DESCRIPTORS,
            impacts=DATASET_C10_IMPACTS,
            confidence=DATASET_C10_CONFIDENCE,
            dataset_name="DATASET_C10 (Energy Transition - 10 descriptors × 3 states)",
            output_path="dataset_c10_cim.txt",
        )
        _save_scenario_scoring_to_file(
            descriptors=DATASET_C10_DESCRIPTORS,
            impacts=DATASET_C10_IMPACTS,
            initial_scenario=DATASET_C10_INITIAL_SCENARIO,
            output_path="scenario_scoring_output_c10.txt",
        )
        return

    raise ValueError("dataset must be 'B5' or 'C10'")

