"""
Dynamic (multi-period) CIB simulation framework (simulation-first).

This module implements a practical dynamic CIB mode:
  - simulate discrete paths across a small number of periods,
  - optionally sample uncertain CIMs per run (Monte Carlo ensemble),
  - optionally apply threshold-triggered CIM modifiers,
  - optionally evolve cyclic descriptors between periods via transition matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np

from cib.core import CIBMatrix, Scenario
from cib.example_data import seeds_for_run
from cib.succession import GlobalSuccession, SuccessionOperator
from cib.cyclic import CyclicDescriptor
from cib.pathway import TransformationPathway
from cib.threshold import ThresholdRule


class _LockedSuccessionOperator(SuccessionOperator):
    """
    Wrapper that prevents selected descriptors from being updated by succession.

    DynamicCIB uses CyclicDescriptor transitions to evolve some descriptors
    between periods. Those descriptors represent exogenous/inertial dynamics and
    should remain fixed during within-period succession, otherwise the successor
    step will immediately overwrite the cyclic transition.
    """

    def __init__(self, inner: SuccessionOperator, locked: Dict[str, str]) -> None:
        """
        Initialize locked succession operator.

        Args:
            inner: Base succession operator to wrap.
            locked: Dictionary mapping descriptor names to state values that
                should remain fixed during succession.
        """
        self.inner = inner
        self.locked = dict(locked)

    def find_successor(self, scenario: Scenario, matrix: CIBMatrix) -> Scenario:
        """
        Find successor scenario while preserving locked descriptor states.

        Args:
            scenario: Current scenario to find successor for.
            matrix: CIB matrix for computing impacts.

        Returns:
            Successor scenario with locked descriptors preserved at their
            specified values.
        """
        nxt = self.inner.find_successor(scenario, matrix)
        if not self.locked:
            return nxt
        state = nxt.to_dict()
        for d, v in self.locked.items():
            if d in state:
                state[d] = v
        return Scenario(state, matrix)


@dataclass
class DynamicCIB:
    """
    Simulation-first dynamic CIB wrapper.
    """

    base_matrix: CIBMatrix
    periods: List[int]

    def __post_init__(self) -> None:
        """
        Initialize dynamic CIB instance after dataclass creation.

        Raises:
            ValueError: If periods list is empty.
        """
        if not self.periods:
            raise ValueError("periods cannot be empty")
        self.threshold_rules: List[ThresholdRule] = []
        self.cyclic_descriptors: Dict[str, CyclicDescriptor] = {}

    def add_threshold_rule(self, rule: ThresholdRule) -> None:
        """
        Add a threshold rule to modify the CIM conditionally.

        Args:
            rule: Threshold rule to add. Rules are evaluated in order during
                simulation, and the first matching rule's modifier is applied.
        """
        self.threshold_rules.append(rule)

    def add_cyclic_descriptor(self, cyclic: CyclicDescriptor) -> None:
        """
        Add a cyclic descriptor for exogenous/inertial dynamics.

        Args:
            cyclic: Cyclic descriptor defining transition probabilities between
                periods. The descriptor will evolve between periods but remain
                fixed during within-period succession.

        Raises:
            ValueError: If the cyclic descriptor fails validation.
        """
        cyclic.validate()
        self.cyclic_descriptors[cyclic.name] = cyclic

    def _apply_thresholds(self, matrix: CIBMatrix, scenario: Scenario) -> CIBMatrix:
        """
        Apply threshold rules to modify the active CIM based on scenario state.

        Args:
            matrix: Base CIB matrix to potentially modify.
            scenario: Current scenario to evaluate threshold conditions against.

        Returns:
            Modified CIB matrix if any threshold rule matches, otherwise
            returns the original matrix.
        """
        active = matrix
        for rule in self.threshold_rules:
            if rule.condition(scenario):
                active = rule.modifier(active)
        return active

    def _apply_cyclic_transitions(
        self, scenario_dict: Dict[str, str], rng: np.random.Generator
    ) -> Dict[str, str]:
        """
        Apply cyclic descriptor transitions to evolve exogenous variables.

        Args:
            scenario_dict: Current scenario state as dictionary.
            rng: Random number generator for sampling transitions.

        Returns:
            Updated scenario dictionary with cyclic descriptors evolved to
            their next states according to transition probabilities.

        Raises:
            ValueError: If a cyclic descriptor is missing from the scenario.
        """
        out = dict(scenario_dict)
        for name, cyclic in self.cyclic_descriptors.items():
            if name not in out:
                raise ValueError(f"Cyclic descriptor {name!r} missing from scenario")
            out[name] = cyclic.sample_next(out[name], rng)
        return out

    def simulate_path(
        self,
        *,
        initial: Dict[str, str],
        seed: Optional[int] = None,
        succession_operator: Optional[SuccessionOperator] = None,
        dynamic_shocks_by_period: Optional[Dict[int, Dict[tuple[str, str], float]]] = None,
        judgment_sigma_scale_by_period: Optional[Dict[int, float]] = None,
        structural_sigma: Optional[float] = None,
        structural_seed_base: Optional[int] = None,
        max_iterations: int = 1000,
        tie_break: str = "deterministic_first",
        equilibrium_mode: Literal["none", "relax_unshocked"] = "none",
    ) -> TransformationPathway:
        """
        A single discrete pathway across periods is simulated.

        Threshold rules and cyclic descriptors:
            - Cyclic transitions (if configured) are applied at the start of each new period
              (except the first), evolving exogenous/inertial descriptors between periods.
            - Threshold rules are evaluated after any cyclic transitions are applied, using
              the resulting scenario state to determine the active CIM used for within-period
              succession.

        Args:
            initial: Initial scenario as a descriptor -> state mapping.
            seed: Seed used for stochastic elements (cyclic transitions and tie breaks).
            succession_operator: Succession operator used within each period.
            dynamic_shocks_by_period: Optional per-period shock field used for within-period
                succession (score perturbations at the descriptor-state level).
            judgment_sigma_scale_by_period: Optional per-period sigma scale used when the base
                matrix supports `sample_matrix(...)`.
            structural_sigma: Optional structural shock magnitude applied to the per-period
                matrix prior to within-period succession.
            structural_seed_base: Optional base seed used for structural shocks.
            max_iterations: Maximum number of succession iterations per period.
            tie_break: Cycle representative selection policy when a cycle is detected.
            equilibrium_mode: Optional equilibrium output mode. When set to
                `"relax_unshocked"`, an unshocked relaxation is performed after the realised
                attractor is selected, and `equilibrium_scenarios` is populated on the returned
                pathway.

        Returns:
            A `TransformationPathway` containing the realised per-period scenarios, and optionally
            equilibrium scenarios when `equilibrium_mode` is enabled.

        Raises:
            ValueError: If input parameters are invalid.
            RuntimeError: If succession does not converge within `max_iterations`.
        """
        if succession_operator is None:
            succession_operator = GlobalSuccession()
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if tie_break not in {"deterministic_first", "random"}:
            raise ValueError("Unsupported tie_break")
        if equilibrium_mode not in {"none", "relax_unshocked"}:
            raise ValueError("Unsupported equilibrium_mode")

        rng = np.random.default_rng(seed)

        scenarios: List[Scenario] = []
        equilibrium_scenarios: Optional[List[Scenario]] = (
            [] if equilibrium_mode == "relax_unshocked" else None
        )
        current_state = dict(initial)

        for period_idx, t in enumerate(self.periods):
            # Cyclic transitions are applied at the start of each new period (except the first).
            if period_idx > 0 and self.cyclic_descriptors:
                current_state = self._apply_cyclic_transitions(current_state, rng)

            current = Scenario(current_state, self.base_matrix)

            # Optionally, the CIM is re-sampled per period with a time-varying sigma scale.
            matrix_period: CIBMatrix = self.base_matrix
            if judgment_sigma_scale_by_period is not None:
                scale = float(judgment_sigma_scale_by_period.get(int(t), 1.0))
                if hasattr(self.base_matrix, "sample_matrix"):
                    try:
                        matrix_period = self.base_matrix.sample_matrix(  # type: ignore[attr-defined]
                            int(seed or 0) + 1000 * period_idx, sigma_scale=scale
                        )
                    except TypeError:
                        # Backward compatibility: sampling is performed without sigma_scale.
                        matrix_period = self.base_matrix.sample_matrix(  # type: ignore[attr-defined]
                            int(seed or 0) + 1000 * period_idx
                        )

            if structural_sigma is not None:
                from cib.shocks import ShockModel

                if structural_seed_base is None:
                    structural_seed_base = int(seed or 0) + 50_000
                sm = ShockModel(matrix_period)
                sm.add_structural_shocks(sigma=float(structural_sigma))
                matrix_period = sm.sample_shocked_matrix(
                    int(structural_seed_base) + int(period_idx)
                )

            # Threshold rules are evaluated to select the active CIM for this period.
            # Note: for period_idx > 0, cyclic descriptors (if configured) have already
            # advanced at the start of the period, so thresholds “see” the post-cyclic
            # scenario state.
            matrix_t = self._apply_thresholds(matrix_period, current)

            op = succession_operator
            locked: Dict[str, str] = {}
            # If cyclic descriptors are configured, they are treated as exogenous/inertial
            # variables for this period: they evolve via transitions between periods,
            # but remain fixed during within-period succession.
            if self.cyclic_descriptors:
                locked = {
                    name: current_state[name] for name in self.cyclic_descriptors.keys()
                }
                op = _LockedSuccessionOperator(op, locked)
            if dynamic_shocks_by_period is not None and int(t) in dynamic_shocks_by_period:
                from cib.shocks import ShockAwareGlobalSuccession

                op = ShockAwareGlobalSuccession(dynamic_shocks_by_period[int(t)])
                if self.cyclic_descriptors:
                    locked = {
                        name: current_state[name]
                        for name in self.cyclic_descriptors.keys()
                    }
                    op = _LockedSuccessionOperator(op, locked)

            result = op.find_attractor(
                current, matrix_t, max_iterations=max_iterations
            )

            chosen: Scenario
            if result.is_cycle:
                cycle = result.attractor
                assert isinstance(cycle, list)
                if tie_break == "deterministic_first":
                    chosen = cycle[0]
                else:
                    chosen = cycle[int(rng.integers(0, len(cycle)))]
            else:
                attractor = result.attractor
                assert isinstance(attractor, Scenario)
                chosen = attractor

            scenarios.append(chosen)
            current_state = chosen.to_dict()

            if equilibrium_scenarios is not None:
                # An unshocked relaxation is performed to obtain a matrix-consistent
                # equilibrium scenario for the active period matrix.
                eq_op: SuccessionOperator = succession_operator
                if locked:
                    eq_op = _LockedSuccessionOperator(eq_op, locked)
                eq_start = Scenario(chosen.to_dict(), matrix_t)
                eq_result = eq_op.find_attractor(
                    eq_start, matrix_t, max_iterations=max_iterations
                )
                if eq_result.is_cycle:
                    eq_cycle = eq_result.attractor
                    assert isinstance(eq_cycle, list)
                    equilibrium_scenarios.append(eq_cycle[0])
                else:
                    eq_attractor = eq_result.attractor
                    assert isinstance(eq_attractor, Scenario)
                    equilibrium_scenarios.append(eq_attractor)

        eq_out = tuple(equilibrium_scenarios) if equilibrium_scenarios is not None else None
        return TransformationPathway(
            periods=tuple(int(t) for t in self.periods),
            scenarios=tuple(scenarios),
            equilibrium_scenarios=eq_out,
        )

    def simulate_ensemble(
        self,
        *,
        initial: Dict[str, str],
        n_runs: int,
        base_seed: int,
        structural_sigma: Optional[float] = None,
        dynamic_tau: Optional[float] = None,
        dynamic_tau_growth: float = 0.0,
        judgment_sigma_growth: float = 0.0,
        dynamic_rho: float = 0.5,
        succession_operator: Optional[SuccessionOperator] = None,
        max_iterations: int = 1000,
        equilibrium_mode: Literal["none", "relax_unshocked"] = "none",
    ) -> List[TransformationPathway]:
        """
        An ensemble of pathways is simulated with reproducible seeding.

        If the base matrix supports `sample_matrix(seed)`, each run samples a
        static CIM once (judgment uncertainty). Otherwise, runs reuse base_matrix.

        Stochastic dynamic behavior (recommended for workshop realism):
          - If dynamic_tau is provided, each run samples AR(1) dynamic shocks per period and
            applies them during within-period succession.
          - If structural_sigma is provided, each run applies a structural shock to the CIM
            before simulating the path.

        Args:
            initial: Initial scenario as a descriptor -> state mapping.
            n_runs: Number of Monte Carlo runs.
            base_seed: Base seed used to generate per-run seeds.
            structural_sigma: Optional structural shock magnitude applied per run.
            dynamic_tau: Optional long-run scale for AR(1) dynamic shocks.
            dynamic_tau_growth: Non-negative growth used to increase tau over the horizon.
            judgment_sigma_growth: Non-negative growth used to increase judgment sigma scales
                over the horizon when the base matrix supports sampling.
            dynamic_rho: AR(1) persistence parameter in [-1, 1].
            succession_operator: Succession operator used within each period.
            max_iterations: Maximum number of succession iterations per period.
            equilibrium_mode: Optional equilibrium output mode passed through to `simulate_path`.

        Returns:
            A list of `TransformationPathway` objects.

        Raises:
            ValueError: If input parameters are invalid.
            RuntimeError: If succession does not converge within `max_iterations`.
        """
        if n_runs <= 0:
            raise ValueError("n_runs must be positive")
        if structural_sigma is not None and float(structural_sigma) <= 0:
            raise ValueError("structural_sigma must be positive if provided")
        if dynamic_tau is not None and float(dynamic_tau) <= 0:
            raise ValueError("dynamic_tau must be positive if provided")
        if float(dynamic_tau_growth) < 0:
            raise ValueError("dynamic_tau_growth must be non-negative")
        if float(judgment_sigma_growth) < 0:
            raise ValueError("judgment_sigma_growth must be non-negative")
        if succession_operator is None:
            succession_operator = GlobalSuccession()

        pathways: List[TransformationPathway] = []
        for m in range(int(n_runs)):
            seeds = seeds_for_run(int(base_seed), int(m))

            # 1) A per-run CIM is started from (judgment uncertainty if available).
            matrix_run: CIBMatrix
            if hasattr(self.base_matrix, "sample_matrix"):
                if float(judgment_sigma_growth) > 0:
                    # The sampling-capable matrix is kept so resampling can be performed per period with
                    # increasing sigma scales (response-driven widening uncertainty).
                    matrix_run = self.base_matrix
                else:
                    # Sampling is performed once per run at baseline scale.
                    try:
                        matrix_run = self.base_matrix.sample_matrix(  # type: ignore[attr-defined]
                            seeds["judgment_uncertainty_seed"], sigma_scale=1.0
                        )
                    except TypeError:
                        matrix_run = self.base_matrix.sample_matrix(  # type: ignore[attr-defined]
                            seeds["judgment_uncertainty_seed"]
                        )
            else:
                matrix_run = self.base_matrix

            # 2) A structural shock is applied to the CIM (optional).
            if structural_sigma is not None and float(judgment_sigma_growth) <= 0:
                from cib.shocks import ShockModel

                sm = ShockModel(matrix_run)
                sm.add_structural_shocks(sigma=float(structural_sigma))
                matrix_run = sm.sample_shocked_matrix(seeds["structural_shock_seed"])

            # 3) Dynamic shocks are sampled (optional) and applied during succession.
            dynamic_shocks_by_period = None
            if dynamic_tau is not None:
                from cib.shocks import ShockModel

                dm = ShockModel(matrix_run)
                dm.add_dynamic_shocks(
                    periods=self.periods, tau=float(dynamic_tau), rho=float(dynamic_rho)
                )
                if float(dynamic_tau_growth) > 0:
                    # Tau is increased over the horizon: tau(t_i) = tau * (1 + growth * i).
                    tau_by_period = {
                        int(t): float(dynamic_tau) * (1.0 + float(dynamic_tau_growth) * idx)
                        for idx, t in enumerate(self.periods)
                    }
                    dynamic_shocks_by_period = dm.sample_dynamic_shocks_time_varying(
                        seed=seeds["dynamic_shock_seed"], tau_by_period=tau_by_period
                    )
                else:
                    dynamic_shocks_by_period = dm.sample_dynamic_shocks(
                        seeds["dynamic_shock_seed"]
                    )

            dyn = DynamicCIB(matrix_run, periods=list(self.periods))
            dyn.threshold_rules = list(self.threshold_rules)
            dyn.cyclic_descriptors = dict(self.cyclic_descriptors)

            judgment_sigma_scale_by_period = None
            if float(judgment_sigma_growth) > 0:
                judgment_sigma_scale_by_period = {
                    int(t): 1.0 + float(judgment_sigma_growth) * idx
                    for idx, t in enumerate(self.periods)
                }

            p = dyn.simulate_path(
                initial=initial,
                seed=seeds["dynamic_shock_seed"],  # used for cyclic draws/tie breaks
                succession_operator=succession_operator,
                dynamic_shocks_by_period=dynamic_shocks_by_period,
                judgment_sigma_scale_by_period=judgment_sigma_scale_by_period,
                structural_sigma=structural_sigma if float(judgment_sigma_growth) > 0 else None,
                structural_seed_base=seeds["structural_shock_seed"],
                max_iterations=max_iterations,
                equilibrium_mode=equilibrium_mode,
            )
            pathways.append(p)

        return pathways

