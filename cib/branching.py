"""
Hybrid branching pathway construction (enumerate when feasible; sample otherwise).

This module implements a “middle way” between:
  - enumeration-based branching per-period scenario-set enumeration (A/B/C per sub-period), and
  - pure Monte Carlo simulation of single paths.

For each transition between periods, the builder can:
  - enumerate all consistent scenarios for the next period when the scenario space
    is small (guarded by `max_states_to_enumerate`), or
  - approximate the next-period branching distribution by repeated sampling of
    uncertainty/shocks and succession (Monte Carlo).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import warnings
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from cib.core import CIBMatrix, ConsistencyChecker, Scenario
from cib.cyclic import CyclicDescriptor
from cib.example_data import seeds_for_run
from cib.shocks import ShockAwareGlobalSuccession, ShockModel
from cib.succession import GlobalSuccession, SuccessionOperator
from cib.threshold import ThresholdRule


@dataclass(frozen=True)
class BranchingResult:
    """
    Output of a branching pathway build.
    """

    periods: Tuple[int, ...]
    scenarios_by_period: Tuple[Tuple[Scenario, ...], ...]
    # Edges between consecutive periods are stored as index weights.
    # Format is: edges[(period_idx, src_idx)][tgt_idx] = weight.
    edges: Mapping[Tuple[int, int], Mapping[int, float]]
    # Method used for each layer transition: "enumerate" or "sample".
    transition_method: Mapping[int, str]
    # Top-K most likely node-index paths and their weights are stored.
    top_paths: Tuple[Tuple[Tuple[int, ...], float], ...]


class _LockedSuccessionOperator(SuccessionOperator):
    """
    Wrapper that prevents selected descriptors from being updated by succession.
    """

    def __init__(self, inner: SuccessionOperator, locked: Mapping[str, str]) -> None:
        self.inner = inner
        self.locked = dict(locked)

    def find_successor(self, scenario: Scenario, matrix: CIBMatrix) -> Scenario:
        nxt = self.inner.find_successor(scenario, matrix)
        if not self.locked:
            return nxt
        state = nxt.to_dict()
        for d, v in self.locked.items():
            if d in state:
                state[d] = v
        return Scenario(state, matrix)


def _scenario_space_size(descriptors: Mapping[str, Sequence[str]]) -> int:
    n = 1
    for states in descriptors.values():
        n *= int(len(states))
    return int(n)


def _enumerate_consistent_with_locks(
    *,
    matrix: CIBMatrix,
    locked: Mapping[str, str],
) -> List[Scenario]:
    """
    Enumerate consistent scenarios for `matrix`, holding `locked` descriptors fixed.
    """
    var_descs = [d for d in matrix.descriptors.keys() if d not in locked]
    state_lists = [matrix.descriptors[d] for d in var_descs]

    out: List[Scenario] = []
    for comb in product(*state_lists):
        sdict = dict(locked)
        sdict.update(dict(zip(var_descs, comb)))
        s = Scenario(sdict, matrix)
        if ConsistencyChecker.check_consistency(s, matrix):
            out.append(s)
    return out


def _prune_distribution(
    dist: Mapping[Scenario, float],
    *,
    prune_policy: str,
    per_parent_top_k: Optional[int],
    min_edge_weight: Optional[float],
) -> Dict[Scenario, float]:
    """
    An edge-pruning policy is applied to a single parent's outgoing distribution.
    """
    out = {k: float(v) for k, v in dist.items() if float(v) > 0.0}
    if not out:
        return {}

    if prune_policy == "incoming_mass":
        # No per-parent pruning is applied for this policy.
        pass
    elif prune_policy == "per_parent_topk":
        if per_parent_top_k is None or int(per_parent_top_k) <= 0:
            raise ValueError("per_parent_top_k must be positive for prune_policy='per_parent_topk'")
        k = int(per_parent_top_k)
        kept = sorted(
            out.items(),
            key=lambda kv: (float(kv[1]), tuple(kv[0].to_indices())),
            reverse=True,
        )[:k]
        out = {s: w for s, w in kept}
    elif prune_policy == "min_edge_weight":
        if min_edge_weight is None or float(min_edge_weight) < 0.0:
            raise ValueError("min_edge_weight must be non-negative for prune_policy='min_edge_weight'")
        thr = float(min_edge_weight)
        out = {s: w for s, w in out.items() if float(w) >= thr}
    else:
        raise ValueError(
            "prune_policy must be 'incoming_mass', 'per_parent_topk', or 'min_edge_weight'"
        )

    if not out:
        # A fallback is used so that graph connectivity is preserved.
        best = max(dist.items(), key=lambda kv: (float(kv[1]), tuple(kv[0].to_indices())))[0]
        return {best: 1.0}

    s = float(sum(out.values()))
    if s <= 0.0:
        best = max(out.items(), key=lambda kv: (float(kv[1]), tuple(kv[0].to_indices())))[0]
        return {best: 1.0}
    return {k: float(v) / s for k, v in out.items()}


class BranchingPathwayBuilder:
    """
    Build a per-period branching pathway graph.

    Notes:
    - Cyclic descriptors are treated as exogenous/inertial between periods and are
      locked during within-period succession, consistent with `DynamicCIB`.
    - The active CIM for period t+1 is determined by the scenario at the start
      of period t+1 (after cyclic transitions). Threshold rules are evaluated on
      that post-cyclic scenario so that behaviour matches `DynamicCIB.simulate_path()`.

    Enumeration vs sampling:
        The builder automatically chooses between enumeration and sampling modes
        based on scenario-space size (controlled by `max_states_to_enumerate`).

        - Enumeration mode (scenario space size <= max_states_to_enumerate):
          - Enumerates all consistent scenarios for a deterministic base matrix
          - Ignores `judgment_sigma_scale_by_period` and `structural_sigma`
          - Produces complete results for the CIB-consistent scenario set, conditional on the realised cyclic transitions

        - Sampling mode (scenario space size > max_states_to_enumerate):
          - Estimates transition distributions via Monte Carlo sampling
          - Respects `judgment_sigma_scale_by_period` and `structural_sigma`
          - Produces stochastic, approximate results

        Note: if uncertainty parameters are set but enumeration mode is used,
        those parameters are ignored. To ensure uncertainty is applied, decrease
        `max_states_to_enumerate` to force sampling mode.
    """

    def __init__(
        self,
        *,
        base_matrix: CIBMatrix,
        periods: Sequence[int],
        initial: Mapping[str, str],
        cyclic_descriptors: Optional[Sequence[CyclicDescriptor]] = None,
        threshold_rules: Optional[Sequence[ThresholdRule]] = None,
        threshold_match_policy: Literal["first_match", "all_matches"] = "all_matches",
        succession_operator: Optional[SuccessionOperator] = None,
        node_mode: Literal["equilibrium", "realized"] = "equilibrium",
        max_states_to_enumerate: int = 20_000,
        n_transition_samples: int = 200,
        max_nodes_per_period: Optional[int] = None,
        prune_policy: Literal["incoming_mass", "per_parent_topk", "min_edge_weight"] = "incoming_mass",
        per_parent_top_k: Optional[int] = None,
        min_edge_weight: Optional[float] = None,
        base_seed: int = 123,
        # Optional uncertainty/shock settings are provided for sampling transitions.
        structural_sigma: Optional[float] = None,
        judgment_sigma_scale_by_period: Optional[Mapping[int, float]] = None,
        dynamic_tau: Optional[float] = None,
        dynamic_rho: float = 0.6,
        dynamic_innovation_dist: str = "normal",
        dynamic_innovation_df: Optional[float] = None,
        dynamic_jump_prob: float = 0.0,
        dynamic_jump_scale: Optional[float] = None,
    ) -> None:
        """
        A branching pathway builder is initialised.

        Args:
            base_matrix: Base CIB matrix used for transitions.
            periods: Period labels for the pathway graph.
            initial: Initial scenario as a descriptor -> state mapping.
            cyclic_descriptors: Optional cyclic descriptors (treated as exogenous/inertial).
            threshold_rules: Optional threshold rules applied between periods.
            threshold_match_policy: Threshold rule matching policy. When set to
                "first_match", only the first matching rule is applied. When set to
                "all_matches", modifiers for matching rules are applied sequentially
                (order matters).
            succession_operator: Succession operator used for within-period attractor finding.
            node_mode: Node representation mode. When set to `"equilibrium"`, an unshocked
                relaxation is performed after dynamic-shock succession so that nodes remain
                CIB-consistent with the period matrix.
            max_states_to_enumerate: Maximum scenario space size allowed for enumeration mode.
            n_transition_samples: Number of transition samples used in sampling mode.
            max_nodes_per_period: Optional cap used to prune each period layer for readability.
            prune_policy: Pruning policy applied to each parent's outgoing edge distribution.
                When set to "incoming_mass", only layer-level pruning (if configured) is applied.
                When set to "per_parent_topk", the top-K outgoing edges are kept per parent.
                When set to "min_edge_weight", edges below the given threshold are removed.
            per_parent_top_k: Number of outgoing edges kept per parent when prune_policy is "per_parent_topk".
            min_edge_weight: Minimum edge weight kept when prune_policy is "min_edge_weight".
            base_seed: Base seed used for reproducible sampling.
            structural_sigma: Optional structural shock magnitude applied to sampled matrices.
                Note: only used in sampling mode; ignored in enumeration mode.
            judgment_sigma_scale_by_period: Optional per-period sigma scales used for matrix sampling.
                Note: only used in sampling mode; ignored in enumeration mode.
            dynamic_tau: Optional long-run scale for AR(1) dynamic shocks.
            dynamic_rho: AR(1) persistence parameter in [-1, 1].
            dynamic_innovation_dist: Innovation distribution used for dynamic shocks.
            dynamic_innovation_df: Optional degrees of freedom used for Student-t innovations.
            dynamic_jump_prob: Optional jump probability used for dynamic innovations.
            dynamic_jump_scale: Optional jump scale used for dynamic innovations.

        Raises:
            ValueError: If input parameters are invalid.
        """
        if not periods:
            raise ValueError("periods cannot be empty")
        if max_states_to_enumerate <= 0:
            raise ValueError("max_states_to_enumerate must be positive")
        if n_transition_samples <= 0:
            raise ValueError("n_transition_samples must be positive")
        if max_nodes_per_period is not None and int(max_nodes_per_period) <= 0:
            raise ValueError("max_nodes_per_period must be positive when provided")
        if node_mode not in {"equilibrium", "realized"}:
            raise ValueError("node_mode must be 'equilibrium' or 'realized'")
        if threshold_match_policy not in {"first_match", "all_matches"}:
            raise ValueError("threshold_match_policy must be 'first_match' or 'all_matches'")
        if prune_policy not in {"incoming_mass", "per_parent_topk", "min_edge_weight"}:
            raise ValueError("prune_policy is not recognised")
        if prune_policy == "per_parent_topk" and (per_parent_top_k is None or int(per_parent_top_k) <= 0):
            raise ValueError("per_parent_top_k must be positive for prune_policy='per_parent_topk'")
        if prune_policy == "min_edge_weight" and (min_edge_weight is None or float(min_edge_weight) < 0.0):
            raise ValueError("min_edge_weight must be non-negative for prune_policy='min_edge_weight'")

        self.base_matrix = base_matrix
        self.periods = [int(t) for t in periods]
        self.initial = dict(initial)
        self.cyclic_descriptors = list(cyclic_descriptors or [])
        for cd in self.cyclic_descriptors:
            cd.validate()
        self.threshold_rules = list(threshold_rules or [])
        self.threshold_match_policy = str(threshold_match_policy)
        self.succession_operator = succession_operator or GlobalSuccession()
        self.node_mode = str(node_mode)

        self.max_states_to_enumerate = int(max_states_to_enumerate)
        self.n_transition_samples = int(n_transition_samples)
        self.max_nodes_per_period = int(max_nodes_per_period) if max_nodes_per_period is not None else None
        self.prune_policy = str(prune_policy)
        self.per_parent_top_k = int(per_parent_top_k) if per_parent_top_k is not None else None
        self.min_edge_weight = float(min_edge_weight) if min_edge_weight is not None else None
        self.base_seed = int(base_seed)

        self.structural_sigma = structural_sigma
        self.judgment_sigma_scale_by_period = (
            {int(k): float(v) for k, v in judgment_sigma_scale_by_period.items()}
            if judgment_sigma_scale_by_period is not None
            else None
        )
        self.dynamic_tau = dynamic_tau
        self.dynamic_rho = float(dynamic_rho)
        self.dynamic_innovation_dist = str(dynamic_innovation_dist)
        self.dynamic_innovation_df = dynamic_innovation_df
        self.dynamic_jump_prob = float(dynamic_jump_prob)
        self.dynamic_jump_scale = dynamic_jump_scale

    def _apply_thresholds(self, matrix: CIBMatrix, scenario: Scenario) -> CIBMatrix:
        active = matrix
        for rule in self.threshold_rules:
            if rule.condition(scenario):
                active = rule.modifier(active)
                if self.threshold_match_policy == "first_match":
                    break
        return active

    def _apply_cyclic_transitions(
        self, state: Mapping[str, str], rng: np.random.Generator
    ) -> Dict[str, str]:
        out = dict(state)
        for cd in self.cyclic_descriptors:
            if cd.name not in out:
                raise ValueError(f"Cyclic descriptor {cd.name!r} missing from scenario")
            out[cd.name] = cd.sample_next(out[cd.name], rng)
        return out

    def _lock_map(self, state: Mapping[str, str]) -> Dict[str, str]:
        return {cd.name: str(state[cd.name]) for cd in self.cyclic_descriptors}

    def _sample_matrix_for_period(self, *, period_idx: int, seed: int) -> CIBMatrix:
        m: CIBMatrix = self.base_matrix
        if self.judgment_sigma_scale_by_period is not None and hasattr(m, "sample_matrix"):
            scale = float(self.judgment_sigma_scale_by_period.get(self.periods[period_idx], 1.0))
            try:
                m = m.sample_matrix(int(seed), sigma_scale=scale)  # type: ignore[attr-defined]
            except TypeError:
                m = m.sample_matrix(int(seed))  # type: ignore[attr-defined]
        return m

    def _sample_active_matrix_for_next_period(
        self,
        *,
        scenario_for_threshold: Scenario,
        next_period_idx: int,
        seed: int,
    ) -> CIBMatrix:
        """
        Sample/perturb the matrix for the next period (optional), then apply thresholds.

        The scenario passed in must be the state at the start of the next period
        (after cyclic transitions). Threshold rules are evaluated on this scenario
        so that the active CIM is chosen consistently with `DynamicCIB.simulate_path()`.
        """
        m = self._sample_matrix_for_period(period_idx=next_period_idx, seed=seed)

        if self.structural_sigma is not None:
            sm = ShockModel(m)
            sm.add_structural_shocks(sigma=float(self.structural_sigma))
            m = sm.sample_shocked_matrix(int(seed) + 10_000 + int(next_period_idx))

        return self._apply_thresholds(m, scenario_for_threshold)

    def _find_attractor(
        self,
        *,
        scenario: Scenario,
        matrix: CIBMatrix,
        locked: Mapping[str, str],
        dynamic_shocks: Optional[Mapping[Tuple[str, str], float]] = None,
        max_iterations: int = 1000,
    ) -> Scenario:
        base_op: SuccessionOperator = self.succession_operator

        op_realized: SuccessionOperator = base_op
        if dynamic_shocks is not None:
            op_realized = ShockAwareGlobalSuccession(dynamic_shocks)
        if locked:
            op_realized = _LockedSuccessionOperator(op_realized, locked)

        realized_res = op_realized.find_attractor(
            scenario, matrix, max_iterations=max_iterations
        )
        if realized_res.is_cycle:
            realized_cycle = realized_res.attractor
            assert isinstance(realized_cycle, list)
            realized = realized_cycle[0]
        else:
            realized_attractor = realized_res.attractor
            assert isinstance(realized_attractor, Scenario)
            realized = realized_attractor

        if self.node_mode == "realized" or dynamic_shocks is None:
            return realized

        # An unshocked relaxation is performed so that nodes remain CIB-consistent
        # with the period matrix, while edge weights continue to reflect forcing.
        eq_op: SuccessionOperator = base_op
        if locked:
            eq_op = _LockedSuccessionOperator(eq_op, locked)
        eq_start = Scenario(realized.to_dict(), matrix)
        eq_res = eq_op.find_attractor(eq_start, matrix, max_iterations=max_iterations)
        if eq_res.is_cycle:
            eq_cycle = eq_res.attractor
            assert isinstance(eq_cycle, list)
            return eq_cycle[0]
        eq_attractor = eq_res.attractor
        assert isinstance(eq_attractor, Scenario)
        return eq_attractor

    def build(self, *, top_k: int = 10) -> BranchingResult:
        """
        Build a branching pathway graph across periods.
        """
        periods = tuple(int(t) for t in self.periods)

        has_uncertainty = (
            self.structural_sigma is not None
            or self.judgment_sigma_scale_by_period is not None
        )
        scenario_space_size = _scenario_space_size(self.base_matrix.descriptors)
        can_enumerate = scenario_space_size <= self.max_states_to_enumerate
        if has_uncertainty and can_enumerate:
            warnings.warn(
                "Uncertainty parameters (structural_sigma and/or "
                "judgment_sigma_scale_by_period) were provided, but enumeration "
                f"mode will be used (scenario space size: {scenario_space_size} "
                f"<= {self.max_states_to_enumerate}). These uncertainty parameters "
                "are ignored in enumeration mode. To ensure uncertainty is applied, "
                "decrease `max_states_to_enumerate` to force sampling mode.",
                UserWarning,
                stacklevel=2,
            )

        # The initial state is mapped to its attractor at period 0 (deterministic).
        init_state = dict(self.initial)
        init_lock = self._lock_map(init_state)
        init_matrix = self._apply_thresholds(self.base_matrix, Scenario(init_state, self.base_matrix))
        init_s = Scenario(init_state, init_matrix)
        root = self._find_attractor(scenario=init_s, matrix=init_matrix, locked=init_lock)

        scenarios_by_period: List[List[Scenario]] = [[root]]
        edges: Dict[Tuple[int, int], Dict[int, float]] = {}
        transition_method: Dict[int, str] = {}

        # Expansion is performed forward period by period.
        for p_idx in range(len(periods) - 1):
            t = periods[p_idx]
            t_next = periods[p_idx + 1]

            layer = scenarios_by_period[p_idx]
            next_nodes: List[Scenario] = []
            next_index: Dict[Scenario, int] = {}

            # Method is chosen based on scenario-space size of *unlocked* descriptors.
            # This is a global choice per transition layer (simple and predictable).
            # The worst-case space is computed (no locks), which is conservative.
            method = "enumerate" if can_enumerate else "sample"
            transition_method[int(t)] = method

            for src_idx, parent in enumerate(layer):
                parent_state = parent.to_dict()

                # For each source node, a distribution over next-period nodes is computed.
                if method == "enumerate":
                    rng = np.random.default_rng(
                        int(self.base_seed) + 1000 * p_idx + 10 * src_idx
                    )
                    next_state = self._apply_cyclic_transitions(parent_state, rng)
                    locked = self._lock_map(next_state)
                    active = self._apply_thresholds(
                        self.base_matrix, Scenario(next_state, self.base_matrix)
                    )

                    # Consistent scenarios are enumerated with cyclic descriptors fixed.
                    candidates = _enumerate_consistent_with_locks(matrix=active, locked=locked)
                    if not candidates:
                        # The deterministic successor is used as a fallback if enumeration yields none.
                        s0 = Scenario(next_state, active)
                        chosen = self._find_attractor(scenario=s0, matrix=active, locked=locked)
                        candidates = [chosen]

                    w = 1.0 / float(len(candidates))
                    dist_s: Dict[Scenario, float] = {c: w for c in candidates}
                    dist_s = _prune_distribution(
                        dist_s,
                        prune_policy=self.prune_policy,
                        per_parent_top_k=self.per_parent_top_k,
                        min_edge_weight=self.min_edge_weight,
                    )
                    out: Dict[int, float] = {}
                    for c, ww in dist_s.items():
                        if c not in next_index:
                            next_index[c] = len(next_nodes)
                            next_nodes.append(c)
                        out[next_index[c]] = out.get(next_index[c], 0.0) + float(ww)
                    edges[(p_idx, src_idx)] = out
                    continue

                # Sampling mode: transition distribution is estimated by Monte Carlo.
                counts_s: Dict[Scenario, int] = {}
                for m in range(self.n_transition_samples):
                    seeds = seeds_for_run(self.base_seed + 1000 * p_idx + 17 * src_idx, m)
                    rng = np.random.default_rng(int(seeds["dynamic_shock_seed"]))

                    next_state = self._apply_cyclic_transitions(parent_state, rng)
                    locked = self._lock_map(next_state)

                    active = self._sample_active_matrix_for_next_period(
                        scenario_for_threshold=Scenario(next_state, self.base_matrix),
                        next_period_idx=p_idx + 1,
                        seed=int(seeds["judgment_uncertainty_seed"]),
                    )

                    dyn_shocks = None
                    if self.dynamic_tau is not None:
                        sm = ShockModel(active)
                        sm.add_dynamic_shocks(
                            periods=[int(t_next)],
                            tau=float(self.dynamic_tau),
                            rho=float(self.dynamic_rho),
                            innovation_dist=self.dynamic_innovation_dist,
                            innovation_df=self.dynamic_innovation_df,
                            jump_prob=self.dynamic_jump_prob,
                            jump_scale=self.dynamic_jump_scale,
                        )
                        dyn = sm.sample_dynamic_shocks(int(seeds["dynamic_shock_seed"]))
                        dyn_shocks = dyn[int(t_next)]

                    s0 = Scenario(next_state, active)
                    child = self._find_attractor(
                        scenario=s0,
                        matrix=active,
                        locked=locked,
                        dynamic_shocks=dyn_shocks,
                    )

                    counts_s[child] = counts_s.get(child, 0) + 1

                total = float(sum(counts_s.values()) or 1.0)
                dist_s = {s: float(c) / total for s, c in counts_s.items()}
                dist_s = _prune_distribution(
                    dist_s,
                    prune_policy=self.prune_policy,
                    per_parent_top_k=self.per_parent_top_k,
                    min_edge_weight=self.min_edge_weight,
                )
                out: Dict[int, float] = {}
                for child, ww in dist_s.items():
                    if child not in next_index:
                        next_index[child] = len(next_nodes)
                        next_nodes.append(child)
                    out[next_index[child]] = out.get(next_index[child], 0.0) + float(ww)
                edges[(p_idx, src_idx)] = out

            # Optional layer-level pruning is performed to prevent node explosion.
            if self.max_nodes_per_period is not None and len(next_nodes) > self.max_nodes_per_period:
                incoming: Dict[int, float] = {}
                for src_idx in range(len(layer)):
                    out = edges.get((p_idx, src_idx), {})
                    for tgt_idx, w in out.items():
                        incoming[int(tgt_idx)] = incoming.get(int(tgt_idx), 0.0) + float(w)

                keep_n = max(1, int(self.max_nodes_per_period))
                kept_old = {
                    idx for idx, _w in sorted(incoming.items(), key=lambda x: x[1], reverse=True)[:keep_n]
                }
                if not kept_old:
                    kept_old = {0}

                remap = {old: new for new, old in enumerate(sorted(kept_old))}
                new_next_nodes = [next_nodes[old] for old in sorted(kept_old)]

                # Each parent's outgoing distribution is filtered and renormalized.
                for src_idx in range(len(layer)):
                    out = edges.get((p_idx, src_idx), {})
                    filtered = {remap[int(k)]: float(v) for k, v in out.items() if int(k) in kept_old}
                    s = float(sum(filtered.values()))
                    if s <= 0.0:
                        # If everything was pruned away, the most likely kept node is used as a fallback.
                        # This keeps the graph connected for plotting and top-path extraction.
                        best_old = max(kept_old, key=lambda i: incoming.get(int(i), 0.0))
                        filtered = {remap[int(best_old)]: 1.0}
                    else:
                        filtered = {k: v / s for k, v in filtered.items()}
                    edges[(p_idx, src_idx)] = filtered

                scenarios_by_period.append(new_next_nodes)
            else:
                scenarios_by_period.append(next_nodes)

        # Top-K most likely paths are computed (simple beam search).
        top_k = max(1, int(top_k))
        paths: List[Tuple[Tuple[int, ...], float]] = [((0,), 1.0)]
        for p_idx in range(len(periods) - 1):
            new_paths: List[Tuple[Tuple[int, ...], float]] = []
            for node_path, w in paths:
                src = node_path[-1]
                out = edges.get((p_idx, src), {})
                for nxt, p in out.items():
                    new_paths.append((node_path + (int(nxt),), float(w) * float(p)))
            new_paths.sort(key=lambda x: x[1], reverse=True)
            paths = new_paths[:top_k]

        return BranchingResult(
            periods=periods,
            scenarios_by_period=tuple(tuple(layer) for layer in scenarios_by_period),
            edges=edges,
            transition_method=transition_method,
            top_paths=tuple(paths),
        )

