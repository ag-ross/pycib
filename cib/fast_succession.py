"""
Fast succession utilities built on the numeric scoring backend.

In this module, low-overhead succession loops are provided in index space.
These utilities are intended to support Monte Carlo attractor discovery and
other scaling workflows where repeated succession is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, Tuple, Union

import numpy as np

class _ScoringBackend(Protocol):
    """
    A scoring backend interface is defined for fast succession.
    """

    descriptors: Sequence[str]
    state_counts: np.ndarray

    def scores_for_scenario(self, z_idx: np.ndarray) -> np.ndarray: ...

    def apply_change(
        self,
        scores: np.ndarray,
        *,
        changed_descriptor_idx: int,
        old_state_idx: int,
        new_state_idx: int,
    ) -> np.ndarray: ...


SuccessionRule = Literal["global", "local"]


@dataclass(frozen=True)
class AttractorIndicesResult:
    """
    Result of a fast succession run in index space.
    """

    attractor: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]
    path: Tuple[Tuple[int, ...], ...]
    iterations: int
    is_cycle: bool


def run_to_attractor_indices(
    *,
    scorer: _ScoringBackend,
    initial_z_idx: Sequence[int],
    rule: SuccessionRule = "global",
    max_iterations: int = 1000,
) -> AttractorIndicesResult:
    """
    Run succession until a fixed point or a cycle is reached.

    Notes:
    - Incremental score updates are used so that scores are computed once and
      then updated as descriptor states change.
    - Cycle detection is performed using a visited map keyed by the full state
      vector.
    """
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")

    z = np.asarray(list(initial_z_idx), dtype=np.int64)
    n_desc = int(len(scorer.descriptors))
    if z.shape != (n_desc,):
        raise ValueError(
            f"initial_z_idx must have length {n_desc}, but length {int(z.size)} was provided"
        )

    scores = scorer.scores_for_scenario(z)
    path: list[Tuple[int, ...]] = [tuple(int(x) for x in z)]
    visited: dict[Tuple[int, ...], int] = {path[0]: 0}

    for iteration in range(1, int(max_iterations) + 1):
        if rule == "global":
            nxt = np.empty((n_desc,), dtype=np.int64)
            changed: list[int] = []
            for j in range(n_desc):
                n_states = int(scorer.state_counts[j])
                nxt_j = int(np.argmax(scores[j, :n_states]))
                nxt[j] = nxt_j
                if nxt_j != int(z[j]):
                    changed.append(j)
        elif rule == "local":
            nxt = np.asarray(z, dtype=np.int64).copy()
            best_gap = float("-inf")
            target = None
            target_state = None
            for j in range(n_desc):
                n_states = int(scorer.state_counts[j])
                chosen_idx = int(z[j])
                chosen = float(scores[j, chosen_idx])
                best_idx = int(np.argmax(scores[j, :n_states]))
                best = float(scores[j, best_idx])
                gap = best - chosen
                if gap > best_gap:
                    best_gap = gap
                    target = j
                    target_state = best_idx
            changed = []
            if target is not None and target_state is not None and best_gap > 0.0:
                changed = [int(target)]
                nxt[int(target)] = int(target_state)
        else:
            raise ValueError("rule must be 'global' or 'local'")

        # Incremental update of the score matrix is applied for the change set.
        for j in changed:
            scorer.apply_change(
                scores,
                changed_descriptor_idx=int(j),
                old_state_idx=int(z[j]),
                new_state_idx=int(nxt[j]),
            )

        z = nxt
        key = tuple(int(x) for x in z)
        path.append(key)

        if key == path[-2]:
            return AttractorIndicesResult(
                attractor=key,
                path=tuple(path),
                iterations=int(iteration),
                is_cycle=False,
            )

        if key in visited:
            cycle_start = int(visited[key])
            cycle = tuple(path[cycle_start:-1])
            return AttractorIndicesResult(
                attractor=cycle,
                path=tuple(path),
                iterations=int(iteration),
                is_cycle=True,
            )

        visited[key] = int(iteration)

    raise RuntimeError(
        f"Succession did not converge within {int(max_iterations)} iterations"
    )

