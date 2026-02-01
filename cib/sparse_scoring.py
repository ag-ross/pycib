"""
Sparse scoring backend for CIB matrices.

In this module, an alternative scoring backend is provided for cases where the
impact structure is sparse and the dense padded tensor used by `FastCIBScorer`
is not desirable.

The scoring semantics are matched to the deterministic reference:
  - Impact balances are computed as sums of impacts from the currently selected
    source states to each target state.
  - Only explicitly non-zero impacts are stored and processed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from cib.core import CIBMatrix, Scenario


@dataclass(frozen=True)
class SparseCIBScorer:
    """
    Sparse scorer for a `CIBMatrix`.

    Notes:
    - Impacts are stored per source state as sparse row contributions into the
      (target_descriptor, target_state) score matrix.
    - The public surface is aligned to `FastCIBScorer` so it can be used by the
      same succession and Monte Carlo code paths.
    """

    descriptors: Sequence[str]
    state_labels: Sequence[Sequence[str]]
    state_counts: np.ndarray
    max_states: int
    descriptor_index: Mapping[str, int]
    state_index: Sequence[Mapping[str, int]]

    # Per source-state row (flattened by descriptor offsets):
    # each row contributes values to (tgt_desc_idx, tgt_state_idx).
    row_tgt_desc: Sequence[np.ndarray]
    row_tgt_state: Sequence[np.ndarray]
    row_values: Sequence[np.ndarray]

    offsets: np.ndarray

    @staticmethod
    def from_matrix(matrix: CIBMatrix) -> "SparseCIBScorer":
        descriptors = list(matrix.descriptors.keys())
        state_labels: List[List[str]] = [list(matrix.descriptors[d]) for d in descriptors]
        state_counts = np.asarray([len(s) for s in state_labels], dtype=np.int64)
        max_states = int(state_counts.max()) if len(state_counts) else 0

        desc_to_idx: Dict[str, int] = {d: i for i, d in enumerate(descriptors)}
        state_to_idx: List[Dict[str, int]] = []
        for states in state_labels:
            state_to_idx.append({s: i for i, s in enumerate(states)})

        offsets = np.zeros((len(descriptors) + 1,), dtype=np.int64)
        for j in range(len(descriptors)):
            offsets[j + 1] = offsets[j] + int(state_counts[j])
        total_states = int(offsets[-1])

        rows_desc: List[List[int]] = [[] for _ in range(total_states)]
        rows_state: List[List[int]] = [[] for _ in range(total_states)]
        rows_val: List[List[float]] = [[] for _ in range(total_states)]

        # Only explicitly non-zero impacts are stored.
        for (src_desc, src_state, tgt_desc, tgt_state), v in matrix.iter_impacts():
            fv = float(v)
            if float(fv) == 0.0:
                continue
            i = int(desc_to_idx[str(src_desc)])
            j = int(desc_to_idx[str(tgt_desc)])
            k = int(state_to_idx[i][str(src_state)])
            l = int(state_to_idx[j][str(tgt_state)])

            src_flat = int(offsets[i] + k)
            rows_desc[src_flat].append(int(j))
            rows_state[src_flat].append(int(l))
            rows_val[src_flat].append(float(fv))

        row_tgt_desc: List[np.ndarray] = []
        row_tgt_state: List[np.ndarray] = []
        row_values: List[np.ndarray] = []
        for r in range(total_states):
            row_tgt_desc.append(np.asarray(rows_desc[r], dtype=np.int64))
            row_tgt_state.append(np.asarray(rows_state[r], dtype=np.int64))
            row_values.append(np.asarray(rows_val[r], dtype=np.float64))

        return SparseCIBScorer(
            descriptors=descriptors,
            state_labels=state_labels,
            state_counts=state_counts,
            max_states=max_states,
            descriptor_index=desc_to_idx,
            state_index=state_to_idx,
            row_tgt_desc=row_tgt_desc,
            row_tgt_state=row_tgt_state,
            row_values=row_values,
            offsets=offsets,
        )

    def scenario_to_indices(self, scenario: Scenario) -> np.ndarray:
        return np.asarray(scenario.to_indices(), dtype=np.int64)

    def scores_for_scenario(self, z_idx: np.ndarray) -> np.ndarray:
        z = np.asarray(z_idx, dtype=np.int64)
        n_desc = int(len(self.descriptors))
        if z.shape != (n_desc,):
            raise ValueError(
                f"z_idx must have shape ({n_desc},), but shape {tuple(z.shape)} was provided"
            )

        scores = np.zeros((n_desc, int(self.max_states)), dtype=np.float64)
        for i in range(n_desc):
            k = int(z[i])
            src_flat = int(self.offsets[i] + k)
            td = self.row_tgt_desc[src_flat]
            ts = self.row_tgt_state[src_flat]
            vv = self.row_values[src_flat]
            if int(vv.size) == 0:
                continue
            np.add.at(scores, (td, ts), vv)
        return scores

    def apply_change(
        self,
        scores: np.ndarray,
        *,
        changed_descriptor_idx: int,
        old_state_idx: int,
        new_state_idx: int,
    ) -> np.ndarray:
        j = int(changed_descriptor_idx)
        old = int(old_state_idx)
        new = int(new_state_idx)

        old_flat = int(self.offsets[j] + old)
        td = self.row_tgt_desc[old_flat]
        ts = self.row_tgt_state[old_flat]
        vv = self.row_values[old_flat]
        if int(vv.size) != 0:
            np.add.at(scores, (td, ts), -vv)

        new_flat = int(self.offsets[j] + new)
        td2 = self.row_tgt_desc[new_flat]
        ts2 = self.row_tgt_state[new_flat]
        vv2 = self.row_values[new_flat]
        if int(vv2.size) != 0:
            np.add.at(scores, (td2, ts2), vv2)

        return scores

    def is_consistent(
        self,
        z_idx: np.ndarray,
        *,
        float_atol: float = 1e-08,
        float_rtol: float = 1e-05,
    ) -> bool:
        scores = self.scores_for_scenario(z_idx)
        z = np.asarray(z_idx, dtype=np.int64)
        for j in range(int(len(self.descriptors))):
            n_states = int(self.state_counts[j])
            chosen_idx = int(z[j])
            chosen = float(scores[j, chosen_idx])
            best = float(scores[j, :n_states].max(initial=float("-inf")))
            if best > chosen and not np.isclose(
                best, chosen, atol=float(float_atol), rtol=float(float_rtol)
            ):
                return False
        return True

