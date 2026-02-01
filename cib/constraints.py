"""
Feasibility constraints for scenarios (external to CIB consistency).

These constraints represent user/domain rules such as:
  - forbidden state combinations across descriptors,
  - simple implications (A=a => B=b),
  - allowed-state restrictions per descriptor.

They are compiled into index space once per matrix so they can be applied in:
  - exact enumeration with partial assignments (z uses -1 for unassigned),
  - succession / Monte Carlo workflows (full assignments).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from cib.core import CIBMatrix


@dataclass(frozen=True)
class ForbiddenPair:
    """
    Forbid simultaneous assignment of (a_desc=a_state) and (b_desc=b_state).
    """

    a_desc: str
    a_state: str
    b_desc: str
    b_state: str


@dataclass(frozen=True)
class Implies:
    """
    Enforce implication: (a_desc=a_state) => (b_desc=b_state).
    """

    a_desc: str
    a_state: str
    b_desc: str
    b_state: str


@dataclass(frozen=True)
class AllowedStates:
    """
    Restrict a descriptor to an explicit allowed set of state labels.
    """

    desc: str
    allowed: Set[str]


ConstraintSpec = Union[ForbiddenPair, Implies, AllowedStates]


class ConstraintIndex:
    """
    Compiled constraint helper in index space.

    Conventions:
      - Descriptor indices match `list(matrix.descriptors.keys())`.
      - State indices match `matrix.descriptors[desc].index(label)`.
      - Partial assignments use -1 for "unassigned".
    """

    def __init__(
        self,
        *,
        descriptors: Sequence[str],
        state_counts: Sequence[int],
        allowed_masks: Sequence[np.ndarray],
        forbidden: Mapping[Tuple[int, int], Set[Tuple[int, int]]],
        implies_out: Mapping[Tuple[int, int], Tuple[Tuple[int, int], ...]],
    ) -> None:
        self.descriptors = list(descriptors)
        self.state_counts = np.asarray(list(state_counts), dtype=np.int64)
        self.allowed_masks: List[np.ndarray] = [np.asarray(m, dtype=bool) for m in allowed_masks]
        self.forbidden: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {
            (int(a), int(b)): set(v) for (a, b), v in forbidden.items()
        }
        self.implies_out: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {
            (int(a), int(sa)): tuple((int(b), int(sb)) for (b, sb) in outs)
            for (a, sa), outs in implies_out.items()
        }

        n = int(len(self.descriptors))
        if self.state_counts.shape != (n,):
            raise ValueError("state_counts must have shape (n_descriptors,)")
        if len(self.allowed_masks) != n:
            raise ValueError("allowed_masks must have one entry per descriptor")
        for j in range(n):
            if self.allowed_masks[j].shape != (int(self.state_counts[j]),):
                raise ValueError("allowed_masks entries must match descriptor arities")

    @staticmethod
    def from_specs(
        matrix: CIBMatrix, specs: Optional[Sequence[ConstraintSpec]]
    ) -> Optional["ConstraintIndex"]:
        if not specs:
            return None

        descriptors = list(matrix.descriptors.keys())
        desc_to_idx: Dict[str, int] = {d: i for i, d in enumerate(descriptors)}
        state_to_idx: List[Dict[str, int]] = [
            {s: k for k, s in enumerate(matrix.descriptors[d])} for d in descriptors
        ]
        state_counts = [len(matrix.descriptors[d]) for d in descriptors]

        allowed_masks: List[np.ndarray] = [
            np.ones((int(n),), dtype=bool) for n in state_counts
        ]

        forbidden: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        implies_out: Dict[Tuple[int, int], Dict[int, int]] = {}

        def _desc_idx(name: str) -> int:
            if name not in desc_to_idx:
                raise ValueError(f"Unknown descriptor {name!r} in constraints")
            return int(desc_to_idx[name])

        def _state_idx(d_idx: int, label: str) -> int:
            try:
                return int(state_to_idx[d_idx][label])
            except Exception as e:  # noqa: BLE001
                d = descriptors[d_idx]
                raise ValueError(
                    f"Unknown state {label!r} for descriptor {d!r} in constraints"
                ) from e

        for spec in specs:
            if isinstance(spec, AllowedStates):
                d = _desc_idx(str(spec.desc))
                allowed = set(str(x) for x in spec.allowed)
                if not allowed:
                    raise ValueError("AllowedStates.allowed must be non-empty")
                mask = np.zeros((int(state_counts[d]),), dtype=bool)
                for lab in allowed:
                    k = _state_idx(d, lab)
                    mask[int(k)] = True
                allowed_masks[d] = np.logical_and(allowed_masks[d], mask)
            elif isinstance(spec, ForbiddenPair):
                a = _desc_idx(str(spec.a_desc))
                b = _desc_idx(str(spec.b_desc))
                sa = _state_idx(a, str(spec.a_state))
                sb = _state_idx(b, str(spec.b_state))
                forbidden.setdefault((a, b), set()).add((sa, sb))
                forbidden.setdefault((b, a), set()).add((sb, sa))
            elif isinstance(spec, Implies):
                a = _desc_idx(str(spec.a_desc))
                b = _desc_idx(str(spec.b_desc))
                sa = _state_idx(a, str(spec.a_state))
                sb = _state_idx(b, str(spec.b_state))
                key = (a, sa)
                row = implies_out.setdefault(key, {})
                prev = row.get(b)
                if prev is not None and int(prev) != int(sb):
                    raise ValueError(
                        "Conflicting Implies constraints: "
                        f"{spec.a_desc}={spec.a_state} requires {spec.b_desc}="
                        f"{descriptors[b]}[{prev}] and {spec.b_state}"
                    )
                row[b] = int(sb)
            else:
                raise ValueError(f"Unknown constraint spec type: {type(spec)!r}")

        # Validate that no descriptor is rendered infeasible by AllowedStates.
        for d, mask in enumerate(allowed_masks):
            if not bool(mask.any()):
                raise ValueError(
                    f"AllowedStates constraints eliminate all states for descriptor {descriptors[d]!r}"
                )

        implies_out_tuples: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {
            key: tuple(sorted(row.items(), key=lambda x: x[0])) for key, row in implies_out.items()
        }
        return ConstraintIndex(
            descriptors=descriptors,
            state_counts=state_counts,
            allowed_masks=allowed_masks,
            forbidden=forbidden,
            implies_out=implies_out_tuples,
        )

    def is_full_valid(self, z_idx: Sequence[int]) -> bool:
        z = np.asarray(list(z_idx), dtype=np.int64)
        if z.shape != (int(len(self.descriptors)),):
            raise ValueError("z_idx has wrong shape")
        return bool(self.is_partial_valid(z))

    def is_partial_valid(self, z_idx: np.ndarray, *, just_set: Optional[int] = None) -> bool:
        z = np.asarray(z_idx, dtype=np.int64)
        n = int(len(self.descriptors))
        if z.shape != (n,):
            raise ValueError(f"z_idx must have shape ({n},)")

        if just_set is None:
            # Unary constraints are checked.
            for j in range(n):
                sj = int(z[j])
                if sj < 0:
                    continue
                if not bool(self.allowed_masks[j][sj]):
                    return False

            # Binary constraints are checked in O(n^2), used sparingly.
            for a in range(n):
                sa = int(z[a])
                if sa < 0:
                    continue
                outs = self.implies_out.get((a, sa))
                if outs:
                    for (b, sb) in outs:
                        sb_cur = int(z[int(b)])
                        if sb_cur >= 0 and sb_cur != int(sb):
                            return False
                for b in range(n):
                    sb = int(z[b])
                    if sb < 0:
                        continue
                    forb = self.forbidden.get((a, b))
                    if forb and (sa, sb) in forb:
                        return False
            return True

        d = int(just_set)
        sd = int(z[d])
        if sd < 0:
            return True
        if not bool(self.allowed_masks[d][sd]):
            return False

        # Implications where (d, sd) is antecedent are checked.
        outs = self.implies_out.get((d, sd))
        if outs:
            for (b, sb_req) in outs:
                sb_cur = int(z[int(b)])
                if sb_cur >= 0 and sb_cur != int(sb_req):
                    return False

        # Constraints involving already-assigned descriptors are checked.
        for j in range(n):
            if j == d:
                continue
            sj = int(z[j])
            if sj < 0:
                continue

            forb = self.forbidden.get((d, j))
            if forb and (sd, sj) in forb:
                return False

            # Implications where (j, sj) is antecedent and d is consequent are checked.
            outs_j = self.implies_out.get((j, sj))
            if outs_j:
                for (b, sb_req) in outs_j:
                    if int(b) == int(d) and int(sb_req) != int(sd):
                        return False

        return True

    def allowed_states_for_descriptor(self, z_idx: np.ndarray, *, desc_idx: int) -> np.ndarray:
        """
        Compute a boolean mask of allowed states for `desc_idx` given the current
        (possibly partial) assignment.
        """
        z = np.asarray(z_idx, dtype=np.int64)
        n = int(len(self.descriptors))
        d = int(desc_idx)
        if z.shape != (n,):
            raise ValueError(f"z_idx must have shape ({n},)")
        n_states = int(self.state_counts[d])

        allowed = self.allowed_masks[d].copy()

        # Any already-assigned descriptor may constrain d via:
        #  - forbidden pairs
        #  - implications where the assigned descriptor is antecedent and d is consequent
        for j in range(n):
            if j == d:
                continue
            sj = int(z[j])
            if sj < 0:
                continue

            forb = self.forbidden.get((d, j))
            if forb:
                for (sd, sj2) in forb:
                    if int(sj2) == int(sj) and 0 <= int(sd) < n_states:
                        allowed[int(sd)] = False

            outs_j = self.implies_out.get((j, sj))
            if outs_j:
                for (b, sb_req) in outs_j:
                    if int(b) == int(d):
                        # Only one state is allowed by this implication.
                        req = int(sb_req)
                        m = np.zeros((n_states,), dtype=bool)
                        if 0 <= req < n_states:
                            m[req] = True
                        allowed = np.logical_and(allowed, m)

        return allowed

