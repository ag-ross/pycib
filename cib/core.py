"""
Core CIB data structures and consistency checking.

This module provides the fundamental classes for Cross-Impact Balance analysis:
CIBMatrix for storing impact relationships, Scenario for representing state
assignments, ConsistencyChecker for validating scenarios, and ImpactBalance
for computing impact scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np


class CIBMatrix:
    """
    Cross-Impact Balance matrix storing impact relationships.

    The matrix stores an N×N hypermatrix where each element C_ij is an
    s_i × s_j matrix representing impacts from descriptor i to descriptor j.
    Each cell C_ij(k,l) represents the impact of state k of descriptor i
    on state l of descriptor j.

    Attributes:
        descriptors: Dictionary mapping descriptor names to their state lists.
        n_descriptors: Number of descriptors in the system.
        state_counts: List of state counts for each descriptor.
        _impacts: Internal storage of impact values as (src, src_state, tgt, tgt_state) -> value.
    """

    def __init__(self, descriptors: Dict[str, List[str]]) -> None:
        """
        Initialize CIB matrix with descriptor definitions.

        Args:
            descriptors: Dictionary mapping descriptor names to lists of
                possible state labels.

        Raises:
            ValueError: If descriptors dictionary is empty or contains
                invalid state lists.
        """
        if not descriptors:
            raise ValueError("Descriptors dictionary cannot be empty")

        for desc_name, states in descriptors.items():
            if not states:
                raise ValueError(
                    f"Descriptor '{desc_name}' must have at least one state"
                )
            if len(set(states)) != len(states):
                raise ValueError(
                    f"Descriptor '{desc_name}' has duplicate states"
                )

        self.descriptors: Dict[str, List[str]] = descriptors.copy()
        self.n_descriptors: int = len(descriptors)
        self.state_counts: List[int] = [
            len(states) for states in descriptors.values()
        ]
        self._impacts: Dict[Tuple[str, str, str, str], float] = {}

    def set_impact(
        self,
        src_desc: str,
        src_state: str,
        tgt_desc: str,
        tgt_state: str,
        value: float,
    ) -> None:
        """
        Set a single impact value in the matrix.

        Args:
            src_desc: Source descriptor name.
            src_state: Source state label.
            tgt_desc: Target descriptor name.
            tgt_state: Target state label.
            value: Impact value (typically in range [-3, +3]).

        Raises:
            ValueError: If descriptor or state names are not found.
        """
        if src_desc not in self.descriptors:
            raise ValueError(f"Source descriptor '{src_desc}' not found")
        if tgt_desc not in self.descriptors:
            raise ValueError(f"Target descriptor '{tgt_desc}' not found")
        if src_desc == tgt_desc:
            raise ValueError(
                "Self-impacts (diagonal judgment sections) are omitted by convention in CIB"
            )
        if src_state not in self.descriptors[src_desc]:
            raise ValueError(
                f"Source state '{src_state}' not found for "
                f"descriptor '{src_desc}'"
            )
        if tgt_state not in self.descriptors[tgt_desc]:
            raise ValueError(
                f"Target state '{tgt_state}' not found for "
                f"descriptor '{tgt_desc}'"
            )

        key = (src_desc, src_state, tgt_desc, tgt_state)
        self._impacts[key] = float(value)

    def set_impacts(
        self, impacts: Dict[Tuple[str, str, str, str], float]
    ) -> None:
        """
        Set multiple impact values in bulk.

        Args:
            impacts: Dictionary mapping (src_desc, src_state, tgt_desc,
                tgt_state) tuples to impact values.

        Raises:
            ValueError: If any descriptor or state names are invalid.
        """
        for key, value in impacts.items():
            src_desc, src_state, tgt_desc, tgt_state = key
            self.set_impact(src_desc, src_state, tgt_desc, tgt_state, value)

    def get_impact(
        self, src_desc: str, src_state: str, tgt_desc: str, tgt_state: str
    ) -> float:
        """
        Retrieve an impact value from the matrix.

        Args:
            src_desc: Source descriptor name.
            src_state: Source state label.
            tgt_desc: Target descriptor name.
            tgt_state: Target state label.

        Returns:
            The impact value, or 0.0 if not explicitly set.

        Raises:
            ValueError: If descriptor or state names are not found.
        """
        if src_desc not in self.descriptors:
            raise ValueError(f"Source descriptor '{src_desc}' not found")
        if tgt_desc not in self.descriptors:
            raise ValueError(f"Target descriptor '{tgt_desc}' not found")
        if src_state not in self.descriptors[src_desc]:
            raise ValueError(
                f"Source state '{src_state}' not found for "
                f"descriptor '{src_desc}'"
            )
        if tgt_state not in self.descriptors[tgt_desc]:
            raise ValueError(
                f"Target state '{tgt_state}' not found for "
                f"descriptor '{tgt_desc}'"
            )

        key = (src_desc, src_state, tgt_desc, tgt_state)
        return self._impacts.get(key, 0.0)

    def iter_impacts(self):
        """
        Iterate over explicitly stored non-zero impacts.

        Returns:
            An iterator over ((src_desc, src_state, tgt_desc, tgt_state), value) pairs.
        """
        return self._impacts.items()

    def calculate_impact_score(
        self, scenario: Scenario, descriptor: str, state: str
    ) -> float:
        """
        Calculate impact score for a specific descriptor-state combination.

        The impact score θ_jl represents the sum of all impacts from other
        descriptors' current states in the scenario to the target state.

        Args:
            scenario: The scenario containing current state assignments.
            descriptor: Target descriptor name.
            state: Target state label for the descriptor.

        Returns:
            The computed impact score as a float.

        Raises:
            ValueError: If descriptor or state is not found in the matrix.
        """
        if descriptor not in self.descriptors:
            raise ValueError(f"Descriptor '{descriptor}' not found")
        if state not in self.descriptors[descriptor]:
            raise ValueError(
                f"State '{state}' not found for descriptor '{descriptor}'"
            )

        score = 0.0
        for src_desc in self.descriptors:
            if src_desc == descriptor:
                continue
            src_state = scenario.get_state(src_desc)
            impact = self.get_impact(src_desc, src_state, descriptor, state)
            score += impact

        return score

    def calculate_impact_balance(
        self, scenario: Scenario, descriptor: str
    ) -> Dict[str, float]:
        """
        Calculate impact balance for all states of a descriptor.

        Args:
            scenario: The scenario containing current state assignments.
            descriptor: Target descriptor name.

        Returns:
            Dictionary mapping state labels to their impact scores.

        Raises:
            ValueError: If descriptor is not found.
        """
        if descriptor not in self.descriptors:
            raise ValueError(f"Descriptor '{descriptor}' not found")

        balance: Dict[str, float] = {}
        for state in self.descriptors[descriptor]:
            balance[state] = self.calculate_impact_score(
                scenario, descriptor, state
            )

        return balance

    def standardize(self) -> None:
        """
        Apply standardization to judgment groups.

        For each judgment group (row in a judgment section), ensure the
        sum equals zero by adjusting values. This implements IO-1
        (Addition Invariance) normalization.
        """
        for src_desc in self.descriptors:
            for src_state in self.descriptors[src_desc]:
                for tgt_desc in self.descriptors:
                    if src_desc == tgt_desc:
                        continue

                    judgment_group = []
                    for tgt_state in self.descriptors[tgt_desc]:
                        key = (src_desc, src_state, tgt_desc, tgt_state)
                        if key in self._impacts:
                            judgment_group.append((key, self._impacts[key]))

                    if judgment_group:
                        current_sum = sum(val for _, val in judgment_group)
                        adjustment = -current_sum / len(judgment_group)
                        for key, val in judgment_group:
                            self._impacts[key] = val + adjustment

    def apply_io1(
        self, descriptor: str, state: str, value: float
    ) -> None:
        """
        Apply IO-1 (Addition Invariance) operation.

        Add a constant value to all cells in a judgment group. This preserves
        consistency relationships.

        Args:
            descriptor: Source descriptor name.
            state: Source state label.
            value: Constant value to add to all judgment group cells.

        Raises:
            ValueError: If descriptor or state is not found.
        """
        if descriptor not in self.descriptors:
            raise ValueError(f"Descriptor '{descriptor}' not found")
        if state not in self.descriptors[descriptor]:
            raise ValueError(
                f"State '{state}' not found for descriptor '{descriptor}'"
            )

        for tgt_desc in self.descriptors:
            if tgt_desc == descriptor:
                continue
            for tgt_state in self.descriptors[tgt_desc]:
                key = (descriptor, state, tgt_desc, tgt_state)
                if key in self._impacts:
                    self._impacts[key] += value
                else:
                    self._impacts[key] = value

    def apply_io2(self, descriptor: str, multiplier: float) -> None:
        """
        Apply IO-2 (Local Multiplication Invariance) operation.

        Multiply all cells in a descriptor column by a positive number.
        This preserves consistency relationships.

        Args:
            descriptor: Target descriptor name.
            multiplier: Positive multiplier value.

        Raises:
            ValueError: If descriptor is not found or multiplier is
                non-positive.
        """
        if descriptor not in self.descriptors:
            raise ValueError(f"Descriptor '{descriptor}' not found")
        if multiplier <= 0:
            raise ValueError("Multiplier must be positive")

        for src_desc in self.descriptors:
            if src_desc == descriptor:
                continue
            for src_state in self.descriptors[src_desc]:
                for tgt_state in self.descriptors[descriptor]:
                    key = (src_desc, src_state, descriptor, tgt_state)
                    if key in self._impacts:
                        self._impacts[key] *= multiplier

    def apply_io3(self, multiplier: float) -> None:
        """
        Apply IO-3 (Global Multiplication Invariance) operation.

        Multiply the entire (off-diagonal) matrix by a positive number.
        This preserves consistency relationships.

        Args:
            multiplier: Positive multiplier value.

        Raises:
            ValueError: If multiplier is non-positive.
        """
        if multiplier <= 0:
            raise ValueError("Multiplier must be positive")

        for key in list(self._impacts.keys()):
            self._impacts[key] *= multiplier

    def apply_io4(
        self,
        src_desc: str,
        from_state: str,
        to_state: str,
        tgt_desc: str,
        value: float,
    ) -> None:
        """
        Apply IO-4 (Transfer Invariance) operation.

        Transfer a constant between two judgment groups (rows) within the same
        judgment section (src_desc -> tgt_desc), while preserving consistency.

        Operationally:
          - Subtract `value` from all cells (src_desc, from_state, tgt_desc, *)
          - Add      `value` to all cells (src_desc, to_state,   tgt_desc, *)

        This preserves consistency because, for any scenario, the source
        descriptor contributes a constant offset (applied equally to all target
        states), and constant offsets do not change which target state is a
        maximum.

        Args:
            src_desc: Source descriptor name.
            from_state: Source state to transfer from.
            to_state: Source state to transfer to.
            tgt_desc: Target descriptor name.
            value: Constant value to transfer (can be negative).

        Raises:
            ValueError: If descriptors/states are invalid, or if src_desc == tgt_desc.
        """
        if src_desc not in self.descriptors:
            raise ValueError(f"Source descriptor '{src_desc}' not found")
        if tgt_desc not in self.descriptors:
            raise ValueError(f"Target descriptor '{tgt_desc}' not found")
        if src_desc == tgt_desc:
            raise ValueError(
                "IO-4 transfer is defined only for off-diagonal judgment sections"
            )
        if from_state not in self.descriptors[src_desc]:
            raise ValueError(
                f"Source state '{from_state}' not found for descriptor '{src_desc}'"
            )
        if to_state not in self.descriptors[src_desc]:
            raise ValueError(
                f"Source state '{to_state}' not found for descriptor '{src_desc}'"
            )

        value = float(value)
        for tgt_state in self.descriptors[tgt_desc]:
            from_key = (src_desc, from_state, tgt_desc, tgt_state)
            to_key = (src_desc, to_state, tgt_desc, tgt_state)
            self._impacts[from_key] = self._impacts.get(from_key, 0.0) - value
            self._impacts[to_key] = self._impacts.get(to_key, 0.0) + value


class Scenario:
    """
    Represents a scenario as a state assignment vector.

    A scenario z = [z_1, z_2, ..., z_N] where each z_i is a state choice
    for descriptor i. Internally, states are represented as 0-based indices,
    but the API uses state labels (strings) for user-friendliness.

    Attributes:
        _state_indices: Internal 0-based state indices.
        descriptors: List of descriptor names in order.
        _descriptor_states: Dictionary mapping descriptor names to state lists.
    """

    def __init__(
        self,
        state_dict: Union[Dict[str, str], List[int]],
        matrix: CIBMatrix,
    ) -> None:
        """
        Create a scenario from a state dictionary or index list.

        Args:
            state_dict: Either a dictionary mapping descriptor names to
                state labels, or a list of 0-based state indices.
            matrix: CIBMatrix providing descriptor and state definitions.

        Raises:
            ValueError: If state assignments are invalid.
        """
        self.descriptors: List[str] = list(matrix.descriptors.keys())
        self._descriptor_states: Dict[str, List[str]] = matrix.descriptors.copy()
        self._state_indices: List[int] = []

        if isinstance(state_dict, dict):
            for desc in self.descriptors:
                if desc not in state_dict:
                    raise ValueError(
                        f"Missing state assignment for descriptor '{desc}'"
                    )
                state_label = state_dict[desc]
                if state_label not in matrix.descriptors[desc]:
                    raise ValueError(
                        f"Invalid state '{state_label}' for "
                        f"descriptor '{desc}'"
                    )
                state_idx = matrix.descriptors[desc].index(state_label)
                self._state_indices.append(state_idx)
        else:
            if len(state_dict) != len(self.descriptors):
                raise ValueError(
                    f"State list length {len(state_dict)} does not match "
                    f"number of descriptors {len(self.descriptors)}"
                )
            for idx, state_idx in enumerate(state_dict):
                max_idx = matrix.state_counts[idx] - 1
                if state_idx < 0 or state_idx > max_idx:
                    raise ValueError(
                        f"State index {state_idx} out of range [0, {max_idx}] "
                        f"for descriptor '{self.descriptors[idx]}'"
                    )
            self._state_indices = list(state_dict)

    def get_state(self, descriptor: str) -> str:
        """
        Get the state label for a descriptor.

        Args:
            descriptor: Descriptor name.

        Returns:
            State label as a string.

        Raises:
            ValueError: If descriptor is not found.
        """
        if descriptor not in self.descriptors:
            raise ValueError(f"Descriptor '{descriptor}' not found")
        idx = self.descriptors.index(descriptor)
        state_idx = self._state_indices[idx]
        return self._descriptor_states[descriptor][state_idx]

    def get_state_index(self, descriptor: str) -> int:
        """
        Get the 0-based state index for a descriptor.

        Args:
            descriptor: Descriptor name.

        Returns:
            0-based state index.

        Raises:
            ValueError: If descriptor is not found.
        """
        if descriptor not in self.descriptors:
            raise ValueError(f"Descriptor '{descriptor}' not found")
        idx = self.descriptors.index(descriptor)
        return self._state_indices[idx]

    def to_dict(self) -> Dict[str, str]:
        """
        Convert scenario to dictionary mapping.

        Returns:
            Dictionary mapping descriptor names to state labels.
        """
        result: Dict[str, str] = {}
        for desc in self.descriptors:
            result[desc] = self.get_state(desc)
        return result

    def to_indices(self) -> List[int]:
        """
        Convert scenario to index vector.

        Returns:
            List of 0-based state indices.
        """
        return self._state_indices.copy()

    def __eq__(self, other: object) -> bool:
        """
        Check equality with another scenario.

        Args:
            other: Another Scenario object.

        Returns:
            True if scenarios have identical state assignments.
        """
        if not isinstance(other, Scenario):
            return False
        return (
            self.descriptors == other.descriptors
            and self._state_indices == other._state_indices
        )

    def __hash__(self) -> int:
        """
        Compute hash for scenario.

        Returns:
            Hash value for use in sets and dictionaries.
        """
        return hash((tuple(self.descriptors), tuple(self._state_indices)))

    def __repr__(self) -> str:
        """
        Generate string representation of scenario.

        Returns:
            Readable string representation.
        """
        state_dict = self.to_dict()
        return f"Scenario({state_dict})"


class ConsistencyChecker:
    """
    Checks scenario consistency against CIB matrix.

    A scenario is consistent if for each descriptor j, the chosen state z_j
    has the maximum (or equal maximum) impact score among all possible states.
    """

    @staticmethod
    def check_consistency(
        scenario: Scenario,
        matrix: CIBMatrix,
        use_fast: bool = False,
    ) -> bool:
        """
        Verify if a scenario satisfies the consistency condition.

        Args:
            scenario: Scenario to check.
            matrix: CIB matrix containing impact relationships.

        Returns:
            True if scenario is consistent, False otherwise.
        """
        if bool(use_fast):
            try:
                from cib.fast_scoring import FastCIBScorer

                scorer = FastCIBScorer.from_matrix(matrix)
                return bool(scorer.is_consistent(scorer.scenario_to_indices(scenario)))
            except Exception:
                # The reference implementation is used as a safe fallback.
                pass

        for descriptor in matrix.descriptors:
            current_state = scenario.get_state(descriptor)
            balance = matrix.calculate_impact_balance(scenario, descriptor)
            current_score = balance[current_state]

            for state, score in balance.items():
                if not np.isclose(score, current_score) and score > current_score:
                    return False

        return True

    @staticmethod
    def check_consistency_detailed(
        scenario: Scenario, matrix: CIBMatrix
    ) -> Dict[str, Any]:
        """
        Check consistency and return detailed diagnostics.

        Args:
            scenario: Scenario to check.
            matrix: CIB matrix containing impact relationships.

        Returns:
            Dictionary containing consistency status, impact balances,
            and inconsistency details.
        """
        is_consistent = True
        balances: Dict[str, Dict[str, float]] = {}
        inconsistencies: List[Dict[str, Any]] = []

        for descriptor in matrix.descriptors:
            current_state = scenario.get_state(descriptor)
            balance = matrix.calculate_impact_balance(scenario, descriptor)
            balances[descriptor] = balance
            current_score = balance[current_state]

            for state, score in balance.items():
                if not np.isclose(score, current_score) and score > current_score:
                    is_consistent = False
                    inconsistencies.append(
                        {
                            "descriptor": descriptor,
                            "current_state": current_state,
                            "current_score": current_score,
                            "better_state": state,
                            "better_score": score,
                        }
                    )

        return {
            "is_consistent": is_consistent,
            "balances": balances,
            "inconsistencies": inconsistencies,
        }

    @staticmethod
    def find_inconsistent_descriptors(
        scenario: Scenario, matrix: CIBMatrix
    ) -> List[str]:
        """
        Find descriptors that violate consistency condition.

        Args:
            scenario: Scenario to check.
            matrix: CIB matrix containing impact relationships.

        Returns:
            List of descriptor names that are inconsistent.
        """
        inconsistent: List[str] = []
        for descriptor in matrix.descriptors:
            current_state = scenario.get_state(descriptor)
            balance = matrix.calculate_impact_balance(scenario, descriptor)
            current_score = balance[current_state]

            for state, score in balance.items():
                if not np.isclose(score, current_score) and score > current_score:
                    if descriptor not in inconsistent:
                        inconsistent.append(descriptor)
                    break

        return inconsistent


@dataclass
class ImpactBalance:
    """
    Stores impact balance calculations for a scenario.

    The impact balance contains impact scores for all descriptor-state
    combinations, computed from the scenario's current state assignments.

    Attributes:
        balance: Dictionary mapping descriptor names to dictionaries
            mapping state labels to impact scores.
    """

    balance: Dict[str, Dict[str, float]]

    def __init__(self, scenario: Scenario, matrix: CIBMatrix) -> None:
        """
        Compute impact balance for a scenario.

        Args:
            scenario: Scenario containing state assignments.
            matrix: CIB matrix containing impact relationships.
        """
        self.balance = {}
        for descriptor in matrix.descriptors:
            self.balance[descriptor] = matrix.calculate_impact_balance(
                scenario, descriptor
            )

    def get_max_state(self, descriptor: str) -> str:
        """
        Get the state with maximum impact score for a descriptor.

        Args:
            descriptor: Descriptor name.

        Returns:
            State label with maximum impact score. If multiple states
            have the same maximum, returns the first one encountered.

        Raises:
            ValueError: If descriptor is not found.
        """
        if descriptor not in self.balance:
            raise ValueError(f"Descriptor '{descriptor}' not found")

        max_score = float("-inf")
        max_state = None
        for state, score in self.balance[descriptor].items():
            if score > max_score:
                max_score = score
                max_state = state

        if max_state is None:
            raise ValueError(f"No states found for descriptor '{descriptor}'")

        return max_state

    def get_max_states(self) -> Dict[str, str]:
        """
        Get all maximum-impact states for all descriptors.

        Returns:
            Dictionary mapping descriptor names to their maximum-impact
            state labels.
        """
        result: Dict[str, str] = {}
        for descriptor in self.balance:
            result[descriptor] = self.get_max_state(descriptor)
        return result

    def get_score(self, descriptor: str, state: str) -> float:
        """
        Retrieve impact score for a specific descriptor-state combination.

        Args:
            descriptor: Descriptor name.
            state: State label.

        Returns:
            Impact score as a float.

        Raises:
            ValueError: If descriptor or state is not found.
        """
        if descriptor not in self.balance:
            raise ValueError(f"Descriptor '{descriptor}' not found")
        if state not in self.balance[descriptor]:
            raise ValueError(
                f"State '{state}' not found for descriptor '{descriptor}'"
            )

        return self.balance[descriptor][state]
