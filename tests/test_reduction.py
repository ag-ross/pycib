"""
Unit tests for model reduction utilities.
"""

from __future__ import annotations

import numpy as np

from cib.core import CIBMatrix, Scenario
from cib.reduction import map_scenario_to_reduced, reduce_matrix


def test_reduce_matrix_binning_and_aggregation() -> None:
    desc = {"A": ["a0", "a1", "a2"], "B": ["b0", "b1", "b2"]}
    m = CIBMatrix(desc)

    # Impacts are set so that the aggregation can be verified.
    # A -> B
    m.set_impact("A", "a0", "B", "b0", 1.0)
    m.set_impact("A", "a1", "B", "b0", 3.0)
    m.set_impact("A", "a2", "B", "b0", 5.0)

    mapping = {
        "A": {"a0": "low", "a1": "low", "a2": "high"},
        "B": {"b0": "x", "b1": "y", "b2": "y"},
    }
    weights = {"A": {"a0": 1.0, "a1": 1.0, "a2": 1.0}}

    r = reduce_matrix(m, mapping=mapping, aggregation="weighted_mean", weights=weights)
    assert r.descriptors["A"] == ["low", "high"]
    assert r.descriptors["B"] == ["x", "y"]

    # For A=low, B=x, values are mean of [1,3] = 2.
    assert np.isclose(r.get_impact("A", "low", "B", "x"), 2.0)
    # For A=high, B=x, value is [5] = 5.
    assert np.isclose(r.get_impact("A", "high", "B", "x"), 5.0)


def test_map_scenario_to_reduced() -> None:
    desc = {"A": ["a0", "a1"], "B": ["b0", "b1"]}
    m = CIBMatrix(desc)

    mapping = {"A": {"a0": "low", "a1": "high"}, "B": {"b0": "x", "b1": "y"}}
    r = reduce_matrix(m, mapping=mapping, aggregation="mean", weights=None)

    s = Scenario({"A": "a0", "B": "b1"}, m)
    sr = map_scenario_to_reduced(s, reduced_matrix=r, mapping=mapping)
    assert sr.to_dict() == {"A": "low", "B": "y"}

