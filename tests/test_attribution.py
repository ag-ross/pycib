from __future__ import annotations

from cib.attribution import attribute_scenario, flip_candidates_for_descriptor
from cib.core import CIBMatrix, Scenario


def _simple_matrix() -> CIBMatrix:
    # Two-descriptor coordination system with fixed points at (Low,Low) and (High,High).
    desc = {"A": ["Low", "High"], "B": ["Low", "High"]}
    m = CIBMatrix(desc)
    # A influences B.
    m.set_impact("A", "Low", "B", "Low", 2.0)
    m.set_impact("A", "Low", "B", "High", -2.0)
    m.set_impact("A", "High", "B", "Low", -2.0)
    m.set_impact("A", "High", "B", "High", 2.0)
    # B influences A.
    m.set_impact("B", "Low", "A", "Low", 2.0)
    m.set_impact("B", "Low", "A", "High", -2.0)
    m.set_impact("B", "High", "A", "Low", -2.0)
    m.set_impact("B", "High", "A", "High", 2.0)
    return m


def test_attribute_scenario_margin_and_contribution() -> None:
    m = _simple_matrix()
    s = Scenario({"A": "Low", "B": "Low"}, m)
    a = attribute_scenario(s, m)
    by = a.by_descriptor()

    aA = by["A"]
    assert aA.chosen_state == "Low"
    assert aA.alternative_state == "High"
    assert aA.margin_to_switch == 4.0
    assert len(aA.contributions) == 1
    c = aA.contributions[0]
    assert c.src_descriptor == "B"
    assert c.src_state == "Low"
    assert c.delta == 4.0


def test_flip_candidates_include_feasible_cell_change() -> None:
    m = _simple_matrix()
    s = Scenario({"A": "Low", "B": "Low"}, m)
    a = attribute_scenario(s, m)
    by = a.by_descriptor()
    aA = by["A"]

    flips = flip_candidates_for_descriptor(aA, k=10)
    assert len(flips) > 0
    assert any(f.feasible_under_clip for f in flips)

