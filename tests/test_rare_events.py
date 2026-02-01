from __future__ import annotations

from cib.core import CIBMatrix, Scenario
from cib.rare_events import event_rate_diagnostics, near_miss_rate, wilson_interval_from_count


def _simple_matrix() -> CIBMatrix:
    # Same as in scoring tests.
    desc = {"A": ["Low", "High"], "B": ["Low", "High"]}
    m = CIBMatrix(desc)
    m.set_impact("A", "Low", "B", "Low", 2.0)
    m.set_impact("A", "Low", "B", "High", -2.0)
    m.set_impact("A", "High", "B", "Low", -2.0)
    m.set_impact("A", "High", "B", "High", 2.0)
    m.set_impact("B", "Low", "A", "Low", 2.0)
    m.set_impact("B", "Low", "A", "High", -2.0)
    m.set_impact("B", "High", "A", "Low", -2.0)
    m.set_impact("B", "High", "A", "High", 2.0)
    return m


def test_wilson_interval_bounds() -> None:
    ci = wilson_interval_from_count(3, 10, level=0.95)
    assert 0.0 <= ci.lower <= ci.upper <= 1.0


def test_event_rate_under_sampled_flag() -> None:
    d = event_rate_diagnostics(k=1, n=100, min_expected_hits=20.0)
    assert d.is_under_sampled is True


def test_near_miss_rate_runs() -> None:
    m = _simple_matrix()
    scenarios = [
        Scenario({"A": "Low", "B": "Low"}, m),
        Scenario({"A": "High", "B": "High"}, m),
    ]
    r = near_miss_rate(scenarios, m, epsilon=0.0)
    assert 0.0 <= r <= 1.0

