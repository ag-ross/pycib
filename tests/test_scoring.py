"""
Unit tests for scenario diagnostics and qualitative impact labels.
"""

from __future__ import annotations

import pytest

from cib.core import CIBMatrix, Scenario
from cib.scoring import impact_label, judgment_section_labels, scenario_diagnostics


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


class TestImpactLabel:
    def test_bucketing(self) -> None:
        assert impact_label(-3.0) == "strongly_hindering"
        assert impact_label(-1.0) == "hindering"
        assert impact_label(0.0) == "neutral"
        assert impact_label(1.0) == "promoting"
        assert impact_label(3.0) == "strongly_promoting"

    def test_invalid_thresholds(self) -> None:
        with pytest.raises(ValueError):
            impact_label(0.0, weak_threshold=1.0, strong_threshold=1.0)


class TestScenarioDiagnostics:
    def test_consistent_scenario_has_non_negative_margin(self) -> None:
        m = _simple_matrix()
        s = Scenario({"A": "Low", "B": "Low"}, m)
        d = scenario_diagnostics(s, m)
        assert d.is_consistent is True
        assert d.consistency_margin >= 0.0
        assert d.consistency_margin == 4.0

    def test_inconsistent_scenario_has_negative_margin(self) -> None:
        m = _simple_matrix()
        s = Scenario({"A": "Low", "B": "High"}, m)
        d = scenario_diagnostics(s, m)
        assert d.is_consistent is False
        assert d.consistency_margin < 0.0
        assert len(d.inconsistencies) >= 1

    def test_total_impact_score_present(self) -> None:
        m = _simple_matrix()
        s = Scenario({"A": "High", "B": "High"}, m)
        d = scenario_diagnostics(s, m)
        assert isinstance(d.total_impact_score, float)


class TestJudgmentSectionLabels:
    def test_labels_cover_full_section(self) -> None:
        m = _simple_matrix()
        labels = judgment_section_labels(m, src_desc="A", tgt_desc="B")
        assert len(labels) == 4
        assert labels[("Low", "Low")] == "strongly_promoting"
        assert labels[("Low", "High")] == "strongly_hindering"

