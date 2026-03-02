from __future__ import annotations

from careai.mimiccode.concept_runner import REQUIRED_SUBSET_SQL


def test_required_subset_contains_core_dependencies() -> None:
    required = set(REQUIRED_SUBSET_SQL)
    assert "demographics/icustay_hourly.sql" in required
    assert "score/sofa.sql" in required
    assert "medication/vasoactive_agent.sql" in required
    assert "treatment/ventilation.sql" in required
    assert "treatment/crrt.sql" in required

