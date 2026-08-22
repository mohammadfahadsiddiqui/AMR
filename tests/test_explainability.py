"""Unit tests for SHAP explainability engine."""

import pytest
import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explain import explain_patient_prediction, get_global_shap_summary
from src.config import ANTIBIOTIC_TARGET_MAP


@pytest.fixture
def test_patient():
    return {
        "age": 52,
        "sex": "F",
        "infection_type": "UTI",
        "organism": "E_coli",
        "diabetes": 0,
        "recent_hospitalization_90d": 1,
        "recent_antibiotic_use_90d": 1,
        "num_prior_uti_1yr": 2,
        "catheter_use": 0,
        "immunocompromised": 0,
        "nursing_home_resident": 0,
        "prior_resistant_culture_1yr": 1,
        "creatinine_mg_dl": 1.15,
        "wbc_count_k_ul": 11.2,
        "travel_last_6mo": 0,
        "healthcare_worker": 0,
    }


def test_explain_patient_prediction(test_patient):
    for abx in ANTIBIOTIC_TARGET_MAP.keys():
        exp = explain_patient_prediction(test_patient, abx)

        assert exp["antibiotic"] == abx
        assert "estimated_resistance_probability" in exp
        assert 0.0 <= exp["estimated_resistance_probability"] <= 1.0
        assert "base_value" in exp
        assert "top_positive_factors" in exp
        assert "top_negative_factors" in exp
        assert "waterfall_features" in exp
        assert "attribution_notice" in exp
        assert "medical_disclaimer" in exp


def test_global_shap_summary():
    summary = get_global_shap_summary()
    assert len(summary) == 8
    for abx in ANTIBIOTIC_TARGET_MAP.keys():
        assert abx in summary
        assert "top_features" in summary[abx]
        assert len(summary[abx]["top_features"]) > 0
