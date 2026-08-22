"""Unit tests for patient inference engine."""

import pytest
import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import predict_patient
from src.preprocessing import DataValidationError


@pytest.fixture
def test_patient():
    return {
        "age": 60,
        "sex": "M",
        "infection_type": "Complicated_UTI",
        "organism": "Klebsiella_pneumoniae",
        "diabetes": 1,
        "recent_hospitalization_90d": 1,
        "recent_antibiotic_use_90d": 1,
        "num_prior_uti_1yr": 2,
        "catheter_use": 1,
        "immunocompromised": 0,
        "nursing_home_resident": 1,
        "prior_resistant_culture_1yr": 1,
        "creatinine_mg_dl": 1.4,
        "wbc_count_k_ul": 13.2,
        "travel_last_6mo": 0,
        "healthcare_worker": 0,
    }


def test_predict_patient_returns_all_8_antibiotics(test_patient):
    res = predict_patient(test_patient)

    assert res["status"] == "success"
    assert res["total_antibiotics_evaluated"] == 8
    assert len(res["predictions"]) == 8
    assert "execution_time_seconds" in res
    assert res["execution_time_seconds"] >= 0.0

    for p in res["predictions"]:
        assert "antibiotic" in p
        assert "estimated_resistance_probability" in p
        prob = p["estimated_resistance_probability"]
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of bounds for {p['antibiotic']}"
        assert p["risk_category"] in ["Low", "Moderate", "High"]
        assert p["model_type"] in ["Random Forest", "XGBoost"]


def test_predict_patient_invalid_data_raises_error():
    invalid_patient = {
        "age": 150,  # invalid age
        "sex": "F",
    }
    with pytest.raises(DataValidationError):
        predict_patient(invalid_patient)
