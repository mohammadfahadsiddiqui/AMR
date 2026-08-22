"""Unit tests for patient clinical data validation."""

import pytest
import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import validate_patient_data, DataValidationError


@pytest.fixture
def valid_patient():
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


def test_valid_patient_passes(valid_patient):
    is_valid, errors, cleaned = validate_patient_data(valid_patient)
    assert is_valid is True
    assert len(errors) == 0
    assert cleaned["age"] == 52.0
    assert cleaned["sex"] == "F"
    assert cleaned["organism"] == "E_coli"


def test_missing_feature_detected(valid_patient):
    del valid_patient["age"]
    is_valid, errors, cleaned = validate_patient_data(valid_patient)
    assert is_valid is False
    assert any("age" in err for err in errors)


def test_invalid_categorical_detected(valid_patient):
    valid_patient["organism"] = "Superbug_alien_species"
    is_valid, errors, cleaned = validate_patient_data(valid_patient)
    assert is_valid is False
    assert any("organism" in err for err in errors)


def test_out_of_bound_numeric_detected(valid_patient):
    valid_patient["creatinine_mg_dl"] = 999.0
    is_valid, errors, cleaned = validate_patient_data(valid_patient)
    assert is_valid is False
    assert any("creatinine_mg_dl" in err for err in errors)


def test_invalid_binary_detected(valid_patient):
    valid_patient["diabetes"] = 5
    is_valid, errors, cleaned = validate_patient_data(valid_patient)
    assert is_valid is False
    assert any("diabetes" in err for err in errors)


def test_forbidden_leakage_columns_detected(valid_patient):
    valid_patient["patient_id"] = "P99999"
    valid_patient["Ciprofloxacin_resistance_probability"] = 0.85
    is_valid, errors, cleaned = validate_patient_data(valid_patient)
    assert is_valid is False
    assert any("patient_id" in err for err in errors)
    assert any("Ciprofloxacin_resistance_probability" in err for err in errors)
