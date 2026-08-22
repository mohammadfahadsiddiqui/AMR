"""Integration tests for FastAPI endpoints."""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

client = TestClient(app)


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["models_ready"] is True


def test_api_config():
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    assert "presets" in data
    assert len(data["presets"]) >= 3
    assert "antibiotics" in data
    assert len(data["antibiotics"]) == 8


def test_api_predict_success():
    payload = {
        "age": 55,
        "sex": "F",
        "infection_type": "UTI",
        "organism": "E_coli",
        "diabetes": 0,
        "recent_hospitalization_90d": 1,
        "recent_antibiotic_use_90d": 1,
        "num_prior_uti_1yr": 1,
        "catheter_use": 0,
        "immunocompromised": 0,
        "nursing_home_resident": 0,
        "prior_resistant_culture_1yr": 0,
        "creatinine_mg_dl": 1.0,
        "wbc_count_k_ul": 9.5,
        "travel_last_6mo": 0,
        "healthcare_worker": 0,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert len(data["predictions"]) == 8
    assert "execution_time_seconds" in data


def test_api_explain_success():
    payload = {
        "patient_data": {
            "age": 55,
            "sex": "F",
            "infection_type": "UTI",
            "organism": "E_coli",
            "diabetes": 0,
            "recent_hospitalization_90d": 1,
            "recent_antibiotic_use_90d": 1,
            "num_prior_uti_1yr": 1,
            "catheter_use": 0,
            "immunocompromised": 0,
            "nursing_home_resident": 0,
            "prior_resistant_culture_1yr": 0,
            "creatinine_mg_dl": 1.0,
            "wbc_count_k_ul": 9.5,
            "travel_last_6mo": 0,
            "healthcare_worker": 0,
        },
        "antibiotic": "Ciprofloxacin"
    }
    res = client.post("/api/explain", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["antibiotic"] == "Ciprofloxacin"
    assert "waterfall_features" in data


def test_api_models():
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "registry" in data
    assert "detailed_metrics" in data


def test_api_eda():
    res = client.get("/api/eda")
    assert res.status_code == 200
    data = res.json()
    assert "resistance_prevalence" in data
    assert "organism_distribution" in data
