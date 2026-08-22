"""Inference Engine for AMR-Predict.

Executes real-time multi-antibiotic resistance prediction, enforces data validation,
measures high-precision inference latency, categorizes prototype risk levels,
and returns structured results for all 8 target antibiotics.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import joblib

from src.config import (
    MODELS_DIR,
    ARTIFACTS_DIR,
    ANTIBIOTIC_TARGET_MAP,
    get_risk_category,
    RISK_THRESHOLDS,
    MEDICAL_DISCLAIMER,
)
from src.preprocessing import (
    validate_patient_data,
    load_preprocessor,
    DataValidationError,
)

logger = logging.getLogger("AMR_Predict")

# In-memory cached artifacts
_CACHED_PREPROCESSOR: Optional[Any] = None
_CACHED_REGISTRY: Optional[Dict[str, Any]] = None
_CACHED_MODELS: Dict[str, Any] = {}


def load_inference_artifacts() -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Loads and caches in memory the preprocessor, model registry, and all 8 trained models."""
    global _CACHED_PREPROCESSOR, _CACHED_REGISTRY, _CACHED_MODELS

    if _CACHED_PREPROCESSOR is None:
        prep_path = MODELS_DIR / "preprocessor.joblib"
        if not prep_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {prep_path}. Please run train.py first.")
        _CACHED_PREPROCESSOR = load_preprocessor(str(prep_path))

    if _CACHED_REGISTRY is None:
        reg_path = ARTIFACTS_DIR / "model_registry.json"
        if not reg_path.exists():
            raise FileNotFoundError(f"Model registry not found at {reg_path}. Please run train.py first.")
        with open(reg_path, "r") as f:
            _CACHED_REGISTRY = json.load(f)

    # Pre-load all registered models into memory for sub-second inference
    for abx, meta in _CACHED_REGISTRY.items():
        if abx not in _CACHED_MODELS:
            m_path = MODELS_DIR / meta["model_file"]
            if not m_path.exists():
                raise FileNotFoundError(f"Model file {m_path} missing.")
            _CACHED_MODELS[abx] = joblib.load(m_path)

    return _CACHED_PREPROCESSOR, _CACHED_REGISTRY, _CACHED_MODELS


def predict_patient(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Executes resistance probability prediction across all 8 target antibiotics.

    Args:
        patient_data: Dictionary of raw clinical inputs.

    Returns:
        Structured prediction output with probabilities, risk tiers, latency, and disclaimers.

    Raises:
        DataValidationError: If input validation fails.
    """
    start_time = time.perf_counter()

    # 1. Validate Input Data
    is_valid, errors, cleaned_data = validate_patient_data(patient_data)
    if not is_valid:
        raise DataValidationError(f"Validation failed: {'; '.join(errors)}")

    # 2. Load Pipeline Artifacts
    preprocessor, registry, models = load_inference_artifacts()

    # 3. Transform Features
    df_patient = pd.DataFrame([cleaned_data])
    X_enc = preprocessor.transform(df_patient)

    # 4. Run Multi-Model Predictions
    predictions: List[Dict[str, Any]] = []

    for abx, meta in registry.items():
        model = models[abx]
        model_type = meta["model_type"]
        target_col = meta["target"]

        # Raw probability estimate for class 1 (Resistant)
        raw_prob = float(model.predict_proba(X_enc)[0, 1])
        bounded_prob = max(0.0, min(1.0, raw_prob))

        risk_tier = get_risk_category(bounded_prob)

        predictions.append({
            "antibiotic": abx,
            "target": target_col,
            "estimated_resistance_probability": round(bounded_prob, 4),
            "percentage_display": f"{bounded_prob * 100:.1f}%",
            "risk_category": risk_tier,
            "model_type": model_type,
            "model_selection_score": meta.get("selection_score", 0.0),
            "interpretation_label": (
                "Lower model-estimated resistance probability"
                if risk_tier == "Low"
                else "Moderate model-estimated resistance probability"
                if risk_tier == "Moderate"
                else "Elevated model-estimated resistance probability"
            ),
        })

    # Sort predictions by estimated resistance probability descending (highest risk first)
    predictions_sorted = sorted(
        predictions,
        key=lambda x: x["estimated_resistance_probability"],
        reverse=True
    )

    elapsed_seconds = time.perf_counter() - start_time
    elapsed_ms = elapsed_seconds * 1000.0

    return {
        "status": "success",
        "patient_inputs": cleaned_data,
        "predictions": predictions_sorted,
        "total_antibiotics_evaluated": len(predictions_sorted),
        "execution_time_seconds": round(elapsed_seconds, 4),
        "execution_time_ms": round(elapsed_ms, 2),
        "execution_time_display": f"Prediction generated in {elapsed_seconds:.3f} seconds ({elapsed_ms:.1f} ms)",
        "risk_thresholds": RISK_THRESHOLDS,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }


if __name__ == "__main__":
    # Quick CLI Test with sample patient profile
    sample_patient = {
        "age": 52,
        "sex": "F",
        "infection_type": "UTI",
        "organism": "E_coli",
        "diabetes": 1,
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
    result = predict_patient(sample_patient)
    print(f"\n{result['execution_time_display']}\n")
    for p in result["predictions"]:
        print(f"  {p['antibiotic']:32} | Prob: {p['percentage_display']:>6} | Risk: {p['risk_category']:<8} | Model: {p['model_type']}")
