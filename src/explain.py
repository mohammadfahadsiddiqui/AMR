"""SHAP Explainable-AI Engine for AMR-Predict.

Computes local patient-specific and global feature attributions using
shap.TreeExplainer. Clarifies mathematical attribution vs. clinical causality.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import shap
import joblib

from src.config import (
    MODELS_DIR,
    ARTIFACTS_DIR,
    ANTIBIOTIC_TARGET_MAP,
    FEATURE_DISPLAY_NAMES,
    MODEL_ATTRIBUTION_NOTICE,
    MEDICAL_DISCLAIMER,
)
from src.preprocessing import (
    validate_patient_data,
    load_preprocessor,
    get_preprocessed_feature_names,
)

logger = logging.getLogger("AMR_Explain")

# Cached explainers in memory
_EXPLAINER_CACHE: Dict[str, shap.TreeExplainer] = {}
_MODEL_CACHE: Dict[str, Any] = {}
_PREPROCESSOR_CACHE: Optional[Any] = None
_REGISTRY_CACHE: Optional[Dict[str, Any]] = None


def _load_resources() -> Tuple[Any, Dict[str, Any]]:
    """Loads and caches preprocessor and model registry."""
    global _PREPROCESSOR_CACHE, _REGISTRY_CACHE

    if _PREPROCESSOR_CACHE is None:
        prep_path = MODELS_DIR / "preprocessor.joblib"
        if not prep_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {prep_path}")
        _PREPROCESSOR_CACHE = load_preprocessor(str(prep_path))

    if _REGISTRY_CACHE is None:
        reg_path = ARTIFACTS_DIR / "model_registry.json"
        if not reg_path.exists():
            raise FileNotFoundError(f"Model registry artifact not found at {reg_path}")
        with open(reg_path, "r") as f:
            _REGISTRY_CACHE = json.load(f)

    return _PREPROCESSOR_CACHE, _REGISTRY_CACHE


def _get_model_and_explainer(antibiotic: str) -> Tuple[Any, shap.TreeExplainer, str]:
    """Retrieves cached model and TreeExplainer for the specified antibiotic."""
    _, registry = _load_resources()

    if antibiotic not in registry:
        raise KeyError(
            f"Antibiotic '{antibiotic}' not found in registry. "
            f"Available: {list(registry.keys())}"
        )

    meta = registry[antibiotic]
    model_file = meta["model_file"]
    model_type = meta["model_type"]

    if antibiotic not in _MODEL_CACHE:
        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {model_path} not found.")
        _MODEL_CACHE[antibiotic] = joblib.load(model_path)

    model = _MODEL_CACHE[antibiotic]

    if antibiotic not in _EXPLAINER_CACHE:
        try:
            _EXPLAINER_CACHE[antibiotic] = shap.TreeExplainer(model)
        except Exception as e:
            logger.error(f"Error initializing TreeExplainer for {antibiotic}: {e}")
            raise

    explainer = _EXPLAINER_CACHE[antibiotic]
    return model, explainer, model_type


def _format_feature_label(raw_col_name: str, patient_val: Any) -> str:
    """Formats an encoded feature name and raw value into a human-friendly clinical description."""
    if raw_col_name.startswith("sex_"):
        suffix = raw_col_name[len("sex_"):]
        return f"Sex: {suffix}"
    elif raw_col_name.startswith("infection_type_"):
        suffix = raw_col_name[len("infection_type_"):].replace("_", " ")
        return f"Infection Type: {suffix}"
    elif raw_col_name.startswith("organism_"):
        suffix = raw_col_name[len("organism_"):].replace("_", " ")
        return f"Pathogen: {suffix}"

    # Numeric or binary
    display_title = FEATURE_DISPLAY_NAMES.get(raw_col_name, raw_col_name.replace("_", " ").title())
    if patient_val is not None:
        if raw_col_name in ["diabetes", "recent_hospitalization_90d", "recent_antibiotic_use_90d",
                            "catheter_use", "immunocompromised", "nursing_home_resident",
                            "prior_resistant_culture_1yr", "travel_last_6mo", "healthcare_worker"]:
            val_text = "Yes" if int(patient_val) == 1 else "No"
            return f"{display_title} ({val_text})"
        return f"{display_title} = {patient_val}"
    return display_title


def explain_patient_prediction(
    patient_data: Dict[str, Any],
    antibiotic: str,
    top_n: int = 8
) -> Dict[str, Any]:
    """Generates local SHAP explanation for a specific patient and antibiotic.

    Args:
        patient_data: Dictionary of patient clinical inputs.
        antibiotic: Target antibiotic to explain.
        top_n: Number of positive and negative contributing factors to return.

    Returns:
        Structured explanation dictionary with waterfall breakdown and attribution notice.
    """
    preprocessor, registry = _load_resources()
    is_valid, errors, cleaned_data = validate_patient_data(patient_data)
    if not is_valid:
        raise ValueError(f"Patient data validation failed: {'; '.join(errors)}")

    model, explainer, model_type = _get_model_and_explainer(antibiotic)

    # Convert to DataFrame
    df_patient = pd.DataFrame([cleaned_data])
    X_enc = preprocessor.transform(df_patient)
    feature_names = get_preprocessed_feature_names(preprocessor)

    # Compute SHAP values
    shap_vals = explainer.shap_values(X_enc)

    # Handle shape variations
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        # Class 1 (Resistant)
        sv = shap_vals[1][0]
    elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
        sv = shap_vals[0, :, 1]
    elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 2:
        sv = shap_vals[0]
    else:
        sv = np.array(shap_vals).flatten()

    # Base expected value
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1] if len(base_val) > 1 else base_val[0])
    else:
        base_val = float(base_val)

    # Model prediction
    prob_resistant = float(model.predict_proba(X_enc)[0, 1])

    # Decompose into contributions
    contributions = []
    for i, (fname, s_val) in enumerate(zip(feature_names, sv)):
        # Match raw value
        raw_val = None
        if fname in cleaned_data:
            raw_val = cleaned_data[fname]
        else:
            for cat_f in ["sex", "infection_type", "organism"]:
                if fname.startswith(cat_f + "_"):
                    cat_val = fname[len(cat_f) + 1:]
                    raw_val = 1 if cleaned_data.get(cat_f) == cat_val else 0

        contributions.append({
            "feature_key": fname,
            "display_name": _format_feature_label(fname, raw_val),
            "shap_value": round(float(s_val), 4),
            "abs_shap": abs(float(s_val)),
            "direction": "Positive (+Risk)" if s_val > 0 else "Negative (-Risk)" if s_val < 0 else "Neutral",
        })

    # Sort contributions
    positive_factors = [c for c in contributions if c["shap_value"] > 0]
    positive_factors.sort(key=lambda x: x["shap_value"], reverse=True)

    negative_factors = [c for c in contributions if c["shap_value"] < 0]
    negative_factors.sort(key=lambda x: x["shap_value"])  # Most negative first

    all_sorted = sorted(contributions, key=lambda x: x["abs_shap"], reverse=True)

    waterfall_items = all_sorted[:top_n]

    return {
        "antibiotic": antibiotic,
        "model_type": model_type,
        "estimated_resistance_probability": round(prob_resistant, 4),
        "base_value": round(base_val, 4),
        "top_positive_factors": positive_factors[:top_n],
        "top_negative_factors": negative_factors[:top_n],
        "waterfall_features": waterfall_items,
        "attribution_notice": MODEL_ATTRIBUTION_NOTICE,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }


def get_global_shap_summary(antibiotic: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves pre-calculated global SHAP feature importances."""
    shap_path = ARTIFACTS_DIR / "global_shap_summary.json"
    if not shap_path.exists():
        raise FileNotFoundError("Global SHAP summary not found in artifacts.")

    with open(shap_path, "r") as f:
        data = json.load(f)

    if antibiotic:
        if antibiotic not in data:
            raise KeyError(f"Antibiotic '{antibiotic}' not in global SHAP summary.")
        return {
            "antibiotic": antibiotic,
            "global_shap": data[antibiotic],
            "attribution_notice": MODEL_ATTRIBUTION_NOTICE,
        }
    return data
