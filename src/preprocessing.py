"""Data Preprocessing and Validation Module for AMR-Predict.

Constructs reproducible scikit-learn ColumnTransformer pipelines, enforces
strict input validation against data types, allowed categorical domains,
and numerical ranges, and prevents data leakage.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    MODEL_FEATURES,
    VALID_CATEGORIES,
    NUMERICAL_BOUNDS,
    BINARY_FEATURES,
    EXCLUDED_COLUMNS,
)


class DataValidationError(ValueError):
    """Custom exception raised when patient data validation fails."""
    pass


def validate_patient_data(
    data: Dict[str, Any],
    strict: bool = True
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validates raw patient dictionary against clinical feature rules and ranges.

    Args:
        data: Dictionary of input clinical features.
        strict: If True, warns about unrecognized extra keys.

    Returns:
        Tuple of (is_valid, list_of_error_messages, cleaned_data_dict)
    """
    errors: List[str] = []
    cleaned: Dict[str, Any] = {}

    if not isinstance(data, dict):
        return False, ["Input patient data must be a valid dictionary/JSON object."], {}

    # Check for forbidden leakage columns
    for forbidden in EXCLUDED_COLUMNS:
        if forbidden in data:
            errors.append(f"Forbidden column detected (data leakage protection): '{forbidden}'.")

    # Check missing required features
    missing = [f for f in MODEL_FEATURES if f not in data]
    if missing:
        errors.append(f"Missing required clinical features: {', '.join(missing)}.")

    # Validate each present feature
    for feat in MODEL_FEATURES:
        if feat not in data:
            continue

        raw_val = data[feat]

        if raw_val is None or (isinstance(raw_val, str) and raw_val.strip() == ""):
            errors.append(f"Field '{feat}' cannot be null or empty.")
            continue

        # Categorical Validation
        if feat in CATEGORICAL_FEATURES:
            val_str = str(raw_val).strip()
            allowed = VALID_CATEGORIES.get(feat, [])
            if val_str not in allowed:
                errors.append(
                    f"Invalid value '{val_str}' for categorical feature '{feat}'. "
                    f"Allowed values: {', '.join(allowed)}."
                )
            else:
                cleaned[feat] = val_str

        # Binary Feature Validation
        elif feat in BINARY_FEATURES:
            try:
                val_int = int(raw_val)
                if val_int not in (0, 1):
                    errors.append(
                        f"Binary feature '{feat}' must be either 0 or 1, got {raw_val}."
                    )
                else:
                    cleaned[feat] = val_int
            except (ValueError, TypeError):
                errors.append(
                    f"Binary feature '{feat}' must be an integer (0 or 1), got '{raw_val}'."
                )

        # Numerical Continuous Feature Validation
        elif feat in NUMERICAL_BOUNDS:
            try:
                val_num = float(raw_val)
                bounds = NUMERICAL_BOUNDS[feat]
                min_b = bounds["min"]
                max_b = bounds["max"]

                if np.isnan(val_num) or np.isinf(val_num):
                    errors.append(f"Numerical feature '{feat}' must be a finite number.")
                elif val_num < min_b or val_num > max_b:
                    errors.append(
                        f"Feature '{feat}' value {val_num} out of physiological range [{min_b}, {max_b}]."
                    )
                else:
                    cleaned[feat] = val_num
            except (ValueError, TypeError):
                errors.append(
                    f"Numerical feature '{feat}' must be numeric, got '{raw_val}'."
                )

    is_valid = len(errors) == 0
    return is_valid, errors, cleaned


def create_preprocessor() -> ColumnTransformer:
    """Creates a scikit-learn ColumnTransformer for feature encoding.

    - Categorical features: OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    - Numeric features: StandardScaler()
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )

    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numeric_transformer, NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_preprocessed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extracts human-readable feature column names from a fitted preprocessor."""
    try:
        names = preprocessor.get_feature_names_out().tolist()
        return [str(n) for n in names]
    except Exception:
        # Fallback manual derivation if get_feature_names_out fails
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        return list(cat_names) + list(NUMERIC_FEATURES)


def save_preprocessor(preprocessor: ColumnTransformer, filepath: str) -> None:
    """Saves fitted preprocessor to disk."""
    joblib.dump(preprocessor, filepath)


def load_preprocessor(filepath: str) -> ColumnTransformer:
    """Loads fitted preprocessor from disk."""
    return joblib.load(filepath)
