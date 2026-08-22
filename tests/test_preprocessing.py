"""Unit tests for feature preprocessing pipeline."""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import create_preprocessor, get_preprocessed_feature_names


def test_preprocessor_fit_transform():
    df = pd.DataFrame([
        {
            "sex": "F",
            "infection_type": "UTI",
            "organism": "E_coli",
            "age": 45,
            "diabetes": 0,
            "recent_hospitalization_90d": 0,
            "recent_antibiotic_use_90d": 1,
            "num_prior_uti_1yr": 1,
            "catheter_use": 0,
            "immunocompromised": 0,
            "nursing_home_resident": 0,
            "prior_resistant_culture_1yr": 0,
            "creatinine_mg_dl": 0.9,
            "wbc_count_k_ul": 8.5,
            "travel_last_6mo": 0,
            "healthcare_worker": 0,
        },
        {
            "sex": "M",
            "infection_type": "Pyelonephritis",
            "organism": "Klebsiella_pneumoniae",
            "age": 70,
            "diabetes": 1,
            "recent_hospitalization_90d": 1,
            "recent_antibiotic_use_90d": 1,
            "num_prior_uti_1yr": 3,
            "catheter_use": 1,
            "immunocompromised": 1,
            "nursing_home_resident": 1,
            "prior_resistant_culture_1yr": 1,
            "creatinine_mg_dl": 2.1,
            "wbc_count_k_ul": 18.0,
            "travel_last_6mo": 1,
            "healthcare_worker": 1,
        }
    ])

    preprocessor = create_preprocessor()
    X_enc = preprocessor.fit_transform(df)

    assert X_enc.shape[0] == 2
    assert X_enc.shape[1] > 16  # Categoricals expanded to one-hot columns
    assert not np.isnan(X_enc).any()

    feature_names = get_preprocessed_feature_names(preprocessor)
    assert len(feature_names) == X_enc.shape[1]


def test_unknown_category_handling():
    # Fit on standard categories
    df_train = pd.DataFrame([
        {
            "sex": "F",
            "infection_type": "UTI",
            "organism": "E_coli",
            "age": 45,
            "diabetes": 0,
            "recent_hospitalization_90d": 0,
            "recent_antibiotic_use_90d": 0,
            "num_prior_uti_1yr": 0,
            "catheter_use": 0,
            "immunocompromised": 0,
            "nursing_home_resident": 0,
            "prior_resistant_culture_1yr": 0,
            "creatinine_mg_dl": 1.0,
            "wbc_count_k_ul": 9.0,
            "travel_last_6mo": 0,
            "healthcare_worker": 0,
        }
    ])

    preprocessor = create_preprocessor()
    preprocessor.fit(df_train)

    # Test unknown category in transform (handle_unknown='ignore' should produce zeros without throwing)
    df_unknown = pd.DataFrame([
        {
            "sex": "UnknownSex",
            "infection_type": "UnknownInfection",
            "organism": "UnknownPathogen",
            "age": 50,
            "diabetes": 0,
            "recent_hospitalization_90d": 0,
            "recent_antibiotic_use_90d": 0,
            "num_prior_uti_1yr": 0,
            "catheter_use": 0,
            "immunocompromised": 0,
            "nursing_home_resident": 0,
            "prior_resistant_culture_1yr": 0,
            "creatinine_mg_dl": 1.0,
            "wbc_count_k_ul": 9.0,
            "travel_last_6mo": 0,
            "healthcare_worker": 0,
        }
    ])

    X_enc = preprocessor.transform(df_unknown)
    assert X_enc.shape[0] == 1
    assert not np.isnan(X_enc).any()
