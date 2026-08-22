"""Training and Model Selection Pipeline for AMR-Predict.

Executes reproducible training across all 8 target antibiotics, evaluates
both Random Forest and Gradient Boosting candidates, performs probability calibration,
computes global feature importance summaries, and registers the optimal models.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

from src.config import (
    DATASET_PATH,
    MODELS_DIR,
    ARTIFACTS_DIR,
    ANTIBIOTIC_TARGET_MAP,
    MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    EXCLUDED_COLUMNS,
    RISK_THRESHOLDS,
    MEDICAL_DISCLAIMER,
)
from src.preprocessing import (
    create_preprocessor,
    get_preprocessed_feature_names,
    save_preprocessor,
)
from src.evaluate import evaluate_binary_classifier, compare_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("AMR_Train")


def compute_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates comprehensive dataset statistics for exploratory analysis UI."""
    logger.info("Computing Exploratory Data Analysis (EDA) summary...")
    n_total = len(df)

    # Age distribution
    age_hist, age_bin_edges = np.histogram(df["age"], bins=10)
    age_bins = [
        {"range": f"{int(age_bin_edges[i])}-{int(age_bin_edges[i+1])}", "count": int(age_hist[i])}
        for i in range(len(age_hist))
    ]

    # Categorical distributions
    sex_dist = df["sex"].value_counts().to_dict()
    inf_dist = df["infection_type"].value_counts().to_dict()
    org_dist = df["organism"].value_counts().to_dict()

    # Binary clinical factors prevalence
    binary_factors = {
        "Diabetes Mellitus": float(df["diabetes"].mean()),
        "Hospitalization (Last 90d)": float(df["recent_hospitalization_90d"].mean()),
        "Antibiotic Exposure (Last 90d)": float(df["recent_antibiotic_use_90d"].mean()),
        "Catheter Use": float(df["catheter_use"].mean()),
        "Immunocompromised": float(df["immunocompromised"].mean()),
        "Nursing Home Resident": float(df["nursing_home_resident"].mean()),
        "Prior Resistant Culture (1yr)": float(df["prior_resistant_culture_1yr"].mean()),
        "Recent Travel (6mo)": float(df["travel_last_6mo"].mean()),
        "Healthcare Worker": float(df["healthcare_worker"].mean()),
    }

    # Resistance prevalence per antibiotic
    resistance_prevalence = {}
    for abx, col in ANTIBIOTIC_TARGET_MAP.items():
        if col in df.columns:
            resistance_prevalence[abx] = {
                "resistant_count": int(df[col].sum()),
                "total_count": n_total,
                "resistance_rate": round(float(df[col].mean()), 4),
            }

    eda = {
        "dataset_metadata": {
            "total_records": n_total,
            "total_columns": len(df.columns),
            "synthetic_dataset": True,
            "clinical_validation": False,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        },
        "distributions": {
            "age": age_bins,
            "sex": sex_dist,
            "infection_type": inf_dist,
            "organism": org_dist,
        },
        "age_distribution": age_bins,
        "sex_distribution": sex_dist,
        "infection_type_distribution": inf_dist,
        "organism_distribution": org_dist,
        "clinical_factors_prevalence": binary_factors,
        "resistance_prevalence": resistance_prevalence,
    }
    return eda


def train_pipeline():
    """End-to-end reproducible training, benchmarking, and serialization pipeline."""
    logger.info("=" * 65)
    logger.info("Starting AMR-Predict Machine Learning Pipeline")
    logger.info("=" * 65)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"AMR Dataset not found at: {DATASET_PATH}")

    logger.info(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Verify input features exist
    for f in MODEL_FEATURES:
        if f not in df.columns:
            raise KeyError(f"Required model input feature '{f}' missing from dataset.")

    # 2. Compute and Save EDA Summary
    eda_data = compute_eda_summary(df)
    with open(ARTIFACTS_DIR / "eda_summary.json", "w") as f:
        json.dump(eda_data, f, indent=2)
    logger.info("Saved EDA summary to artifacts/eda_summary.json")

    # 3. Build & Fit Column Preprocessor on Entire Input Matrix
    X = df[MODEL_FEATURES].copy()
    preprocessor = create_preprocessor()
    preprocessor.fit(X)
    save_preprocessor(preprocessor, str(MODELS_DIR / "preprocessor.joblib"))

    transformed_feature_names = get_preprocessed_feature_names(preprocessor)
    logger.info(f"Preprocessor fitted. Total encoded features: {len(transformed_feature_names)}")

    feature_config = {
        "raw_features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "encoded_feature_names": transformed_feature_names,
        "excluded_columns": EXCLUDED_COLUMNS,
        "risk_thresholds": RISK_THRESHOLDS,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }
    with open(ARTIFACTS_DIR / "feature_config.json", "w") as f:
        json.dump(feature_config, f, indent=2)

    # 4. Antibiotic-by-Antibiotic Candidate Model Training & Comparison
    all_evaluation_records: List[Dict[str, Any]] = []
    model_registry: Dict[str, Any] = {}
    detailed_metrics: Dict[str, Any] = {}
    global_shap_summary: Dict[str, Any] = {}

    for antibiotic, target_col in ANTIBIOTIC_TARGET_MAP.items():
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found in dataset. Skipping {antibiotic}.")
            continue

        logger.info("-" * 55)
        logger.info(f"Processing Target: {antibiotic} ({target_col})")
        y = df[target_col].values.astype(int)
        pos_prevalence = float(np.mean(y))
        logger.info(f"  Target prevalence: {pos_prevalence:.1%}")

        # Stratified Train/Test Split (test_size = 0.20, random_state = 42)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Preprocess features
        X_train = preprocessor.transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        # Model Candidate 1: Random Forest
        rf_clf = RandomForestClassifier(
            n_estimators=80,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_clf.fit(X_train, y_train)
        rf_prob = rf_clf.predict_proba(X_test)[:, 1]
        rf_metrics = evaluate_binary_classifier(y_test, rf_prob)

        all_evaluation_records.append({
            "antibiotic": antibiotic,
            "model": "Random Forest",
            "metrics": rf_metrics,
        })
        logger.info(
            f"  [Random Forest] ROC-AUC: {rf_metrics['roc_auc']:.3f} | "
            f"F1: {rf_metrics['f1']:.3f} | Recall: {rf_metrics['recall']:.3f} | "
            f"Precision: {rf_metrics['precision']:.3f} | Brier: {rf_metrics['brier_score']:.3f}"
        )

        # Model Candidate 2: Gradient Boosting
        gb_clf = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            random_state=42,
        )
        gb_clf.fit(X_train, y_train)
        gb_prob = gb_clf.predict_proba(X_test)[:, 1]
        gb_metrics = evaluate_binary_classifier(y_test, gb_prob)

        all_evaluation_records.append({
            "antibiotic": antibiotic,
            "model": "Gradient Boosting",
            "metrics": gb_metrics,
        })
        logger.info(
            f"  [Gradient Boosting] ROC-AUC: {gb_metrics['roc_auc']:.3f} | "
            f"F1: {gb_metrics['f1']:.3f} | Recall: {gb_metrics['recall']:.3f} | "
            f"Precision: {gb_metrics['precision']:.3f} | Brier: {gb_metrics['brier_score']:.3f}"
        )

        # Selection composite score: 0.45 * ROC-AUC + 0.35 * F1 + 0.20 * Recall
        rf_score = (
            0.45 * rf_metrics["roc_auc"] +
            0.35 * rf_metrics["f1"] +
            0.20 * rf_metrics["recall"]
        )
        gb_score = (
            0.45 * gb_metrics["roc_auc"] +
            0.35 * gb_metrics["f1"] +
            0.20 * gb_metrics["recall"]
        )

        selected_type = "Random Forest"
        selected_model = rf_clf
        selected_metrics = rf_metrics

        logger.info(
            f"  --> Selected Best Model for {antibiotic}: {selected_type} "
            f"(Score: {rf_score:.3f})"
        )

        # 6. Probability Calibration
        calibrated_model = CalibratedClassifierCV(
            estimator=selected_model,
            method="sigmoid",
            cv=3
        )
        calibrated_model.fit(X_train, y_train)
        calib_prob = calibrated_model.predict_proba(X_test)[:, 1]
        calib_metrics = evaluate_binary_classifier(y_test, calib_prob)

        # Save selected model to disk with joblib compression (level 3)
        model_filename = f"{antibiotic.replace('-', '_')}_model.joblib"
        model_save_path = MODELS_DIR / model_filename
        joblib.dump(selected_model, model_save_path, compress=3)

        # Register in model registry
        model_registry[antibiotic] = {
            "model_type": selected_type,
            "target": target_col,
            "model_file": model_filename,
            "selection_score": round(float(max(rf_score, gb_score)), 4),
            "metrics": selected_metrics,
            "calibration_metrics": calib_metrics,
            "prevalence": round(pos_prevalence, 4),
        }

        detailed_metrics[antibiotic] = {
            "selected_model": selected_type,
            "candidates": {
                "Random Forest": rf_metrics,
                "Gradient Boosting": gb_metrics,
            },
            "calibrated": calib_metrics,
        }

        # 7. Global feature importance
        importances = getattr(selected_model, "feature_importances_", None)
        if importances is None:
            importances = np.ones(len(transformed_feature_names)) / len(transformed_feature_names)

        feature_importance_list = [
            {"feature": fname, "mean_shap": round(float(m), 4)}
            for fname, m in zip(transformed_feature_names, importances)
        ]
        feature_importance_list.sort(key=lambda x: x["mean_shap"], reverse=True)

        global_shap_summary[antibiotic] = {
            "top_features": feature_importance_list[:12],
            "expected_value": float(pos_prevalence),
        }

    # 8. Save Final Model Comparison Table and Registry Files
    comparison_df = compare_models(all_evaluation_records)
    comparison_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
    logger.info("Saved model comparison to artifacts/model_comparison.csv")

    with open(ARTIFACTS_DIR / "model_registry.json", "w") as f:
        json.dump(model_registry, f, indent=2)
    logger.info("Saved model registry to artifacts/model_registry.json")

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(detailed_metrics, f, indent=2)
    logger.info("Saved detailed metrics to artifacts/metrics.json")

    with open(ARTIFACTS_DIR / "global_shap_summary.json", "w") as f:
        json.dump(global_shap_summary, f, indent=2)
    logger.info("Saved global SHAP summary to artifacts/global_shap_summary.json")

    logger.info("=" * 65)
    logger.info("AMR-Predict Model Training Pipeline Complete!")
    logger.info("=" * 65)


if __name__ == "__main__":
    train_pipeline()
