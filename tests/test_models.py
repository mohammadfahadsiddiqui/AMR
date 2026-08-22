"""Unit tests for trained models, registry, and probability ranges."""

import json
import pytest
import sys
from pathlib import Path
import joblib

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR, ARTIFACTS_DIR, ANTIBIOTIC_TARGET_MAP


def test_all_models_exist():
    assert MODELS_DIR.exists()
    for abx in ANTIBIOTIC_TARGET_MAP.keys():
        fname = f"{abx.replace('-', '_')}_model.joblib"
        model_path = MODELS_DIR / fname
        assert model_path.exists(), f"Model file missing for {abx}: {model_path}"


def test_preprocessor_exists():
    prep_path = MODELS_DIR / "preprocessor.joblib"
    assert prep_path.exists(), "Preprocessor artifact missing."


def test_registry_validity():
    reg_path = ARTIFACTS_DIR / "model_registry.json"
    assert reg_path.exists(), "Model registry missing."

    with open(reg_path, "r") as f:
        registry = json.load(f)

    assert len(registry) == 8, f"Expected 8 antibiotics in registry, got {len(registry)}"

    for abx, meta in registry.items():
        assert "model_type" in meta
        assert meta["model_type"] in ["Random Forest", "XGBoost"]
        assert "metrics" in meta
        assert "roc_auc" in meta["metrics"]
        assert 0.5 <= meta["metrics"]["roc_auc"] <= 1.0


def test_model_comparison_file():
    comp_path = ARTIFACTS_DIR / "model_comparison.csv"
    assert comp_path.exists()
