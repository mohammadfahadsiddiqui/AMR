"""Evaluation module for AMR-Predict models.

Calculates comprehensive classification metrics, probability calibration metrics,
ROC/PR curves, confusion matrices, and Brier scores.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    downsample_curve_points: int = 50
) -> Dict[str, Any]:
    """Calculates comprehensive classification & calibration metrics for a single model.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probability of resistance (class 1).
        threshold: Decision threshold for discrete classification (default: 0.5).
        downsample_curve_points: Max points to keep for ROC/PR curves to optimize JSON size.

    Returns:
        Structured metrics dictionary.
    """
    # Ensure binary arrays
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    y_pred = (y_prob >= threshold).astype(int)

    # Core metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    brier = float(brier_score_loss(y_true, y_prob))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    # Specificity
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC Curve
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    if len(fpr) > downsample_curve_points:
        indices = np.linspace(0, len(fpr) - 1, downsample_curve_points, dtype=int)
        fpr = fpr[indices]
        tpr = tpr[indices]

    # Precision-Recall Curve
    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_true, y_prob)
    if len(pr_prec) > downsample_curve_points:
        indices = np.linspace(0, len(pr_prec) - 1, downsample_curve_points, dtype=int)
        pr_prec = pr_prec[indices]
        pr_rec = pr_rec[indices]

    # Calibration Curve (Reliability Diagram)
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        calib_data = {
            "prob_true": [float(x) for x in prob_true],
            "prob_pred": [float(x) for x in prob_pred],
        }
    except Exception:
        calib_data = {"prob_true": [], "prob_pred": []}

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "specificity": round(spec, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
        },
        "pr_curve": {
            "precision": [round(float(x), 4) for x in pr_prec],
            "recall": [round(float(x), 4) for x in pr_rec],
        },
        "calibration": calib_data,
        "sample_counts": {
            "total": int(len(y_true)),
            "positive_cases": int(np.sum(y_true)),
            "negative_cases": int(len(y_true) - np.sum(y_true)),
            "prevalence": round(float(np.mean(y_true)), 4),
        }
    }


def compare_models(
    results_list: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Generates a clean comparison DataFrame from multiple model evaluation records.

    Expected columns: antibiotic, model, accuracy, precision, recall, f1, roc_auc, brier_score.
    """
    rows = []
    for r in results_list:
        rows.append({
            "antibiotic": r["antibiotic"],
            "model": r["model"],
            "accuracy": r["metrics"]["accuracy"],
            "precision": r["metrics"]["precision"],
            "recall": r["metrics"]["recall"],
            "f1": r["metrics"]["f1"],
            "roc_auc": r["metrics"]["roc_auc"],
            "brier_score": r["metrics"]["brier_score"],
        })
    df = pd.DataFrame(rows)
    return df
