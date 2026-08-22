"""SQLite Database Persistence for AMR-Predict.

Stores and queries patient records, multi-antibiotic resistance prediction history,
and generated clinical decision-support reports.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import os
import shutil
from src.config import DATA_DIR, BASE_DIR

def get_db_path() -> Path:
    """Returns database path, switching to /tmp in read-only/serverless environments like Vercel."""
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME") or not os.access(DATA_DIR, os.W_OK):
        tmp_db = Path("/tmp") / "amr_history.db"
        if not tmp_db.exists():
            orig_db = DATA_DIR / "amr_history.db"
            if orig_db.exists():
                try:
                    shutil.copy2(orig_db, tmp_db)
                except Exception:
                    pass
        return tmp_db
    return DATA_DIR / "amr_history.db"


def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite database with Row factory enabled."""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database schema and seeds initial demonstration history if empty."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                age REAL NOT NULL,
                sex TEXT NOT NULL,
                infection_type TEXT NOT NULL,
                organism TEXT NOT NULL,
                creatinine_mg_dl REAL NOT NULL,
                wbc_count_k_ul REAL NOT NULL,
                clinical_factors_json TEXT NOT NULL,
                input_data_json TEXT NOT NULL,
                predictions_json TEXT NOT NULL,
                highest_risk_antibiotic TEXT,
                highest_risk_prob REAL,
                highest_risk_category TEXT,
                execution_time_ms REAL,
                model_version TEXT DEFAULT 'AMR-X v1.0'
            )
        """)
        conn.commit()

        # Check if table is empty; if so, seed demo clinical history
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        if count == 0:
            seed_initial_history(conn)


def generate_prediction_id() -> str:
    """Generates a clinical prediction ID: PRD-YYYYMMDD-XXXX."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:4].upper()
    return f"PRD-{date_str}-{short_uuid}"


def generate_patient_id() -> str:
    """Generates a randomized clinical Patient ID: PT-XXXX."""
    return f"PT-{uuid.uuid4().hex[:4].upper()}"


def save_prediction(patient_data: Dict[str, Any], prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Persists a new prediction event and its results into the database."""
    init_db()

    pred_id = generate_prediction_id()
    patient_id = patient_data.get("patient_id") or generate_patient_id()
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    predictions = prediction_result.get("predictions", [])

    # Identify highest risk antibiotic
    highest_risk_abx = "N/A"
    highest_prob = 0.0
    highest_risk_cat = "Low"

    if predictions:
        sorted_preds = sorted(predictions, key=lambda x: x.get("estimated_resistance_probability", 0), reverse=True)
        top = sorted_preds[0]
        highest_risk_abx = top.get("antibiotic", "N/A")
        highest_prob = top.get("estimated_resistance_probability", 0.0)
        highest_risk_cat = top.get("risk_category", "Low")

    # Group binary clinical exposure factors
    clinical_factors = {
        "diabetes": patient_data.get("diabetes", 0),
        "recent_hospitalization_90d": patient_data.get("recent_hospitalization_90d", 0),
        "recent_antibiotic_use_90d": patient_data.get("recent_antibiotic_use_90d", 0),
        "num_prior_uti_1yr": patient_data.get("num_prior_uti_1yr", 0),
        "catheter_use": patient_data.get("catheter_use", 0),
        "immunocompromised": patient_data.get("immunocompromised", 0),
        "nursing_home_resident": patient_data.get("nursing_home_resident", 0),
        "prior_resistant_culture_1yr": patient_data.get("prior_resistant_culture_1yr", 0),
        "travel_last_6mo": patient_data.get("travel_last_6mo", 0),
        "healthcare_worker": patient_data.get("healthcare_worker", 0),
    }

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                id, patient_id, created_at, age, sex, infection_type, organism,
                creatinine_mg_dl, wbc_count_k_ul, clinical_factors_json,
                input_data_json, predictions_json, highest_risk_antibiotic,
                highest_risk_prob, highest_risk_category, execution_time_ms, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred_id,
            patient_id,
            created_at,
            patient_data.get("age", 0),
            patient_data.get("sex", "F"),
            patient_data.get("infection_type", "UTI"),
            patient_data.get("organism", "E_coli"),
            patient_data.get("creatinine_mg_dl", 1.0),
            patient_data.get("wbc_count_k_ul", 9.0),
            json.dumps(clinical_factors),
            json.dumps(patient_data),
            json.dumps(predictions),
            highest_risk_abx,
            highest_prob,
            highest_risk_cat,
            prediction_result.get("execution_time_ms", 0),
            "AMR-X v1.0",
        ))
        conn.commit()

    return {
        "prediction_id": pred_id,
        "patient_id": patient_id,
        "created_at": created_at,
        "highest_risk_antibiotic": highest_risk_abx,
        "highest_risk_prob": highest_prob,
        "highest_risk_category": highest_risk_cat,
    }


def get_prediction_history(
    search: Optional[str] = None,
    risk_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Fetches a list of historical prediction summaries with optional search and filtering."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()

        query = "SELECT * FROM predictions WHERE 1=1"
        params = []

        if search:
            query += " AND (id LIKE ? OR patient_id LIKE ? OR organism LIKE ? OR infection_type LIKE ?)"
            s_param = f"%{search}%"
            params.extend([s_param, s_param, s_param, s_param])

        if risk_filter and risk_filter.lower() != "all":
            query += " AND LOWER(highest_risk_category) = ?"
            params.append(risk_filter.lower())

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "patient_id": row["patient_id"],
                "created_at": row["created_at"],
                "age": row["age"],
                "sex": row["sex"],
                "infection_type": row["infection_type"],
                "organism": row["organism"],
                "highest_risk_antibiotic": row["highest_risk_antibiotic"],
                "highest_risk_prob": row["highest_risk_prob"],
                "highest_risk_category": row["highest_risk_category"],
                "model_version": row["model_version"],
                "execution_time_ms": row["execution_time_ms"],
            })

        return results


def get_prediction_by_id(prediction_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves full details of a specific prediction event."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
        row = cursor.fetchone()
        if not row:
            return None

        return {
            "id": row["id"],
            "patient_id": row["patient_id"],
            "created_at": row["created_at"],
            "age": row["age"],
            "sex": row["sex"],
            "infection_type": row["infection_type"],
            "organism": row["organism"],
            "creatinine_mg_dl": row["creatinine_mg_dl"],
            "wbc_count_k_ul": row["wbc_count_k_ul"],
            "clinical_factors": json.loads(row["clinical_factors_json"]),
            "input_data": json.loads(row["input_data_json"]),
            "predictions": json.loads(row["predictions_json"]),
            "highest_risk_antibiotic": row["highest_risk_antibiotic"],
            "highest_risk_prob": row["highest_risk_prob"],
            "highest_risk_category": row["highest_risk_category"],
            "execution_time_ms": row["execution_time_ms"],
            "model_version": row["model_version"],
        }


def get_patients_directory() -> List[Dict[str, Any]]:
    """Aggregates historical analyses by patient to form a directory."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                patient_id,
                MAX(created_at) as last_analysis_date,
                COUNT(*) as total_analyses,
                age,
                sex,
                organism as last_organism,
                infection_type as last_infection_type,
                highest_risk_category as last_risk_category
            FROM predictions
            GROUP BY patient_id
            ORDER BY last_analysis_date DESC
        """)
        rows = cursor.fetchall()

        return [{
            "patient_id": r["patient_id"],
            "last_analysis_date": r["last_analysis_date"],
            "total_analyses": r["total_analyses"],
            "age": r["age"],
            "sex": r["sex"],
            "last_organism": r["last_organism"],
            "last_infection_type": r["last_infection_type"],
            "last_risk_category": r["last_risk_category"],
        } for r in rows]


def get_db_analytics() -> Dict[str, Any]:
    """Returns database summary statistics for the dashboard overview."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_preds = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT patient_id) FROM predictions")
        unique_patients = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE LOWER(highest_risk_category) = 'high'")
        high_risk_count = cursor.fetchone()[0]

        return {
            "total_predictions": total_preds,
            "unique_patients": unique_patients,
            "high_risk_count": high_risk_count,
        }


def delete_prediction(prediction_id: str) -> bool:
    """Deletes a prediction record by its ID."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        conn.commit()
        return cursor.rowcount > 0


def seed_initial_history(conn: sqlite3.Connection):
    """Populates realistic initial demonstration analyses for hackathon demonstration."""
    demo_records = [
        {
            "id": "PRD-20260822-A8F1",
            "patient_id": "PT-9412",
            "created_at": "2026-08-22 14:32:10",
            "age": 78,
            "sex": "F",
            "infection_type": "Catheter_Associated_UTI",
            "organism": "Klebsiella_pneumoniae",
            "creatinine_mg_dl": 1.65,
            "wbc_count_k_ul": 14.8,
            "factors": {"diabetes": 1, "recent_hospitalization_90d": 1, "recent_antibiotic_use_90d": 1, "num_prior_uti_1yr": 3, "catheter_use": 1, "immunocompromised": 0, "nursing_home_resident": 1, "prior_resistant_culture_1yr": 1, "travel_last_6mo": 0, "healthcare_worker": 0},
            "highest_abx": "Ceftriaxone",
            "highest_prob": 0.893,
            "highest_cat": "High",
            "time_ms": 32.4
        },
        {
            "id": "PRD-20260822-B3C4",
            "patient_id": "PT-6108",
            "created_at": "2026-08-22 11:15:42",
            "age": 64,
            "sex": "M",
            "infection_type": "Complicated_UTI",
            "organism": "Pseudomonas_aeruginosa",
            "creatinine_mg_dl": 1.95,
            "wbc_count_k_ul": 16.2,
            "factors": {"diabetes": 1, "recent_hospitalization_90d": 1, "recent_antibiotic_use_90d": 1, "num_prior_uti_1yr": 2, "catheter_use": 0, "immunocompromised": 1, "nursing_home_resident": 0, "prior_resistant_culture_1yr": 1, "travel_last_6mo": 1, "healthcare_worker": 0},
            "highest_abx": "Ciprofloxacin",
            "highest_prob": 0.874,
            "highest_cat": "High",
            "time_ms": 29.8
        },
        {
            "id": "PRD-20260821-E9D2",
            "patient_id": "PT-3204",
            "created_at": "2026-08-21 16:45:00",
            "age": 36,
            "sex": "F",
            "infection_type": "Pyelonephritis",
            "organism": "E_coli",
            "creatinine_mg_dl": 1.05,
            "wbc_count_k_ul": 13.5,
            "factors": {"diabetes": 0, "recent_hospitalization_90d": 0, "recent_antibiotic_use_90d": 1, "num_prior_uti_1yr": 1, "catheter_use": 0, "immunocompromised": 0, "nursing_home_resident": 0, "prior_resistant_culture_1yr": 0, "travel_last_6mo": 0, "healthcare_worker": 1},
            "highest_abx": "Trimethoprim-Sulfamethoxazole",
            "highest_prob": 0.542,
            "highest_cat": "Moderate",
            "time_ms": 28.1
        },
        {
            "id": "PRD-20260821-F1A0",
            "patient_id": "PT-1157",
            "created_at": "2026-08-21 09:20:15",
            "age": 24,
            "sex": "F",
            "infection_type": "UTI",
            "organism": "E_coli",
            "creatinine_mg_dl": 0.85,
            "wbc_count_k_ul": 7.2,
            "factors": {"diabetes": 0, "recent_hospitalization_90d": 0, "recent_antibiotic_use_90d": 0, "num_prior_uti_1yr": 0, "catheter_use": 0, "immunocompromised": 0, "nursing_home_resident": 0, "prior_resistant_culture_1yr": 0, "travel_last_6mo": 0, "healthcare_worker": 0},
            "highest_abx": "Nitrofurantoin",
            "highest_prob": 0.182,
            "highest_cat": "Low",
            "time_ms": 26.5
        }
    ]

    cursor = conn.cursor()
    for rec in demo_records:
        # Construct sample predictions list for each target antibiotic
        sample_preds = [
            {"antibiotic": "Ceftriaxone", "estimated_resistance_probability": rec["highest_prob"] if rec["highest_abx"] == "Ceftriaxone" else 0.45, "risk_category": rec["highest_cat"] if rec["highest_abx"] == "Ceftriaxone" else "Moderate", "model_type": "XGBoost", "interpretation_label": "High non-susceptibility risk"},
            {"antibiotic": "Ciprofloxacin", "estimated_resistance_probability": rec["highest_prob"] if rec["highest_abx"] == "Ciprofloxacin" else 0.42, "risk_category": rec["highest_cat"] if rec["highest_abx"] == "Ciprofloxacin" else "Moderate", "model_type": "Random Forest", "interpretation_label": "Elevated resistance likelihood"},
            {"antibiotic": "Trimethoprim-Sulfamethoxazole", "estimated_resistance_probability": rec["highest_prob"] if rec["highest_abx"] == "Trimethoprim-Sulfamethoxazole" else 0.38, "risk_category": rec["highest_cat"] if rec["highest_abx"] == "Trimethoprim-Sulfamethoxazole" else "Moderate", "model_type": "Random Forest", "interpretation_label": "Moderate resistance likelihood"},
            {"antibiotic": "Levofloxacin", "estimated_resistance_probability": 0.35, "risk_category": "Moderate", "model_type": "Random Forest", "interpretation_label": "Moderate resistance likelihood"},
            {"antibiotic": "Amoxicillin-Clavulanate", "estimated_resistance_probability": 0.31, "risk_category": "Low", "model_type": "Random Forest", "interpretation_label": "Lower resistance likelihood"},
            {"antibiotic": "Nitrofurantoin", "estimated_resistance_probability": rec["highest_prob"] if rec["highest_abx"] == "Nitrofurantoin" else 0.22, "risk_category": rec["highest_cat"] if rec["highest_abx"] == "Nitrofurantoin" else "Low", "model_type": "XGBoost", "interpretation_label": "Lower resistance likelihood"},
            {"antibiotic": "Gentamicin", "estimated_resistance_probability": 0.19, "risk_category": "Low", "model_type": "XGBoost", "interpretation_label": "Lower resistance likelihood"},
            {"antibiotic": "Fosfomycin", "estimated_resistance_probability": 0.12, "risk_category": "Low", "model_type": "XGBoost", "interpretation_label": "Lower resistance likelihood"},
        ]

        input_data = {
            "age": rec["age"],
            "sex": rec["sex"],
            "infection_type": rec["infection_type"],
            "organism": rec["organism"],
            "creatinine_mg_dl": rec["creatinine_mg_dl"],
            "wbc_count_k_ul": rec["wbc_count_k_ul"],
            **rec["factors"]
        }

        cursor.execute("""
            INSERT INTO predictions (
                id, patient_id, created_at, age, sex, infection_type, organism,
                creatinine_mg_dl, wbc_count_k_ul, clinical_factors_json,
                input_data_json, predictions_json, highest_risk_antibiotic,
                highest_risk_prob, highest_risk_category, execution_time_ms, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rec["id"],
            rec["patient_id"],
            rec["created_at"],
            rec["age"],
            rec["sex"],
            rec["infection_type"],
            rec["organism"],
            rec["creatinine_mg_dl"],
            rec["wbc_count_k_ul"],
            json.dumps(rec["factors"]),
            json.dumps(input_data),
            json.dumps(sample_preds),
            rec["highest_abx"],
            rec["highest_prob"],
            rec["highest_cat"],
            rec["time_ms"],
            "AMR-X v1.0",
        ))
    conn.commit()
