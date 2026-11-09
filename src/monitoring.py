"""
Pipeline & Model Monitoring, Health Checks, and Alert System (Step 8).
Tracks execution latencies, data quality, model availability, and system health.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import psutil

from src.config import (
    DB_PATH, MODELS_DIR, MONITORING_DIR, FEATURE_SCHEMA_VERSION, TABLE_METADATA, TABLE_QUALITY, TABLE_MODEL_METRICS
)
from src.storage import get_connection, get_database_stats
from src.logger import get_logger

logger = get_logger("Monitoring")

def get_pipeline_history(limit: int = 20) -> pd.DataFrame:
    """
    Retrieves recent pipeline execution runs from SQLite metadata table.
    """
    conn = get_connection()
    has_meta = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (TABLE_METADATA,)).fetchone()
    if not has_meta:
        conn.close()
        return pd.DataFrame()

    query = f"""
        SELECT id, pipeline_name, source_file, rows_read, rows_inserted,
               duplicates_removed, outliers_removed, execution_seconds, status, created_at
        FROM {TABLE_METADATA}
        ORDER BY id DESC
        LIMIT {limit}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_quality_reports(limit: int = 20) -> pd.DataFrame:
    """
    Retrieves recent data quality audit reports.
    """
    conn = get_connection()
    has_q = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (TABLE_QUALITY,)).fetchone()
    if not has_q:
        conn.close()
        return pd.DataFrame()

    query = f"""
        SELECT id, source_file, total_records, completeness_score,
               valid_sessions_rate, price_outlier_rate, status, checked_at
        FROM {TABLE_QUALITY}
        ORDER BY id DESC
        LIMIT {limit}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_model_registry_summary() -> Dict[str, Any]:
    """
    Returns current active models and their latest performance metrics.
    """
    reg_file = MODELS_DIR / "model_registry.json"
    if not reg_file.exists():
        return {"status": "NO_REGISTRY", "models": []}

    try:
        with open(reg_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error reading model registry: {e}")
        return {"status": "ERROR", "error": str(e), "models": []}

def get_system_health() -> Dict[str, Any]:
    """
    Returns comprehensive infrastructure, database, and model artifact health indicators.
    """
    db_stats = get_database_stats()
    
    # Model files check
    models_status = {
        "kmeans_segmentation": (MODELS_DIR / "customer_segmentation_kmeans.pkl").exists(),
        "kmeans_scaler": (MODELS_DIR / "customer_segmentation_scaler.pkl").exists(),
        "churn_random_forest": (MODELS_DIR / "churn_prediction_rf.pkl").exists(),
        "registry_json": (MODELS_DIR / "model_registry.json").exists()
    }

    registry = get_model_registry_summary()
    latest = {}
    for record in registry.get("models", []):
        name = record.get("model_name")
        if name:
            latest[name] = record
    schema_current = all(latest.get(name, {}).get("feature_schema_version") == FEATURE_SCHEMA_VERSION
                         for name in ["customer_segmentation_kmeans", "churn_prediction_rf"])

    # Host resource metrics
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(DB_PATH.parent.resolve()))

    return {
        "database": db_stats,
        "models": models_status,
        "model_schema_current": schema_current,
        "system_resources": {
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_available_gb": round(mem.available / (1024**3), 2),
            "ram_usage_percent": mem.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "disk_usage_percent": disk.percent
        }
    }

def evaluate_system_alerts() -> List[Dict[str, str]]:
    """
    Runs automated health audits and triggers alerts if any anomalies or degradation are detected.
    """
    alerts = []
    health = get_system_health()

    # 1. Database Check
    if not health["database"]["exists"]:
        alerts.append({
            "severity": "CRITICAL",
            "component": "Database",
            "message": f"Analytical SQLite database not found at {DB_PATH}. ETL ingestion required."
        })
    elif health["database"]["total_events"] == 0:
        alerts.append({
            "severity": "WARNING",
            "component": "Database",
            "message": "Events table is currently empty."
        })

    # 2. Model Artifacts Check
    missing_models = [m for m, exists in health["models"].items() if not exists]
    if missing_models:
        alerts.append({
            "severity": "HIGH",
            "component": "ModelRegistry",
            "message": f"Missing trained model artifacts: {', '.join(missing_models)}"
        })

    if not health["model_schema_current"]:
        alerts.append({"severity": "HIGH", "component": "ModelSchema",
                       "message": "Models need retraining for the current feature schema: run --mode ml."})

    # 3. Disk Space Check
    if health["system_resources"]["disk_free_gb"] < 5.0:
        alerts.append({
            "severity": "HIGH",
            "component": "StorageInfrastructure",
            "message": f"Low disk space: Only {health['system_resources']['disk_free_gb']} GB available."
        })

    # Inspect the latest outcome of every source, not merely the last five chunks.
    if health["database"]["exists"] and TABLE_METADATA in health["database"]["tables"]:
        conn = get_connection()
        try:
            latest = conn.execute(f"""SELECT source_file, status FROM {TABLE_METADATA}
                WHERE id IN (SELECT MAX(id) FROM {TABLE_METADATA} GROUP BY source_file)""").fetchall()
        finally:
            conn.close()
        for source, status in latest:
            if status in {"FAILED", "QUALITY_WARNING"}:
                alerts.append({
                    "severity": "HIGH" if status == "FAILED" else "WARNING",
                    "component": "ETLPipeline" if status == "FAILED" else "DataQualityGate",
                    "message": f"{source}: {status}. Inspect raw quality reports; invalid rows are excluded during cleaning."
                })

    # Persist an inspectable health snapshot; this is not a drift detector.
    from datetime import datetime, timezone
    snapshot = {"checked_at": datetime.now(timezone.utc).isoformat(), "health": health, "alerts": alerts}
    temporary = MONITORING_DIR / "latest_health.json.tmp"
    temporary.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    temporary.replace(MONITORING_DIR / "latest_health.json")
    return alerts
