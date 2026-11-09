"""
Machine Learning Training, Evaluation, and Model Registry.
Contains Customer Segmentation (K-Means), Churn Prediction (Random Forest),
and Hybrid Product Recommendation engines with full versioning support.
"""

import json
import pickle
import time
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_SCHEMA_VERSION, MODELS_DIR, RANDOM_STATE, KMEANS_N_CLUSTERS, DB_PATH, TABLE_MODEL_METRICS, TABLE_EVENTS
)
from src.features import CLUSTERING_FEATURES, CHURN_FEATURES
from src.storage import get_connection
from src.logger import get_logger

logger = get_logger("MLModels")

def _save_artifact(model, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(model, handle)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ModelRegistry:
    """
    Manages model registration, tracking versions, training metrics, and artifact locations.
    """
    def __init__(self, registry_path: Path = MODELS_DIR / "model_registry.json"):
        self.registry_path = registry_path
        self._ensure_registry()

    def _ensure_registry(self):
        if not self.registry_path.exists():
            default_data = {"version": "1.0", "updated_at": datetime.now().isoformat(), "models": []}
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(default_data, f, indent=2)

    def register_model(
        self,
        model_name: str,
        version: str,
        algorithm: str,
        metrics: Dict[str, Any],
        artifact_path: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registers model record in JSON registry file and SQLite TABLE_MODEL_METRICS.
        """
        version = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        archive_dir = MODELS_DIR / "versions" / version
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived = archive_dir / Path(artifact_path).name
        shutil.copy2(artifact_path, archived)
        if model_name == "customer_segmentation_kmeans":
            shutil.copy2(MODELS_DIR / "customer_segmentation_scaler.pkl", archive_dir)
        entry = {
            "model_name": model_name,
            "version": version,
            "algorithm": algorithm,
            "metrics": metrics,
            "parameters": parameters or {},
            "artifact_path": str(archived),
            "active_artifact_path": str(artifact_path),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "trained_at": datetime.now().isoformat()
        }

        # Update JSON file
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            data["models"] = [m for m in data.get("models", []) if m.get("model_name") != model_name]
            data["models"].append(entry)
            data["updated_at"] = datetime.now().isoformat()

            temporary = self.registry_path.with_suffix(".json.tmp")
            with open(temporary, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(temporary, self.registry_path)
        except Exception as e:
            raise RuntimeError("Failed to update model registry") from e

        # Update SQLite table
        try:
            conn = get_connection()
            conn.execute(f"""
                INSERT INTO {TABLE_MODEL_METRICS} (
                    model_name, version, algorithm, metrics_json, artifact_path
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                model_name, version, algorithm, json.dumps(metrics), str(archived)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            raise RuntimeError("Failed to persist model metrics") from e

        logger.info(f"Model registered successfully: {model_name} (v{version}) with metrics: {metrics}")


def train_customer_segmentation(
    user_features: pd.DataFrame,
    n_clusters: int = KMEANS_N_CLUSTERS,
    version: str = "1.0.0"
) -> Tuple[KMeans, StandardScaler, float, pd.DataFrame]:
    """
    Trains K-Means clustering for customer segmentation.
    Returns (model, scaler, silhouette_score, cluster_summary)
    """
    logger.info("--- Training Customer Segmentation Model (K-Means) ---")
    start_time = time.time()

    if len(user_features) <= n_clusters or len(user_features[CLUSTERING_FEATURES].drop_duplicates()) < n_clusters:
        raise ValueError(f"Segmentation requires more than {n_clusters} users and at least {n_clusters} distinct feature vectors")
    sample_size = min(100_000, len(user_features))
    sample_df = user_features.sample(n=sample_size, random_state=RANDOM_STATE).copy()
    
    X = sample_df[CLUSTERING_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    sample_df["cluster"] = kmeans.fit_predict(X_scaled)

    # Compute silhouette on sample for validation
    eval_sample_size = min(5000, len(X_scaled))
    labels = sample_df["cluster"].iloc[:eval_sample_size]
    score = silhouette_score(X_scaled[:eval_sample_size], labels) if 1 < labels.nunique() < len(labels) else 0.0

    # Cluster summary
    summary = sample_df.groupby("cluster").agg({
        "user_id": "count",
        "total_spent": "mean",
        "total_events": "mean",
        "events_per_day": "mean"
    }).round(2).reset_index()
    # Names reflect learned statistics, never arbitrary K-Means label numbers.
    personas = {}
    for row in summary.itertuples():
        personas[int(row.cluster)] = {
            "name": f"Segment {row.cluster}: avg purchases ${row.total_spent:.2f}, {row.total_events:.0f} events",
            "strategy": "Retention and relevant product recommendations" if row.total_spent > 0 else "Product discovery and first-purchase offers"
        }
    kmeans.personas_ = personas
    kmeans.feature_schema_version_ = FEATURE_SCHEMA_VERSION
    scaler.feature_schema_version_ = FEATURE_SCHEMA_VERSION
    summary["persona"] = summary["cluster"].map(lambda c: personas[c]["name"])

    # Save artifacts
    model_path = MODELS_DIR / "customer_segmentation_kmeans.pkl"
    scaler_path = MODELS_DIR / "customer_segmentation_scaler.pkl"
    _save_artifact(kmeans, model_path)
    _save_artifact(scaler, scaler_path)

    # Register model
    registry = ModelRegistry()
    registry.register_model(
        model_name="customer_segmentation_kmeans",
        version=version,
        algorithm="KMeans",
        metrics={"silhouette_score": round(float(score), 4), "training_samples": sample_size},
        artifact_path=str(model_path),
        parameters={"n_clusters": n_clusters, "features": CLUSTERING_FEATURES}
    )

    logger.info(f"Segmentation training completed in {time.time() - start_time:.2f}s. Silhouette: {score:.4f}")
    return kmeans, scaler, score, summary


def train_churn_prediction(
    user_features: pd.DataFrame,
    version: str = "1.0.0"
) -> Tuple[RandomForestClassifier, Dict[str, float], pd.DataFrame]:
    """
    Trains Random Forest Classifier to predict customer churn.
    Returns (model, metrics_dict, feature_importance_df)
    """
    logger.info("--- Training Churn Prediction Model (Random Forest) ---")
    start_time = time.time()

    X = user_features[CHURN_FEATURES].fillna(0)
    y = user_features["churned"]
    if user_features.attrs.get("evaluation", "").startswith("held-out users") is False:
        raise ValueError("Churn training requires extract_user_features(for_churn=True) to prevent label leakage")
    if len(y) < 10 or y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Churn training needs at least 10 users and at least two users in each class")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
    }

    probabilities = rf_model.predict_proba(X_test)[:, list(rf_model.classes_).index(1)]
    baseline = np.full(len(y_test), int(y_train.mode().iloc[0]))
    metrics.update({
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
        "baseline_accuracy": round(float(accuracy_score(y_test, baseline)), 4),
        "baseline_f1": round(float(f1_score(y_test, baseline, zero_division=0)), 4),
        "training_samples": len(X_train), "test_samples": len(X_test)
    })
    rf_model.feature_schema_version_ = FEATURE_SCHEMA_VERSION
    rf_model.evaluation_ = dict(user_features.attrs)

    # Feature Importance
    importance_df = pd.DataFrame({
        "feature": CHURN_FEATURES,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Save artifact
    model_path = MODELS_DIR / "churn_prediction_rf.pkl"
    _save_artifact(rf_model, model_path)

    # Register model
    registry = ModelRegistry()
    registry.register_model(
        model_name="churn_prediction_rf",
        version=version,
        algorithm="RandomForestClassifier",
        metrics=metrics,
        artifact_path=str(model_path),
        parameters={"n_estimators": 50, "max_depth": 8, "features": CHURN_FEATURES, **user_features.attrs}
    )

    logger.info(f"Churn training completed in {time.time() - start_time:.2f}s. F1-Score: {metrics['f1']:.4f}")
    return rf_model, metrics, importance_df


def compute_recommendations(
    db_path: Path = DB_PATH,
    top_n_per_category: int = 10
) -> pd.DataFrame:
    """
    Computes purchase-frequency recommendations by category.
    Stores recommendations into SQLite for low-latency serving.
    """
    logger.info("Computing product recommendation tables...")
    conn = get_connection(db_path)

    # 1. Popularity per category
    pop_query = f"""
    SELECT 
        category_level1,
        product_id,
        category_code,
        brand,
        ROUND(AVG(price), 2) as price,
        COUNT(*) as popularity
    FROM {TABLE_EVENTS}
    WHERE event_type = 'purchase' AND category_level1 != 'unknown'
    GROUP BY category_level1, product_id
    ORDER BY category_level1, popularity DESC
    """
    pop_df = pd.read_sql_query(pop_query, conn)
    
    if not pop_df.empty:
        # Top N per category
        top_recs = pop_df.groupby("category_level1").head(top_n_per_category).reset_index(drop=True)
        top_recs.to_sql("product_recommendations", conn, if_exists="replace", index=False)
        logger.info(f"Saved {len(top_recs):,} top popularity recommendations across {top_recs['category_level1'].nunique()} categories.")
    else:
        top_recs = pop_df
        top_recs.to_sql("product_recommendations", conn, if_exists="replace", index=False)

    conn.close()
    return top_recs
