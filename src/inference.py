"""
Model Inference and Serving Layer (Step 7).
Provides real-time scoring, Customer 360 profiling, churn risk evaluation,
and personalized product recommendation APIs.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.config import MODELS_DIR, DB_PATH, FEATURE_SCHEMA_VERSION
from src.features import (
    extract_single_user_features, CLUSTERING_FEATURES, CHURN_FEATURES
)
from src.storage import get_connection
from src.logger import get_logger

logger = get_logger("InferenceEngine")

class ModelArtifactCache:
    """
    In-memory singleton cache to load serialized ML models once and serve low-latency requests.
    """
    _cache = {}

    @classmethod
    def _load(cls, filename):
        path = MODELS_DIR / filename
        if not path.exists():
            return None
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        if filename not in cls._cache or cls._cache[filename][0] != stamp:
            with path.open("rb") as handle:
                model = pickle.load(handle)
            if getattr(model, "feature_schema_version_", None) != FEATURE_SCHEMA_VERSION:
                raise RuntimeError("Model feature schema is outdated. Run: python run_pipeline.py --mode ml")
            cls._cache[filename] = (stamp, model)
        return cls._cache[filename][1]

    @classmethod
    def get_kmeans(cls):
        return cls._load("customer_segmentation_kmeans.pkl")

    @classmethod
    def get_scaler(cls):
        return cls._load("customer_segmentation_scaler.pkl")

    @classmethod
    def get_rf(cls):
        return cls._load("churn_prediction_rf.pkl")


def predict_churn(user_id: int, user_feats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Evaluates churn risk for a specific user.
    """
    rf_model = ModelArtifactCache.get_rf()
    if rf_model is None:
        return {"error": "Churn model not found. Please train model first."}

    if user_feats is None:
        user_feats = extract_single_user_features(user_id)

    if user_feats is None:
        return {
            "user_id": user_id,
            "status": "UNKNOWN_USER",
            "message": "User not found in historical events."
        }

    # Prepare feature vector matching training schema
    feature_df = pd.DataFrame([{c: user_feats.get(c, 0.0) for c in CHURN_FEATURES}])
    prob = float(rf_model.predict_proba(feature_df)[0][list(rf_model.classes_).index(1)])
    is_churn = int(prob >= 0.5)

    if prob >= 0.70:
        risk_level = "HIGH"
        action = "Urgent win-back promotion, discount offer via notification"
    elif prob >= 0.40:
        risk_level = "MEDIUM"
        action = "Personalized engagement email and featured product digest"
    else:
        risk_level = "LOW"
        action = "Standard loyalty program engagement"

    # Identify primary risk drivers
    drivers = []
    if user_feats.get("days_active", 0) <= 2:
        drivers.append("Short engagement span (< 3 days)")
    if user_feats.get("purchase_rate", 0) == 0:
        drivers.append("Zero conversions (views only)")
    if user_feats.get("events_per_day", 0) < 2:
        drivers.append("Low interaction frequency")

    return {
        "user_id": user_id,
        "is_churn": is_churn,
        "churn_probability": round(prob, 4),
        "risk_level": risk_level,
        "suggested_action": action,
        "risk_drivers": drivers
    }


def get_segment(user_id: int, user_feats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Determines customer persona and behavioral cluster for a user.
    """
    kmeans = ModelArtifactCache.get_kmeans()
    scaler = ModelArtifactCache.get_scaler()

    if kmeans is None or scaler is None:
        return {"error": "Segmentation model not found. Please train model first."}

    if user_feats is None:
        user_feats = extract_single_user_features(user_id)

    if user_feats is None:
        return {
            "user_id": user_id,
            "status": "UNKNOWN_USER",
            "message": "User not found in historical events."
        }

    raw_df = pd.DataFrame([{c: user_feats.get(c, 0.0) for c in CLUSTERING_FEATURES}])
    scaled_features = scaler.transform(raw_df)
    cluster_id = int(kmeans.predict(scaled_features)[0])

    persona = getattr(kmeans, "personas_", {}).get(cluster_id, {
        "name": f"Segment {cluster_id}",
        "strategy": "Standard marketing campaign"
    })

    return {
        "user_id": user_id,
        "cluster_id": cluster_id,
        "persona_name": persona["name"],
        "marketing_strategy": persona["strategy"],
        "total_spent": user_feats.get("total_spent", 0.0),
        "total_events": user_feats.get("total_events", 0)
    }


def recommend_products(user_id: int, top_n: int = 5, user_feats: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Delivers personalized product recommendations based on user affinity & category popularity.
    """
    if user_feats is None:
        user_feats = extract_single_user_features(user_id)

    fav_category = user_feats.get("favorite_category", "electronics") if user_feats else "electronics"

    conn = get_connection()
    # Check if recommendation table exists
    has_table = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='product_recommendations';").fetchone()

    recommendations = []
    if has_table:
        query = """
            SELECT product_id, category_level1, brand, price, popularity
            FROM product_recommendations
            WHERE category_level1 = ?
            ORDER BY popularity DESC
            LIMIT ?
        """
        rows = conn.execute(query, (fav_category, top_n)).fetchall()
        
        # If not enough in favorite category, backfill with overall popular items
        if len(rows) < top_n:
            backfill_query = """
                SELECT product_id, category_level1, brand, price, popularity
                FROM product_recommendations
                WHERE category_level1 != ?
                ORDER BY popularity DESC
                LIMIT ?
            """
            extra_rows = conn.execute(backfill_query, (fav_category, top_n - len(rows))).fetchall()
            rows.extend(extra_rows)

        for r in rows:
            recommendations.append({
                "product_id": r[0],
                "category": r[1],
                "brand": r[2] if r[2] else "General",
                "price_usd": r[3],
                "popularity_score": r[4],
                "recommendation_reason": f"Top trending in '{r[1]}' category"
            })
    conn.close()

    return recommendations


def get_customer_360(user_id: int) -> Dict[str, Any]:
    """
    End-to-End unified Customer 360 profile combining profile, behavioral traits,
    segmentation, churn risk, and personalized recommendations.
    """
    user_feats = extract_single_user_features(user_id)
    if not user_feats:
        return {
            "status": "NOT_FOUND",
            "message": f"User {user_id} has no records in the event store."
        }

    churn_info = predict_churn(user_id, user_feats=user_feats)
    segment_info = get_segment(user_id, user_feats=user_feats)
    recs = recommend_products(user_id, top_n=5, user_feats=user_feats)

    return {
        "status": "SUCCESS",
        "user_profile": {
            "user_id": user_id,
            "first_interaction": user_feats["first_interaction"],
            "last_interaction": user_feats["last_interaction"],
            "days_active": user_feats["days_active"],
            "total_events": user_feats["total_events"],
            "total_purchases": user_feats["total_purchases"],
            "total_spent_usd": round(user_feats["total_spent"], 2),
            "purchase_rate": f"{user_feats['purchase_rate']:.1%}",
            "favorite_category": user_feats["favorite_category"]
        },
        "customer_segmentation": segment_info,
        "churn_prediction": churn_info,
        "recommended_products": recs
    }


def batch_inference(user_ids: List[int]) -> pd.DataFrame:
    """
    Executes batch inference across a list of user IDs for reporting or marketing campaign targeting.
    """
    records = []
    for uid in user_ids:
        feats = extract_single_user_features(uid)
        if feats:
            c_info = predict_churn(uid, user_feats=feats)
            s_info = get_segment(uid, user_feats=feats)
            records.append({
                "user_id": uid,
                "days_active": feats["days_active"],
                "total_spent": feats["total_spent"],
                "persona": s_info.get("persona_name", "Unknown"),
                "churn_prob": c_info.get("churn_probability", 0.0),
                "risk_level": c_info.get("risk_level", "UNKNOWN")
            })

    return pd.DataFrame(records)
