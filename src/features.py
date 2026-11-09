"""
Feature Engineering for Customer Behavior, Segmentation, and Churn Prediction.
Builds batch feature store tables and real-time user feature representations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from src.config import DB_PATH, CHURN_INACTIVE_DAYS, TABLE_EVENTS
from src.storage import get_connection
from src.logger import get_logger

logger = get_logger("FeatureStore")

# Feature column definitions
CLUSTERING_FEATURES = [
    "total_events", "total_spent", "avg_spent", "unique_products", "events_per_day"
]

CHURN_FEATURES = [
    "total_events", "total_spent", "avg_spent", "unique_products",
    "total_views", "unique_categories", "days_active", "events_per_day", "purchase_rate"
]

def extract_user_features(
    db_path: Path = DB_PATH,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    for_churn: bool = False
) -> pd.DataFrame:
    """
    Extracts aggregated behavioral features for users from the analytical event store.
    Computes RFM (Recency, Frequency, Monetary), engagement, and churn labels.
    """
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be positive")
    logger.info("Extracting user behavioral features from database...")
    conn = get_connection(db_path)

    # Check if table exists
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    if TABLE_EVENTS not in tables:
        conn.close()
        raise RuntimeError(f"Table '{TABLE_EVENTS}' not found in database {db_path}. Please run ETL ingestion first.")

    max_date_row = conn.execute(f"SELECT MAX(event_time) FROM {TABLE_EVENTS}").fetchone()[0]
    cutoff_date = pd.to_datetime(max_date_row, utc=True) - pd.Timedelta(days=CHURN_INACTIVE_DAYS) if max_date_row else None
    cutoff_text = cutoff_date.strftime("%Y-%m-%d %H:%M:%S+00:00") if cutoff_date is not None else ""
    observation_filter = "WHERE event_time < ?" if for_churn else ""
    params = []
    selection = ""
    sampled_filter = observation_filter
    if sample_size:
        # Select IDs in SQL before aggregation, bounding Python memory to the sample.
        selection = f"""WITH selected_users AS (
            SELECT user_id FROM {TABLE_EVENTS} {observation_filter}
            GROUP BY user_id
            ORDER BY (((user_id % 2147483647) * 1103515245 + ?) % 2147483647), user_id
            LIMIT ?
        )"""
        params = ([cutoff_text] if for_churn else []) + [int(random_state), int(sample_size)]
        sampled_filter = (observation_filter + " AND " if for_churn else "WHERE ") + "user_id IN (SELECT user_id FROM selected_users)"
    params += [cutoff_text] if for_churn else []
    user_agg_query = f"""
    {selection}
    SELECT 
        user_id,
        MIN(event_time) AS first_interaction,
        MAX(event_time) AS last_interaction,
        COUNT(*) AS total_events,
        SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS total_spent,
        COALESCE(AVG(CASE WHEN event_type = 'purchase' THEN price END), 0) AS avg_spent,
        COUNT(DISTINCT product_id) AS unique_products,
        COUNT(DISTINCT category_level1) AS unique_categories,
        SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS total_views,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS total_purchases
    FROM {TABLE_EVENTS}
    {sampled_filter}
    GROUP BY user_id
    """

    df = pd.read_sql_query(user_agg_query, conn, params=params)
    
    # Query future activity only for selected users, rather than materializing every user.
    if max_date_row and not df.empty:
        conn.execute("CREATE TEMP TABLE selected_feature_users (user_id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO selected_feature_users VALUES (?)", ((int(u),) for u in df["user_id"]))
        active_users_df = pd.read_sql_query(
            f"SELECT DISTINCT user_id FROM {TABLE_EVENTS} WHERE event_time >= ? AND user_id IN (SELECT user_id FROM selected_feature_users)",
            conn, params=[cutoff_text])
        active_user_ids = set(active_users_df["user_id"].values)
    else:
        active_user_ids = set()

    conn.close()

    if df.empty:
        logger.warning("No user records returned from event store.")
        return df

    # Feature transformations
    df["first_interaction"] = pd.to_datetime(df["first_interaction"])
    df["last_interaction"] = pd.to_datetime(df["last_interaction"])
    df["days_active"] = (df["last_interaction"] - df["first_interaction"]).dt.days + 1
    df["days_active"] = df["days_active"].clip(lower=1)

    df["events_per_day"] = df["total_events"] / df["days_active"]
    df["avg_session_value"] = df["total_spent"] / df["total_events"].replace(0, 1)
    df["purchase_rate"] = df["total_purchases"] / df["total_events"].replace(0, 1)

    # Churn label: 1 if user had no interaction in the last CHURN_INACTIVE_DAYS
    df["churned"] = (~df["user_id"].isin(active_user_ids)).astype(int)
    df = df.fillna(0)

    logger.info(f"Extracted features for {len(df):,} unique users. Churn rate: {df['churned'].mean():.2%}")

    df.attrs["label_cutoff"] = cutoff_text
    df.attrs["evaluation"] = "held-out users; features before cutoff, labels in following 30 days" if for_churn else "descriptive features"
    return df

def extract_single_user_features(user_id: int, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Extracts on-demand behavioral features for a specific user for real-time inference.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    query = f"""
    SELECT 
        user_id,
        MIN(event_time) AS first_interaction,
        MAX(event_time) AS last_interaction,
        COUNT(*) AS total_events,
        COALESCE(SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END), 0.0) AS total_spent,
        COALESCE(AVG(CASE WHEN event_type = 'purchase' THEN price END), 0.0) AS avg_spent,
        COUNT(DISTINCT product_id) AS unique_products,
        COUNT(DISTINCT category_level1) AS unique_categories,
        SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS total_views,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS total_purchases
    FROM {TABLE_EVENTS}
    WHERE user_id = ?
    GROUP BY user_id
    """
    row = cursor.execute(query, (user_id,)).fetchone()
    if not row:
        conn.close()
        return None

    (u_id, first_dt, last_dt, total_events, total_spent, avg_spent,
     uniq_products, uniq_cats, total_views, total_purchases) = row

    first_dt = pd.to_datetime(first_dt)
    last_dt = pd.to_datetime(last_dt)
    days_active = max(1, (last_dt - first_dt).days + 1)
    events_per_day = total_events / days_active
    purchase_rate = total_purchases / max(1, total_events)

    # Favorite category for recommendation context
    fav_cat_row = cursor.execute(f"""
        SELECT category_level1, COUNT(*) as cnt
        FROM {TABLE_EVENTS}
        WHERE user_id = ? AND category_level1 != 'unknown'
        GROUP BY category_level1
        ORDER BY cnt DESC LIMIT 1
    """, (user_id,)).fetchone()

    favorite_category = fav_cat_row[0] if fav_cat_row else "electronics"
    conn.close()

    return {
        "user_id": user_id,
        "first_interaction": str(first_dt),
        "last_interaction": str(last_dt),
        "total_events": int(total_events),
        "total_spent": float(total_spent),
        "avg_spent": float(avg_spent),
        "unique_products": int(uniq_products),
        "unique_categories": int(uniq_cats),
        "total_views": int(total_views),
        "total_purchases": int(total_purchases),
        "days_active": int(days_active),
        "events_per_day": float(events_per_day),
        "purchase_rate": float(purchase_rate),
        "favorite_category": favorite_category
    }
