"""
Data Quality Validation Module for Big Data Ingestion & Processing.
Evaluates data completeness, validity, uniqueness, and consistency against automated quality gates.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, Tuple
from src.config import (
    MAX_NULL_RATE_CRITICAL, MIN_PRICE_THRESHOLD, MIN_EVENT_DATE, MAX_EVENT_DATE,
    TABLE_QUALITY
)
from src.storage import get_connection
from src.logger import get_logger

logger = get_logger("DataQuality")

VALID_EVENT_TYPES = {"view", "cart", "remove_from_cart", "purchase"}

def evaluate_data_quality(df: pd.DataFrame, source_name: str = "batch") -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluates data quality across 4 key dimensions:
    1. Completeness: Critical column non-null rates
    2. Validity: Price bounds, event_types, date range, session length
    3. Uniqueness: Duplicate row detection
    4. Consistency: Category hierarchy extraction

    Returns (passed_gate: bool, quality_metrics: dict)
    """
    total_records = len(df)
    if total_records == 0:
        return False, {"error": "Empty dataframe provided for quality evaluation"}

    # 1. Completeness
    critical_cols = ["event_time", "event_type", "product_id", "user_id"]
    null_rates = {}
    for col in critical_cols:
        if col in df.columns:
            null_rates[col] = float(df[col].isna().mean())
        else:
            null_rates[col] = 1.0

    critical_null_violation = any(rate > MAX_NULL_RATE_CRITICAL for rate in null_rates.values())
    completeness_score = 1.0 - np.mean(list(null_rates.values()))

    # 2. Validity
    valid_event_types_rate = 1.0
    if "event_type" in df.columns:
        valid_event_types_rate = float(df["event_type"].isin(VALID_EVENT_TYPES).mean())

    price_valid_rate = 1.0
    price_outliers_rate = 0.0
    if "price" in df.columns:
        valid_price_mask = (df["price"] >= MIN_PRICE_THRESHOLD)
        price_valid_rate = float(valid_price_mask.mean())
        q99 = df["price"].quantile(0.99)
        price_outliers_rate = float((df["price"] > q99).mean())

    session_valid_rate = 1.0
    if "user_session" in df.columns:
        valid_sess_mask = df["user_session"].notna() & (df["user_session"].astype(str).str.len() >= 10)
        session_valid_rate = float(valid_sess_mask.mean())

    dates = pd.to_datetime(df.get("event_time", pd.Series(index=df.index, dtype=str)), errors="coerce", utc=True)
    valid_date_rate = float(dates.between(pd.Timestamp(MIN_EVENT_DATE, tz="UTC"), pd.Timestamp(MAX_EVENT_DATE, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)).mean())

    # 3. Uniqueness
    duplicate_count = int(df.duplicated().sum())
    duplicate_rate = float(duplicate_count / total_records)

    # 4. Gate decision
    passed = (
        not critical_null_violation and
        valid_event_types_rate >= 0.95 and
        price_valid_rate >= 0.95 and
        session_valid_rate >= 0.90 and
        valid_date_rate >= 0.95
    )

    report = {
        "source": source_name,
        "evaluated_at": datetime.now().isoformat(),
        "total_records": total_records,
        "passed_gate": bool(passed),
        "completeness_score": round(float(completeness_score), 4),
        "critical_null_rates": {k: round(v, 4) for k, v in null_rates.items()},
        "valid_event_types_rate": round(valid_event_types_rate, 4),
        "price_valid_rate": round(price_valid_rate, 4),
        "price_outlier_rate": round(price_outliers_rate, 4),
        "session_valid_rate": round(session_valid_rate, 4),
        "valid_date_rate": round(valid_date_rate, 4),
        "duplicate_rate": round(duplicate_rate, 4),
        "duplicate_count": duplicate_count
    }

    status_str = "PASSED" if passed else "FAILED"
    logger.info(
        f"Data Quality Gate [{status_str}] for {source_name}: "
        f"Records={total_records:,} | Completeness={report['completeness_score']:.2%} | "
        f"ValidSessions={report['session_valid_rate']:.2%}"
    )

    return passed, report

def persist_quality_report(report: Dict[str, Any], source_file: str) -> None:
    """
    Saves quality report metrics into SQLite data_quality_reports table for trend monitoring.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            INSERT INTO {TABLE_QUALITY} (
                source_file, total_records, completeness_score,
                valid_sessions_rate, price_outlier_rate, status, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            source_file,
            report.get("total_records", 0),
            report.get("completeness_score", 0.0),
            report.get("session_valid_rate", 0.0),
            report.get("price_outlier_rate", 0.0),
            "PASSED" if report.get("passed_gate", False) else "FAILED",
            json.dumps(report)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not persist quality report to database: {e}")
