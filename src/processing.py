"""
Data Transformation and Feature Engineering Module.
Performs memory-optimized data cleaning, session validation, invalid value removal,
and temporal/categorical feature extraction on event streams.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from src.config import MIN_EVENT_DATE, MAX_EVENT_DATE
from src.logger import get_logger

logger = get_logger("Processing")

def clean_and_transform_chunk(
    chunk: pd.DataFrame,
    chunk_id: int = 0,
    source_name: str = "batch"
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Cleans and transforms a single data chunk:
    - Deduplicates identical records
    - Fills missing descriptive categories and brands
    - Rejects missing sessions and invalid prices
    - Standardizes text columns
    - Validates sessions (length >= 10)
    - Validates event timestamps and filters out of bounds dates
    - Preserves valid high-value purchases
    - Extracts temporal features (hour, day_of_week, day_name, month, year)
    - Parses hierarchical category taxonomy (level1, level2, level3)

    Returns (transformed_df, chunk_statistics)
    """
    initial_rows = len(chunk)
    stats = {
        "initial_rows": initial_rows,
        "duplicates_removed": 0,
        "missing_handled": 0,
        "sessions_filtered": 0,
        "outliers_removed": 0,
        "datetime_filtered": 0,
        "final_rows": 0
    }

    # 1. Deduplication
    before_dedup = len(chunk)
    chunk = chunk.drop_duplicates().copy()
    stats["duplicates_removed"] = before_dedup - len(chunk)

    # Reject invalid monetary values rather than inventing purchases.
    chunk["category_code"] = chunk["category_code"].fillna("unknown")
    chunk["brand"] = chunk["brand"].fillna("unknown")
    chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
    chunk = chunk[np.isfinite(chunk["price"]) & (chunk["price"] > 0)].copy()
    chunk = chunk[chunk["event_type"].isin({"view", "cart", "remove_from_cart", "purchase"})].copy()

    # Category ID clean formatting (avoid float scientific notation)
    if "category_id" in chunk.columns:
        chunk["category_id"] = chunk["category_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        chunk.loc[chunk["category_id"].isin(["nan", "<NA>", "None", ""], ), "category_id"] = "0"

    # Drop records missing mandatory identifiers
    chunk = chunk.dropna(subset=["event_time", "event_type", "user_id", "product_id"])

    # 3. String Standardization
    chunk["category_code"] = chunk["category_code"].astype(str).str.lower().str.strip()
    chunk["brand"] = chunk["brand"].astype(str).str.lower().str.strip()
    chunk.loc[chunk["category_code"] == "", "category_code"] = "unknown"
    chunk.loc[chunk["brand"] == "", "brand"] = "unknown"

    # 4. Session Validation
    before_sess = len(chunk)
    chunk = chunk[chunk["user_session"].notna() & (chunk["user_session"].astype(str).str.len() >= 10)]
    stats["sessions_filtered"] = before_sess - len(chunk)

    # 5. Datetime Parsing and Range Validation
    chunk["event_time"] = pd.to_datetime(chunk["event_time"], errors="coerce", utc=True).dt.tz_localize(None)
    chunk = chunk[chunk["event_time"].notna()]
    before_dt = len(chunk)
    chunk = chunk[(chunk["event_time"] >= MIN_EVENT_DATE) & (chunk["event_time"] <= MAX_EVENT_DATE + " 23:59:59")]
    stats["datetime_filtered"] = before_dt - len(chunk)

    # 6. Valid positive prices (no percentile clipping)
    before_price = len(chunk)
    chunk = chunk[chunk["price"] > 0]
    stats["outliers_removed"] = before_price - len(chunk)

    # 7. Feature Engineering
    chunk["date"] = chunk["event_time"].dt.date.astype(str)
    chunk["hour"] = chunk["event_time"].dt.hour.astype("int16")
    chunk["day_of_week"] = chunk["event_time"].dt.day_of_week.astype("int8")
    chunk["day_name"] = chunk["event_time"].dt.day_name()
    chunk["month"] = chunk["event_time"].dt.month.astype("int8")
    chunk["year"] = chunk["event_time"].dt.year.astype("int16")

    # Category hierarchy extraction
    cat_parts = chunk["category_code"].str.split(".", expand=True)
    chunk["category_level1"] = cat_parts[0].fillna("unknown") if len(cat_parts.columns) > 0 else "unknown"
    chunk["category_level2"] = cat_parts[1].fillna("unknown") if len(cat_parts.columns) > 1 else "unknown"
    chunk["category_level3"] = cat_parts[2].fillna("unknown") if len(cat_parts.columns) > 2 else "unknown"

    chunk["source_file"] = source_name
    chunk["event_time"] = chunk["event_time"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

    stats["final_rows"] = len(chunk)
    return chunk, stats
