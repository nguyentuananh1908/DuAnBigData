"""
Centralized Configuration for E-Commerce Big Data Analytics Pipeline.
Provides standardized paths, schema definitions, and pipeline constants.
"""

from pathlib import Path
import os

# Project root directory (DuAnBigData)
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage & Data Paths
RUNTIME_DIR = Path(os.environ.get("BIGDATA_RUNTIME_DIR", str(BASE_DIR))).resolve()
DATA_DIR = Path(os.environ.get("BIGDATA_DATA_DIR", str(RUNTIME_DIR / "data"))).resolve()
OUTPUT_DIR = RUNTIME_DIR / "data_clean"
MODELS_DIR = RUNTIME_DIR / "models"
LOGS_DIR = RUNTIME_DIR / "logs"
MONITORING_DIR = RUNTIME_DIR / "monitoring"
DB_PATH = Path(os.environ.get("BIGDATA_DB_PATH", str(RUNTIME_DIR / "data" / "bigdata.db"))).resolve()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Create directories if they do not exist
for directory in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR, MONITORING_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Raw Data Sources (Ordered chronologically)
RAW_CSV_FILES = [
    "2019-Oct.csv",
    "2019-Nov.csv",
    "2019-Dec.csv",
    "2020-Jan.csv",
    "2020-Feb.csv",
]

# ETL Ingestion Parameters
CHUNK_SIZE = 100_000

# High-efficiency dtypes for chunk reading
# Fix category_id float32 precision loss by reading as string/int64 or float64
RAW_DTYPES = {
    "event_type": "category",
    "product_id": "Int64",
    "category_id": "string",  # preserves exact 18-digit IDs without scientific notation corruption
    "category_code": "string",
    "brand": "string",
    "price": "float64",
    "user_id": "Int64",       # int64 prevents int32 overflow for large user IDs
    "user_session": "string",
}

# Date validation range
MIN_EVENT_DATE = "2019-10-01"
MAX_EVENT_DATE = "2020-02-29"

# Table Names in SQLite
TABLE_EVENTS = "events"
TABLE_METADATA = "pipeline_metadata"
TABLE_QUALITY = "data_quality_reports"
TABLE_MODEL_METRICS = "model_performance"

# Machine Learning Parameters
RANDOM_STATE = 42
KMEANS_N_CLUSTERS = 5
CHURN_INACTIVE_DAYS = 30
RECOMMENDATION_TOP_N = 10

# Data Quality Gate Thresholds
MAX_NULL_RATE_CRITICAL = 0.05   # Max 5% null in event_time, user_id, product_id
MIN_PRICE_THRESHOLD = 0.01      # Minimum valid price (USD)

FEATURE_SCHEMA_VERSION = "2.0"
