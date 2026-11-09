"""
Storage Layer for E-Commerce Big Data Architecture.
Handles SQLite connections, schema initialization, indices, Parquet export, and metadata tracking.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from src.config import (
    DB_PATH, OUTPUT_DIR, MODELS_DIR, MONITORING_DIR,
    TABLE_EVENTS, TABLE_METADATA, TABLE_QUALITY, TABLE_MODEL_METRICS
)
from src.logger import get_logger

logger = get_logger("StorageLayer")

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Get an optimized SQLite connection with WAL mode and reasonable timeout.
    """
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    # Enable WAL mode for high concurrent write and read performance
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA cache_size = -131072;")  # Bounded 128 MiB page cache
    conn.execute("PRAGMA temp.cache_size = -16384;")
    return conn

def init_database(db_path: Path = DB_PATH) -> None:
    """
    Initializes database tables and analytical indexes.
    """
    logger.info(f"Initializing database at: {db_path}")
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 1. Fact Table: events
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_EVENTS} (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time TEXT NOT NULL,
            event_type TEXT NOT NULL,
            product_id INTEGER NOT NULL,
            category_id TEXT,
            category_code TEXT,
            brand TEXT,
            price REAL,
            user_id INTEGER NOT NULL,
            user_session TEXT,
            source_file TEXT,
            date TEXT,
            hour INTEGER,
            day_of_week INTEGER,
            day_name TEXT,
            month INTEGER,
            year INTEGER,
            category_level1 TEXT,
            category_level2 TEXT,
            category_level3 TEXT
        );
    """)

    # Indexes for analytical queries
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_events_user ON {TABLE_EVENTS} (user_id);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_events_type ON {TABLE_EVENTS} (event_type);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_events_product ON {TABLE_EVENTS} (product_id);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_events_date ON {TABLE_EVENTS} (date);")

    # 2. Pipeline Metadata Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_METADATA} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline_name TEXT NOT NULL,
            source_file TEXT,
            table_name TEXT,
            rows_read INTEGER DEFAULT 0,
            rows_inserted INTEGER DEFAULT 0,
            duplicates_removed INTEGER DEFAULT 0,
            outliers_removed INTEGER DEFAULT 0,
            execution_seconds REAL DEFAULT 0.0,
            status TEXT NOT NULL,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 3. Data Quality Reports Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_QUALITY} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            total_records INTEGER,
            completeness_score REAL,
            valid_sessions_rate REAL,
            price_outlier_rate REAL,
            status TEXT NOT NULL,
            details TEXT,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 4. Model Registry / Performance Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_MODEL_METRICS} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            version TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_source ON events (source_file)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events (event_time)")
    cursor.execute("""CREATE TABLE IF NOT EXISTS ingestion_manifest (
        source_file TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
        chunk_limit INTEGER, rows_loaded INTEGER NOT NULL,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()
    logger.info("Database and table schemas successfully initialized.")

def log_pipeline_run(
    pipeline_name: str,
    table_name: str,
    source_file: str,
    rows_read: int,
    rows_inserted: int,
    duplicates_removed: int,
    outliers_removed: int,
    execution_seconds: float,
    status: str,
    error_message: Optional[str] = None,
    db_path: Path = DB_PATH
) -> None:
    """
    Records pipeline execution statistics into pipeline_metadata table.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(f"""
        INSERT INTO {TABLE_METADATA} (
            pipeline_name, table_name, source_file, rows_read, rows_inserted,
            duplicates_removed, outliers_removed, execution_seconds, status, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        pipeline_name, table_name, source_file, rows_read, rows_inserted,
        duplicates_removed, outliers_removed, execution_seconds, status, error_message
    ))
    conn.commit()
    conn.close()

def get_database_stats(db_path: Path = DB_PATH) -> Dict[str, Any]:
    """
    Returns high-level statistics of the database for monitoring and reporting.
    """
    if not db_path.exists():
        return {
            "exists": False,
            "size_mb": 0.0,
            "tables": {},
            "total_events": 0
        }

    conn = get_connection(db_path)
    cursor = conn.cursor()

    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';").fetchall()
    table_counts = {}
    for (t_name,) in tables:
        count = cursor.execute(f"SELECT COUNT(*) FROM {t_name};").fetchone()[0]
        table_counts[t_name] = count

    conn.close()
    size_mb = db_path.stat().st_size / (1024 * 1024)

    return {
        "exists": True,
        "size_mb": round(size_mb, 2),
        "tables": table_counts,
        "total_events": table_counts.get(TABLE_EVENTS, 0)
    }

def export_to_parquet(query: str, output_filepath: Path, db_path: Path = DB_PATH) -> Path:
    """
    Exports SQL query results into columnar Parquet format for fast analytics.
    """
    import os
    import pyarrow as pa
    import pyarrow.parquet as pq
    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_filepath.with_suffix(".parquet.tmp")
    conn = get_connection(db_path)
    writer = None
    total = 0
    try:
        # Explicit SQL-derived schema prevents all-null first chunks becoming null types.
        for frame in pd.read_sql_query(query, conn, chunksize=100_000):
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                schema = pa.schema([pa.field(f.name, pa.string() if pa.types.is_null(f.type) else f.type) for f in table.schema])
                writer = pq.ParquetWriter(temporary, schema, compression="snappy")
            table = table.cast(writer.schema)
            writer.write_table(table)
            total += len(frame)
        if writer is None:
            raise ValueError("Query returned no schema for Parquet export")
        writer.close()
        writer = None
        os.replace(temporary, output_filepath)
    finally:
        if writer is not None:
            writer.close()
        conn.close()
        temporary.unlink(missing_ok=True)
    logger.info(f"Exported {total:,} rows to {output_filepath}")
    return output_filepath


def checkpoint_database(db_path: Path = DB_PATH) -> None:
    if not db_path.exists():
        return
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def reset_runtime(*, include_models: bool = True) -> None:
    """Delete derived artifacts. Never deletes raw CSV inputs."""
    checkpoint_database()
    for path in (DB_PATH, Path(str(DB_PATH) + "-wal"), Path(str(DB_PATH) + "-shm")):
        path.unlink(missing_ok=True)
    if OUTPUT_DIR.exists():
        for parquet in OUTPUT_DIR.glob("*.parquet"):
            parquet.unlink()
    if MONITORING_DIR.exists():
        for snapshot in MONITORING_DIR.glob("*.json"):
            snapshot.unlink()
    if include_models and MODELS_DIR.exists():
        for artifact in MODELS_DIR.glob("*.pkl"):
            artifact.unlink()
        registry = MODELS_DIR / "model_registry.json"
        registry.unlink(missing_ok=True)
        versions = MODELS_DIR / "versions"
        if versions.exists():
            import shutil
            shutil.rmtree(versions)
    logger.info("Cleared derived runtime artifacts (database, parquet, monitoring%s).",
                ", models" if include_models else "")
