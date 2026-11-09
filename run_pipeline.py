"""
E-COMMERCE BIG DATA ANALYTICS & RECOMMENDATION SYSTEM
Command Line Interface & Pipeline Runner.

Usage:
    python run_pipeline.py --mode all --sample-chunks 5
    python run_pipeline.py --mode etl
    python run_pipeline.py --mode ml
    python run_pipeline.py --mode inference --user-id 512365478
    python run_pipeline.py --mode monitor
"""

import argparse
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import RAW_CSV_FILES, DATA_DIR, DB_PATH, OUTPUT_DIR, MODELS_DIR
from src.logger import get_logger
from src.storage import init_database, get_database_stats, export_to_parquet, reset_runtime
from src.ingestion import ingest_all_files, ingest_file
from src.features import extract_user_features
from src.ml_models import (
    train_customer_segmentation, train_churn_prediction, compute_recommendations, ModelRegistry
)
from src.inference import get_customer_360, predict_churn, get_segment, recommend_products
from src.monitoring import (
    get_system_health, evaluate_system_alerts, get_pipeline_history, get_quality_reports
)

logger = get_logger("PipelineRunner")

def run_environment_check() -> int:
    import importlib
    errors = []
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or newer is required")
    for package in ("pandas", "numpy", "pyarrow", "sklearn", "psutil"):
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "ok")
            print(f"  package {package}: {version}")
        except ImportError:
            errors.append(f"Missing package {package}")
            print(f"  package {package}: MISSING")
    missing_csv = []
    for name in RAW_CSV_FILES:
        path = DATA_DIR / name
        if path.is_file():
            print(f"  csv {name}: {path.stat().st_size / 1_000_000:.0f} MB")
        else:
            missing_csv.append(name)
            print(f"  csv {name}: MISSING")
    stats = get_database_stats()
    print(f"  database: {DB_PATH} ({stats['total_events']:,} events)")
    parquet = OUTPUT_DIR / "events.parquet"
    print(f"  parquet: {'ready' if parquet.is_file() else 'missing'}")
    for artifact in ("customer_segmentation_kmeans.pkl", "customer_segmentation_scaler.pkl", "churn_prediction_rf.pkl"):
        print(f"  model {artifact}: {'ready' if (MODELS_DIR / artifact).is_file() else 'missing'}")
    if missing_csv:
        errors.append("Place all five monthly CSV files in data/")
    if errors:
        print("Check failed:")
        for item in errors:
            print(f"  - {item}")
        return 1
    print("Check passed. Next: python run_pipeline.py --mode all --sample-chunks 2")
    return 0


def run_etl_stage(sample_chunks: int = None):
    logger.info("==================================================")
    logger.info("   STEP 1 - 4: DATA INGESTION & STORAGE (ETL)")
    logger.info("==================================================")
    init_database()
    results = ingest_all_files(max_chunks_per_file=sample_chunks)
    
    total_loaded = sum(r.get("rows_inserted", 0) for r in results)
    logger.info(f"ETL Stage Complete. Total rows loaded into analytical store: {total_loaded:,}")
    export_to_parquet("SELECT * FROM events", OUTPUT_DIR / "events.parquet")
    return results

def run_ml_stage(sample_size: int = 50_000):
    logger.info("==================================================")
    logger.info("   STEP 5 - 6: FEATURE STORE & MACHINE LEARNING")
    logger.info("==================================================")
    
    stats = get_database_stats()
    if stats["total_events"] == 0:
        raise RuntimeError("Events table is empty. Run ETL first.")

    # Extract features
    user_features = extract_user_features(sample_size=sample_size)
    if user_features.empty:
        raise RuntimeError("No user features could be extracted")

    user_features.to_parquet(OUTPUT_DIR / "user_features.parquet", index=False)

    # 1. Customer Segmentation
    kmeans, scaler, sil_score, seg_summary = train_customer_segmentation(user_features)
    print("\n--- Customer Segments Identified ---")
    print(seg_summary.to_string(index=False))

    # 2. Churn Prediction
    rf, metrics, importance = train_churn_prediction(extract_user_features(sample_size=sample_size, for_churn=True))
    print("\n--- Churn Prediction Performance ---")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")
    print("\nTop 5 Churn Risk Factors:")
    print(importance.head(5).to_string(index=False))

    # 3. Product Recommendations
    recs = compute_recommendations()
    print(f"\nProduct Recommendations Generated: {len(recs):,} items cataloged across categories.")

def run_inference_stage(user_id: int = None):
    logger.info("==================================================")
    logger.info("   STEP 7: APPLICATION & INFERENCE SERVING")
    logger.info("==================================================")

    conn = get_database_stats()
    if not conn["exists"] or conn["total_events"] == 0:
        raise RuntimeError("Database is empty. Run ETL before inference.")

    # If user_id not provided, pick an active one from DB or fallback
    if user_id is None:
        try:
            from src.storage import get_connection
            db_conn = get_connection()
            row = db_conn.execute("SELECT user_id FROM events WHERE event_type = 'purchase' LIMIT 1;").fetchone()
            db_conn.close()
            user_id = row[0] if row else 512365478
        except Exception:
            user_id = 512365478

    logger.info(f"Generating Real-Time Customer 360 for User ID: {user_id}")
    profile = get_customer_360(user_id)

    print("\n" + "="*60)
    print(f"   CUSTOMER 360 PROFILE - USER #{user_id}")
    print("="*60)
    print(json.dumps(profile, indent=2, ensure_ascii=False))

def run_monitoring_stage():
    logger.info("==================================================")
    logger.info("   STEP 8: SYSTEM MONITORING & HEALTH AUDIT")
    logger.info("==================================================")
    
    health = get_system_health()
    print("\n[Database Status]")
    print(f"  Location: {DB_PATH}")
    print(f"  Size: {health['database']['size_mb']:.2f} MB")
    print(f"  Total Events: {health['database']['total_events']:,}")
    print(f"  Tables: {health['database']['tables']}")

    print("\n[Trained Models Status]")
    for m, status in health["models"].items():
        print(f"  - {m}: {'READY' if status else 'MISSING'}")

    print("\n[Host Resources]")
    sys_res = health["system_resources"]
    print(f"  RAM Usage: {sys_res['ram_usage_percent']}% ({sys_res['ram_available_gb']} GB available)")
    print(f"  Disk Usage: {sys_res['disk_usage_percent']}% ({sys_res['disk_free_gb']} GB free)")

    alerts = evaluate_system_alerts()
    print("\n[Active Health Alerts]")
    if not alerts:
        print("  [OK] All systems healthy. No active alerts.")
    else:
        for a in alerts:
            print(f"  [ALERT] [{a['severity']}] {a['component']}: {a['message']}")

def main():
    parser = argparse.ArgumentParser(description="E-Commerce Big Data Analytics Runner")
    parser.add_argument(
        "--mode",
        choices=["all", "etl", "ml", "inference", "monitor", "serve", "check"],
        default="all",
        help="Pipeline execution mode (default: all)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete derived DB/parquet/models/monitoring files before running (never deletes CSV)"
    )
    parser.add_argument(
        "--sample-chunks",
        type=int,
        default=None,
        help="Limit number of chunks per CSV file for rapid demo testing (e.g. 5)"
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Target user ID for inference and Customer 360 profiling"
    )

    parser.add_argument("--port", type=int, default=8000, help="Local HTTP API port")
    args = parser.parse_args()
    if args.sample_chunks is not None and args.sample_chunks <= 0:
        parser.error("--sample-chunks must be positive")
    if args.reset and args.mode in {"serve", "check"}:
        parser.error("--reset cannot be combined with --mode serve or --mode check")
    if args.mode == "check":
        sys.exit(run_environment_check())
    if args.reset:
        logger.info("Resetting derived runtime artifacts")
        reset_runtime()
    if args.mode == "serve":
        if not 1 <= args.port <= 65535:
            parser.error("--port must be in 1..65535")
        from src.api import serve
        serve(args.port)
        return
    start = time.time()

    if args.mode in ["all", "etl"]:
        run_etl_stage(sample_chunks=args.sample_chunks)
    if args.mode in ["all", "ml"]:
        run_ml_stage()
    if args.mode in ["all", "inference"]:
        run_inference_stage(user_id=args.user_id)
    if args.mode in ["all", "monitor"]:
        run_monitoring_stage()

    elapsed = time.time() - start
    logger.info(f"Pipeline execution for mode '{args.mode}' completed in {elapsed:.2f}s.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
