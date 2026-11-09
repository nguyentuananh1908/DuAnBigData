"""
Apache Airflow DAG Definition for E-Commerce Big Data Analytics Pipeline.

Orchestrates the entire 8-step workflow:
Data Sources Check -> Ingestion & Quality Gates -> Parquet Mart Export ->
Feature Store Extraction -> Parallel ML Model Training -> Model Registry Update ->
Health Audit & Alert Notification.
"""

from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to sys.path so tasks can import src
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Airflow imports with safe fallback for environments without Airflow installed
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.empty import EmptyOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = None


default_args = {
    "owner": "data_engineering_team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

def task_check_data_sources():
    from src.config import DATA_DIR, RAW_CSV_FILES
    found = [f for f in RAW_CSV_FILES if (DATA_DIR / f).exists()]
    if len(found) != len(RAW_CSV_FILES):
        raise FileNotFoundError(f"Expected all {len(RAW_CSV_FILES)} CSV files in {DATA_DIR}")
    print(f"Data Source Check: Found {len(found)}/{len(RAW_CSV_FILES)} raw data files ready.")

def task_ingest_and_validate():
    from src.ingestion import ingest_all_files
    # For daily incremental batch, runs chunk-based ingestion
    results = ingest_all_files()
    print(f"Ingestion executed for {len(results)} files.")

def task_export_parquet():
    from src.config import OUTPUT_DIR, TABLE_EVENTS
    from src.storage import export_to_parquet
    today_str = datetime.now().strftime("%Y%m%d")
    output_path = OUTPUT_DIR / f"events_snapshot_{today_str}.parquet"
    export_to_parquet(f"SELECT * FROM {TABLE_EVENTS}", output_path)

def task_extract_features():
    from src.features import extract_user_features
    df = extract_user_features(sample_size=50000)
    from src.config import OUTPUT_DIR
    df.to_parquet(OUTPUT_DIR / "user_features.parquet", index=False)
    print(f"Feature Store Updated: {len(df):,} user feature vectors ready.")

def task_train_segmentation():
    from src.features import extract_user_features
    from src.ml_models import train_customer_segmentation
    df = extract_user_features(sample_size=30000)
    kmeans, scaler, score, _ = train_customer_segmentation(df)
    print(f"Segmentation Training Completed. Silhouette: {score:.4f}")

def task_train_churn():
    from src.features import extract_user_features
    from src.ml_models import train_churn_prediction
    df = extract_user_features(sample_size=30000, for_churn=True)
    rf, metrics, _ = train_churn_prediction(df)
    print(f"Churn Model Training Completed. F1-Score: {metrics['f1']:.4f}")

def task_update_recommendations():
    from src.ml_models import compute_recommendations
    recs = compute_recommendations()
    print(f"Product Recommendations Updated: {len(recs)} catalog entries.")

def task_health_and_alerts():
    from src.monitoring import evaluate_system_alerts, get_system_health
    health = get_system_health()
    alerts = evaluate_system_alerts()
    print("Health Audit:", health["system_resources"])
    if alerts:
        print(f"Triggered {len(alerts)} alerts: {alerts}")
    else:
        print("All pipeline nodes healthy.")

dag = None
if AIRFLOW_AVAILABLE:
    # DAG Declaration
    dag = DAG(
        "ecommerce_bigdata_analytics_pipeline",
        default_args=default_args,
        description="End-to-End E-Commerce Big Data Processing, ML & Monitoring Pipeline",
        schedule_interval=timedelta(days=1),
        catchup=False,
        max_active_runs=1,
        tags=["bigdata", "ecommerce", "mlops", "etl"]
    )

    with dag:
        t_check = PythonOperator(
            task_id="check_data_sources",
            python_callable=task_check_data_sources
        )

        t_ingest = PythonOperator(
            task_id="ingest_and_validate_data",
            python_callable=task_ingest_and_validate
        )

        t_export = PythonOperator(
            task_id="export_analytical_parquet",
            python_callable=task_export_parquet
        )

        t_features = PythonOperator(
            task_id="extract_user_features",
            python_callable=task_extract_features
        )

        t_segmentation = PythonOperator(
            task_id="train_customer_segmentation",
            python_callable=task_train_segmentation
        )

        t_churn = PythonOperator(
            task_id="train_churn_prediction",
            python_callable=task_train_churn
        )

        t_recs = PythonOperator(
            task_id="update_product_recommendations",
            python_callable=task_update_recommendations
        )

        t_health = PythonOperator(
            task_id="health_audit_and_alerts",
            python_callable=task_health_and_alerts
        )

        # Dependency Pipeline Graph
        t_check >> t_ingest >> t_export >> t_features
        t_features >> t_segmentation >> t_churn >> t_recs >> t_health
