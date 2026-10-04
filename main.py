"""
Main Orchestrator - Chạy toàn bộ Data Pipeline
Raw Data -> ETL -> Storage -> ML -> Monitoring

Usage:
    python main.py              # Chạy toàn bộ pipeline
    python main.py --etl        # Chỉ chạy ETL
    python main.py --ml         # Chỉ chạy ML
    python main.py --monitor    # Chỉ chạy monitoring
    python main.py --dashboard  # Xem dashboard

Author: BigData Team
"""

import sys
import argparse
from datetime import datetime
import logging

# Import pipeline modules
from etl_pipeline import run_etl_pipeline, Config as ETLConfig
from ml_pipeline import run_ml_pipeline, Config as MLConfig
from monitoring import (
    MetricsCollector, 
    DataQualityMonitor, 
    ResourceMonitor,
    PipelineMonitor,
    DashboardGenerator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_etl():
    """Chạy ETL Pipeline"""
    logger.info("=" * 60)
    logger.info("STARTING ETL PIPELINE")
    logger.info("=" * 60)
    
    collector = MetricsCollector("etl_pipeline")
    
    try:
        collector.start_pipeline()
        df = run_etl_pipeline()
        
        # Monitor data quality
        quality_report = DataQualityMonitor.check_missing_values(df, "etl_output")
        logger.info(f"Data Quality - Missing ratio: {quality_report['overall_missing_ratio']:.2%}")
        
        dup_report = DataQualityMonitor.check_duplicate_ratio(df, "etl_output")
        logger.info(f"Data Quality - Duplicate ratio: {dup_report['duplicate_ratio']:.2%}")
        
        collector.end_pipeline(status="success")
        return df
        
    except Exception as e:
        collector.end_pipeline(status="failed", error_message=str(e))
        logger.error(f"ETL Pipeline failed: {e}")
        raise


def run_ml():
    """Chạy ML Pipeline"""
    logger.info("=" * 60)
    logger.info("STARTING ML PIPELINE")
    logger.info("=" * 60)
    
    collector = MetricsCollector("ml_pipeline")
    
    try:
        collector.start_pipeline()
        results = run_ml_pipeline()
        collector.end_pipeline(status="success")
        return results
        
    except Exception as e:
        collector.end_pipeline(status="failed", error_message=str(e))
        logger.error(f"ML Pipeline failed: {e}")
        raise


def run_monitoring():
    """Chạy Monitoring"""
    logger.info("=" * 60)
    logger.info("RUNNING MONITORING")
    logger.info("=" * 60)
    
    # Collect system metrics
    system_metrics = ResourceMonitor.get_system_metrics()
    logger.info(f"CPU: {system_metrics['cpu']['percent']:.1f}%")
    logger.info(f"Memory: {system_metrics['memory']['percent']:.1f}%")
    
    # Check pipeline health
    etl_health = PipelineMonitor.check_pipeline_health("etl_pipeline")
    ml_health = PipelineMonitor.check_pipeline_health("ml_pipeline")
    
    logger.info(f"ETL Health: {'✅ Healthy' if etl_health['healthy'] else '❌ Unhealthy'}")
    logger.info(f"ML Health: {'✅ Healthy' if ml_health['healthy'] else '❌ Unhealthy'}")
    
    return {"system": system_metrics, "etl_health": etl_health, "ml_health": ml_health}


def show_dashboard():
    """Hiển thị Dashboard"""
    DashboardGenerator.print_dashboard()


def run_full_pipeline():
    """Chạy toàn bộ pipeline"""
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🚀 BIG DATA PIPELINE - FULL EXECUTION")
    print(f"   Start time: {start_time}")
    print("=" * 70)
    
    try:
        # 1. Run ETL
        print("\n" + "-" * 50)
        print("📦 STEP 1: ETL PIPELINE")
        print("-" * 50)
        df = run_etl()
        
        # 2. Run ML
        print("\n" + "-" * 50)
        print("🤖 STEP 2: ML PIPELINE")
        print("-" * 50)
        results = run_ml()
        
        # 3. Run Monitoring
        print("\n" + "-" * 50)
        print("📊 STEP 3: MONITORING")
        print("-" * 50)
        run_monitoring()
        
        # 4. Show Dashboard
        print("\n" + "-" * 50)
        print("📈 DASHBOARD")
        print("-" * 50)
        show_dashboard()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"   Total duration: {duration}")
        print("=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ PIPELINE FAILED!")
        print(f"   Error: {e}")
        print("=" * 70)
        raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Big Data Pipeline Orchestrator")
    parser.add_argument('--etl', action='store_true', help='Run only ETL pipeline')
    parser.add_argument('--ml', action='store_true', help='Run only ML pipeline')
    parser.add_argument('--monitor', action='store_true', help='Run only monitoring')
    parser.add_argument('--dashboard', action='store_true', help='Show dashboard')
    
    args = parser.parse_args()
    
    if args.etl:
        run_etl()
    elif args.ml:
        run_ml()
    elif args.monitor:
        run_monitoring()
    elif args.dashboard:
        show_dashboard()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
