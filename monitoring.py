"""
Monitoring System cho ETL & ML Pipelines
Track metrics: latency, error rate, data quality, resource usage, model accuracy

Author: BigData Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
import psutil
import os

# ============== CONFIG ==============
class Config:
    LOG_DIR = Path("logs")
    METRICS_DIR = Path("monitoring/metrics")
    ALERTS_DIR = Path("monitoring/alerts")
    DASHBOARD_DIR = Path("monitoring/dashboard")
    
    # Thresholds
    ERROR_RATE_THRESHOLD = 0.05  # 5%
    MISSING_DATA_THRESHOLD = 0.10  # 10%
    LATENCY_THRESHOLD_SEC = 300  # 5 minutes
    CPU_THRESHOLD = 80  # 80%
    MEMORY_THRESHOLD = 80  # 80%

Config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
Config.ALERTS_DIR.mkdir(parents=True, exist_ok=True)
Config.DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== METRICS COLLECTOR ==============
class MetricsCollector:
    """Collect và track pipeline metrics"""
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.metrics_file = Config.METRICS_DIR / f"{pipeline_name}_metrics.json"
        self.start_time = None
        self.end_time = None
        
    def start_pipeline(self):
        """Mark pipeline start"""
        self.start_time = datetime.now()
        logger.info(f"Pipeline '{self.pipeline_name}' started at {self.start_time}")
        
    def end_pipeline(self, status: str = "success", error_message: str = None):
        """Mark pipeline end"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        metric = {
            "pipeline": self.pipeline_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": duration,
            "status": status,
            "error_message": error_message
        }
        
        self._save_metric(metric)
        logger.info(f"Pipeline '{self.pipeline_name}' ended - Duration: {duration:.2f}s, Status: {status}")
        
        return metric
    
    def _save_metric(self, metric: dict):
        """Save metric to JSON file"""
        metrics = self._load_metrics()
        metrics.append(metric)
        
        # Keep only last 100 metrics
        metrics = metrics[-100:]
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _load_metrics(self) -> list:
        """Load existing metrics"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return []


# ============== DATA QUALITY MONITOR ==============
class DataQualityMonitor:
    """Monitor data quality metrics"""
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, source: str = "unknown") -> dict:
        """Check missing values ratio"""
        missing_counts = df.isnull().sum()
        missing_ratio = missing_counts / len(df)
        
        quality_metrics = {
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_by_column": missing_counts.to_dict(),
            "missing_ratio_by_column": missing_ratio.to_dict(),
            "overall_missing_ratio": df.isnull().sum().sum() / (len(df) * len(df.columns))
        }
        
        # Check threshold
        if quality_metrics['overall_missing_ratio'] > Config.MISSING_DATA_THRESHOLD:
            DataQualityMonitor._create_alert(
                "HIGH_MISSING_DATA",
                f"Missing data ratio: {quality_metrics['overall_missing_ratio']:.2%}",
                "warning"
            )
        
        return quality_metrics
    
    @staticmethod
    def check_duplicate_ratio(df: pd.DataFrame, source: str = "unknown") -> dict:
        """Check duplicate rows ratio"""
        duplicate_count = df.duplicated().sum()
        duplicate_ratio = duplicate_count / len(df)
        
        quality_metrics = {
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "duplicate_rows": int(duplicate_count),
            "duplicate_ratio": float(duplicate_ratio)
        }
        
        if duplicate_ratio > 0.01:  # > 1%
            DataQualityMonitor._create_alert(
                "HIGH_DUPLICATES",
                f"Duplicate ratio: {duplicate_ratio:.2%}",
                "warning"
            )
        
        return quality_metrics
    
    @staticmethod
    def check_value_ranges(df: pd.DataFrame, column: str, min_val: float, max_val: float) -> dict:
        """Check if values are within expected range"""
        out_of_range = ((df[column] < min_val) | (df[column] > max_val)).sum()
        out_of_range_ratio = out_of_range / len(df)
        
        return {
            "column": column,
            "min_expected": min_val,
            "max_expected": max_val,
            "out_of_range_count": int(out_of_range),
            "out_of_range_ratio": float(out_of_range_ratio)
        }
    
    @staticmethod
    def _create_alert(alert_type: str, message: str, severity: str):
        """Create alert file"""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        
        alert_file = Config.ALERTS_DIR / f"{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(alert_file, 'w') as f:
            json.dump(alert, f, indent=2)
        
        logger.warning(f"ALERT [{severity.upper()}]: {message}")


# ============== SYSTEM RESOURCE MONITOR ==============
class ResourceMonitor:
    """Monitor system resources"""
    
    @staticmethod
    def get_system_metrics() -> dict:
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": cpu_percent,
                "alert": cpu_percent > Config.CPU_THRESHOLD
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent,
                "alert": memory.percent > Config.MEMORY_THRESHOLD
            },
            "disk": {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "percent": disk.percent
            }
        }
        
        return metrics
    
    @staticmethod
    def save_metrics_snapshot():
        """Save system metrics snapshot"""
        metrics = ResourceMonitor.get_system_metrics()
        
        snapshot_file = Config.METRICS_DIR / f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(snapshot_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics


# ============== PIPELINE MONITOR ==============
class PipelineMonitor:
    """Monitor ETL and ML pipeline health"""
    
    @staticmethod
    def get_pipeline_summary(pipeline_name: str) -> dict:
        """Get pipeline execution summary"""
        metrics_file = Config.METRICS_DIR / f"{pipeline_name}_metrics.json"
        
        if not metrics_file.exists():
            return {"error": "No metrics found"}
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        if not metrics:
            return {"error": "No metrics available"}
        
        # Calculate summary statistics
        successful = [m for m in metrics if m.get('status') == 'success']
        failed = [m for m in metrics if m.get('status') == 'failed']
        
        durations = [m['duration_seconds'] for m in metrics]
        
        summary = {
            "pipeline": pipeline_name,
            "total_runs": len(metrics),
            "successful_runs": len(successful),
            "failed_runs": len(failed),
            "success_rate": len(successful) / len(metrics) if metrics else 0,
            "avg_duration_seconds": np.mean(durations) if durations else 0,
            "min_duration_seconds": min(durations) if durations else 0,
            "max_duration_seconds": max(durations) if durations else 0,
            "last_run": metrics[-1] if metrics else None
        }
        
        return summary
    
    @staticmethod
    def check_pipeline_health(pipeline_name: str) -> dict:
        """Check if pipeline is healthy"""
        summary = PipelineMonitor.get_pipeline_summary(pipeline_name)
        
        if "error" in summary:
            return {"healthy": False, "reason": summary["error"]}
        
        # Check success rate
        if summary["success_rate"] < 0.95:  # < 95%
            DataQualityMonitor._create_alert(
                f"{pipeline_name.upper()}_HEALTH",
                f"Success rate below 95%: {summary['success_rate']:.1%}",
                "warning"
            )
            return {"healthy": False, "reason": "Low success rate"}
        
        # Check latency
        if summary["avg_duration_seconds"] > Config.LATENCY_THRESHOLD_SEC:
            DataQualityMonitor._create_alert(
                f"{pipeline_name.upper()}_LATENCY",
                f"Average latency high: {summary['avg_duration_seconds']:.1f}s",
                "warning"
            )
            return {"healthy": False, "reason": "High latency"}
        
        return {"healthy": True, "summary": summary}


# ============== MODEL MONITOR ==============
class ModelMonitor:
    """Monitor ML model performance"""
    
    @staticmethod
    def track_model_metrics(model_name: str, metrics: dict):
        """Track model performance metrics"""
        monitor_file = Config.METRICS_DIR / f"{model_name}_performance.json"
        
        existing = []
        if monitor_file.exists():
            with open(monitor_file, 'r') as f:
                existing = json.load(f)
        
        record = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        existing.append(record)
        
        # Keep last 50 records
        existing = existing[-50:]
        
        with open(monitor_file, 'w') as f:
            json.dump(existing, f, indent=2)
    
    @staticmethod
    def check_drift(baseline_metrics: dict, current_metrics: dict, threshold: float = 0.1) -> dict:
        """Check if model performance has drifted"""
        drift_report = {}
        
        for metric_name in baseline_metrics:
            if metric_name in current_metrics:
                baseline = baseline_metrics[metric_name]
                current = current_metrics[metric_name]
                
                if baseline != 0:
                    drift = abs(current - baseline) / abs(baseline)
                    drift_report[metric_name] = {
                        "baseline": baseline,
                        "current": current,
                        "drift_percent": drift * 100,
                        "alert": drift > threshold
                    }
        
        # Create alert if drift detected
        for metric, data in drift_report.items():
            if data["alert"]:
                DataQualityMonitor._create_alert(
                    f"MODEL_DRIFT_{metric}",
                    f"Model '{metric}' drift detected: {data['drift_percent']:.1f}%",
                    "warning"
                )
        
        return drift_report


# ============== DASHBOARD GENERATOR ==============
class DashboardGenerator:
    """Generate monitoring dashboard data"""
    
    @staticmethod
    def generate_dashboard():
        """Generate dashboard data for visualization"""
        
        # ETL Pipeline Summary
        etl_summary = PipelineMonitor.get_pipeline_summary("etl_pipeline")
        ml_summary = PipelineMonitor.get_pipeline_summary("ml_pipeline")
        system_metrics = ResourceMonitor.get_system_metrics()
        
        dashboard_data = {
            "generated_at": datetime.now().isoformat(),
            "etl_pipeline": etl_summary,
            "ml_pipeline": ml_summary,
            "system": system_metrics,
            "alerts": DashboardGenerator._get_recent_alerts()
        }
        
        # Save dashboard data
        dashboard_file = Config.DASHBOARD_DIR / "dashboard_data.json"
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        return dashboard_data
    
    @staticmethod
    def _get_recent_alerts() -> list:
        """Get recent alerts"""
        alerts = []
        
        for alert_file in sorted(Config.ALERTS_DIR.glob("*.json"), reverse=True)[:10]:
            with open(alert_file, 'r') as f:
                alerts.append(json.load(f))
        
        return alerts
    
    @staticmethod
    def print_dashboard():
        """Print dashboard summary to console"""
        data = DashboardGenerator.generate_dashboard()
        
        print("\n" + "=" * 60)
        print("📊 PIPELINE MONITORING DASHBOARD")
        print("=" * 60)
        
        print(f"\n⏰ Generated at: {data['generated_at']}")
        
        # ETL Pipeline
        print("\n🔄 ETL PIPELINE")
        etl = data.get('etl_pipeline', {})
        if 'error' not in etl:
            print(f"   Total Runs: {etl.get('total_runs', 0)}")
            print(f"   Success Rate: {etl.get('success_rate', 0)*100:.1f}%")
            print(f"   Avg Duration: {etl.get('avg_duration_seconds', 0):.1f}s")
        else:
            print(f"   Status: {etl.get('error')}")
        
        # ML Pipeline
        print("\n🤖 ML PIPELINE")
        ml = data.get('ml_pipeline', {})
        if 'error' not in ml:
            print(f"   Total Runs: {ml.get('total_runs', 0)}")
            print(f"   Success Rate: {ml.get('success_rate', 0)*100:.1f}%")
        else:
            print(f"   Status: {ml.get('error')}")
        
        # System Resources
        print("\n💻 SYSTEM RESOURCES")
        sys = data.get('system', {})
        print(f"   CPU: {sys.get('cpu', {}).get('percent', 0):.1f}%")
        print(f"   Memory: {sys.get('memory', {}).get('percent', 0):.1f}%")
        print(f"   Disk: {sys.get('disk', {}).get('percent', 0):.1f}%")
        
        # Alerts
        print("\n🚨 RECENT ALERTS")
        alerts = data.get('alerts', [])
        if alerts:
            for alert in alerts[:5]:
                severity = alert.get('severity', 'info')
                print(f"   [{severity.upper()}] {alert.get('message', '')}")
        else:
            print("   No recent alerts")
        
        print("\n" + "=" * 60)


# ============== MAIN ==============
if __name__ == "__main__":
    # Example: Run dashboard
    DashboardGenerator.print_dashboard()
