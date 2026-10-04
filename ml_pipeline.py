"""
Machine Learning Pipeline cho E-commerce
- Recommendation System
- Churn Prediction
- Customer Segmentation

Author: BigData Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
import pickle
import json

# ============== CONFIG ==============
class Config:
    DATA_DIR = Path("data_clean")
    MODEL_DIR = Path("models")
    LOG_DIR = Path("logs")
    
    # Model settings
    N_CLUSTERS = 5  # For customer segmentation
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

# Setup
Config.MODEL_DIR.mkdir(exist_ok=True)
Config.LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_DIR / "ml_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============== LOAD DATA ==============
def load_clean_data():
    """Load cleaned data từ ETL pipeline"""
    logger.info("Loading cleaned data for ML...")
    
    data_path = Config.DATA_DIR / "events_clean.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Clean data not found: {data_path}")
    
    df = pd.read_csv(data_path, parse_dates=['event_time'])
    logger.info(f"Loaded {len(df):,} rows")
    
    return df


def prepare_features(df):
    """Chuẩn bị features cho ML models"""
    logger.info("Preparing features...")
    
    # User-level aggregation
    user_features = df.groupby('user_id').agg({
        'event_time': ['min', 'max', 'count'],
        'price': ['sum', 'mean', 'std'],
        'product_id': 'nunique',
        'event_type': lambda x: (x == 'view').sum(),
        'category_level1': 'nunique'
    }).reset_index()
    
    # Flatten column names
    user_features.columns = [
        'user_id', 'first_interaction', 'last_interaction', 'total_events',
        'total_spent', 'avg_spent', 'std_spent', 'unique_products',
        'total_views', 'unique_categories'
    ]
    
    # Calculate engagement metrics
    user_features['days_active'] = (
        user_features['last_interaction'] - user_features['first_interaction']
    ).dt.days + 1
    
    user_features['events_per_day'] = user_features['total_events'] / user_features['days_active']
    user_features['avg_session_value'] = user_features['total_spent'] / user_features['total_events']
    
    # Fill NaN
    user_features = user_features.fillna(0)
    
    logger.info(f"Prepared features for {len(user_features):,} users")
    
    return user_features


# ============== CUSTOMER SEGMENTATION (K-Means) ==============
def train_customer_segmentation(user_features):
    """
    Customer Segmentation sử dụng K-Means Clustering
    Phân khách hàng thành các nhóm theo hành vi mua sắm
    """
    logger.info("=" * 50)
    logger.info("TRAINING: Customer Segmentation (K-Means)")
    logger.info("=" * 50)
    
    # Select features for clustering
    feature_cols = [
        'total_events', 'total_spent', 'avg_spent', 
        'unique_products', 'unique_categories', 'events_per_day'
    ]
    
    X = user_features[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train K-Means
    kmeans = KMeans(
        n_clusters=Config.N_CLUSTERS,
        random_state=Config.RANDOM_STATE,
        n_init=10
    )
    
    user_features['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Evaluate
    silhouette = silhouette_score(X_scaled, user_features['cluster'])
    logger.info(f"Silhouette Score: {silhouette:.4f}")
    
    # Analyze clusters
    cluster_summary = user_features.groupby('cluster').agg({
        'user_id': 'count',
        'total_spent': 'mean',
        'total_events': 'mean',
        'unique_products': 'mean',
        'events_per_day': 'mean'
    }).round(2)
    
    cluster_summary.columns = ['count', 'avg_spent', 'avg_events', 'avg_products', 'avg_daily_events']
    logger.info("\nCluster Summary:")
    logger.info(f"\n{cluster_summary}")
    
    # Save model
    model_path = Config.MODEL_DIR / "customer_segmentation_kmeans.pkl"
    scaler_path = Config.MODEL_DIR / "customer_segmentation_scaler.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info(f"Models saved to {Config.MODEL_DIR}")
    
    # Save cluster labels
    user_features[['user_id', 'cluster']].to_csv(
        Config.DATA_DIR / "user_clusters.csv", index=False
    )
    
    return kmeans, scaler, user_features


# ============== CHURN PREDICTION ==============
def prepare_churn_labels(df, threshold_days=30):
    """
    Chuẩn bị churn labels
    Churn = user không có hoạt động trong 30 ngày cuối
    """
    logger.info(f"Preparing churn labels (threshold: {threshold_days} days)...")
    
    max_date = df['event_time'].max()
    cutoff_date = max_date - pd.Timedelta(days=threshold_days)
    
    # Find active users (after cutoff)
    active_users = df[df['event_time'] >= cutoff_date]['user_id'].unique()
    
    # Create user activity summary
    user_activity = df.groupby('user_id').agg({
        'event_time': 'max',
        'user_session': 'nunique'
    }).reset_index()
    
    user_activity.columns = ['user_id', 'last_activity', 'total_sessions']
    
    # Label churn
    user_activity['churned'] = (~user_activity['user_id'].isin(active_users)).astype(int)
    
    churn_rate = user_activity['churned'].mean()
    logger.info(f"Churn rate: {churn_rate:.2%}")
    
    return user_activity


def train_churn_prediction(user_features, user_activity):
    """
    Churn Prediction sử dụng Random Forest
    Dự đoán khách hàng nào sẽ churn
    """
    logger.info("=" * 50)
    logger.info("TRAINING: Churn Prediction (Random Forest)")
    logger.info("=" * 50)
    
    # Merge features with churn labels
    model_data = user_features.merge(
        user_activity[['user_id', 'churned', 'total_sessions']], 
        on='user_id', 
        how='inner'
    )
    
    # Drop non-numeric columns
    feature_cols = [
        'total_events', 'total_spent', 'avg_spent', 'std_spent',
        'unique_products', 'total_views', 'unique_categories',
        'days_active', 'events_per_day', 'avg_session_value', 'total_sessions'
    ]
    
    X = model_data[feature_cols].fillna(0)
    y = model_data['churned']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=Config.RANDOM_STATE,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    logger.info(f"\nModel Performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\nTop 5 Important Features:")
    logger.info(f"\n{importance.head()}")
    
    # Save model
    model_path = Config.MODEL_DIR / "churn_prediction_rf.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    
    logger.info(f"Model saved to {model_path}")
    
    # Save predictions
    model_data['churn_probability'] = rf_model.predict_proba(X)[:, 1]
    model_data[['user_id', 'churned', 'churn_probability']].to_csv(
        Config.DATA_DIR / "churn_predictions.csv", index=False
    )
    
    return rf_model, metrics


# ============== PRODUCT RECOMMENDATION ==============
def train_recommendation_system(df):
    """
    Collaborative Filtering cho Product Recommendation
    Gợi ý sản phẩm dựa trên hành vi của users có similar preferences
    """
    logger.info("=" * 50)
    logger.info("TRAINING: Product Recommendation (Collaborative Filtering)")
    logger.info("=" * 50)
    
    # Create user-product interaction matrix
    purchase_df = df[df['event_type'] == 'purchase'][
        ['user_id', 'product_id', 'event_time']
    ].drop_duplicates()
    
    # Count interactions
    interaction_counts = purchase_df.groupby(['user_id', 'product_id']).size().reset_index(name='count')
    
    logger.info(f"Created interaction matrix: {len(interaction_counts):,} user-product pairs")
    
    # Simple popularity-based recommendation
    product_popularity = df[df['event_type'] == 'purchase'].groupby(
        ['category_level1', 'product_id']
    ).size().reset_index(name='popularity')
    
    # Get top products per category
    top_products = product_popularity.sort_values(
        ['category_level1', 'popularity'], ascending=[True, False]
    ).groupby('category_level1').head(10)
    
    logger.info(f"\nTop 10 products per category saved")
    
    # Save recommendations
    top_products.to_csv(
        Config.DATA_DIR / "product_recommendations.csv", index=False
    )
    
    # Also save user-based recommendations
    user_purchases = purchase_df.groupby('user_id')['product_id'].apply(list).to_dict()
    
    # Save user purchase history
    purchase_history = pd.DataFrame([
        {'user_id': uid, 'purchased_products': ','.join(map(str, pids))}
        for uid, pids in user_purchases.items()
    ])
    
    purchase_history.to_csv(
        Config.DATA_DIR / "user_purchase_history.csv", index=False
    )
    
    logger.info(f"User purchase history saved: {len(purchase_history):,} users")
    
    return top_products, purchase_history


# ============== MODEL REGISTRY (MLflow-like) ==============
def save_model_registry():
    """Save model registry metadata (simple version of MLflow)"""
    
    registry = {
        "models": [
            {
                "name": "customer_segmentation_kmeans",
                "type": "clustering",
                "algorithm": "KMeans",
                "n_clusters": Config.N_CLUSTERS,
                "created_at": datetime.now().isoformat(),
                "metrics": {
                    "silhouette_score": "see logs"
                }
            },
            {
                "name": "churn_prediction_rf",
                "type": "classification",
                "algorithm": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 10,
                "created_at": datetime.now().isoformat()
            },
            {
                "name": "product_recommendation_popularity",
                "type": "recommendation",
                "algorithm": "Popularity-based",
                "created_at": datetime.now().isoformat()
            }
        ],
        "latest_versions": {
            "customer_segmentation_kmeans": "v1.0",
            "churn_prediction_rf": "v1.0",
            "product_recommendation_popularity": "v1.0"
        }
    }
    
    registry_path = Config.MODEL_DIR / "model_registry.json"
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Model registry saved to {registry_path}")
    
    return registry


# ============== MAIN ML PIPELINE ==============
def run_ml_pipeline():
    """Main ML Pipeline"""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("ML PIPELINE STARTED")
    logger.info(f"Start time: {start_time}")
    logger.info("=" * 60)
    
    try:
        # Load data
        df = load_clean_data()
        
        # Prepare features
        user_features = prepare_features(df)
        
        # Train Customer Segmentation
        kmeans, scaler, user_features = train_customer_segmentation(user_features)
        
        # Prepare churn labels
        user_activity = prepare_churn_labels(df)
        
        # Train Churn Prediction
        churn_model, churn_metrics = train_churn_prediction(user_features, user_activity)
        
        # Train Recommendation System
        recommendations, purchase_history = train_recommendation_system(df)
        
        # Save model registry
        registry = save_model_registry()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("ML PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Duration: {duration}")
        logger.info("=" * 60)
        
        return {
            'user_features': user_features,
            'churn_metrics': churn_metrics,
            'recommendations': recommendations
        }
        
    except Exception as e:
        logger.error(f"ML Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    results = run_ml_pipeline()
    
    print("\n" + "=" * 50)
    print("ML PIPELINE SUMMARY")
    print("=" * 50)
    print(f"Models trained:")
    print("  1. Customer Segmentation (K-Means)")
    print("  2. Churn Prediction (Random Forest)")
    print("  3. Product Recommendation (Popularity-based)")
    print(f"\nChurn Prediction Accuracy: {results['churn_metrics']['accuracy']:.4f}")
