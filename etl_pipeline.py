"""
ETL Pipeline cho E-commerce Dataset
Raw Data -> Ingestion -> ETL/ELT -> Storage

Author: BigData Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import os
import psycopg2
from psycopg2 import sql
import warnings
warnings.filterwarnings('ignore')

# ============== CONFIG ==============
class Config:
    DATA_DIR = Path("data")
    LOG_DIR = Path("logs")
    
    # PostgreSQL connection
    DB_HOST = "localhost"
    DB_PORT = 5432
    DB_NAME = "ecommerce_db"
    DB_USER = "postgres"
    DB_PASSWORD = "postgres"
    
    # CSV files to process
    CSV_FILES = [
        "2019-Oct.csv",
        "2019-Nov.csv",
        "2019-Dec.csv",
        "2020-Jan.csv",
        "2020-Feb.csv"
    ]

# Setup logging
Config.LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_DIR / "etl_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============== EXTRACT ==============
def extract_csv(file_path: str, chunk_size: int = 100000) -> pd.DataFrame:
    """Đọc CSV file với chunk processing để tiết kiệm memory"""
    logger.info(f"Extracting: {file_path}")
    
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"  -> Loaded {len(df):,} rows")
    return df


def extract_all_data() -> pd.DataFrame:
    """Extract tất cả CSV files"""
    logger.info("=" * 50)
    logger.info("STARTING ETL PIPELINE - EXTRACT PHASE")
    logger.info("=" * 50)
    
    all_data = []
    for csv_file in Config.CSV_FILES:
        file_path = Config.DATA_DIR / csv_file
        if file_path.exists():
            df = extract_csv(str(file_path))
            df['source_file'] = csv_file
            all_data.append(df)
        else:
            logger.warning(f"  File not found: {file_path}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total rows extracted: {len(combined_df):,}")
        return combined_df
    else:
        raise FileNotFoundError("No CSV files found!")


# ============== TRANSFORM ==============
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Xử lý missing values"""
    logger.info("Transforming: Handle missing values")
    
    # Count missing before
    missing_before = df.isnull().sum()
    
    # category_code: Fill with 'Unknown' hoặc extract từ category_id
    df['category_code'] = df['category_code'].fillna('unknown')
    
    # brand: Fill with 'Unknown'
    df['brand'] = df['brand'].fillna('unknown')
    
    # user_session: Fill với temporary session id
    df['user_session'] = df['user_session'].fillna(f"temp_session_{df.index}")
    
    # Price: Fill với median
    df['price'] = df['price'].fillna(df['price'].median())
    
    # event_time: Remove rows có missing
    df = df.dropna(subset=['event_time'])
    
    missing_after = df.isnull().sum()
    logger.info(f"  Missing values handled: {missing_before.sum()} -> {missing_after.sum()}")
    
    return df


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa categories"""
    logger.info("Transforming: Standardize categories")
    
    # Lowercase
    df['category_code'] = df['category_code'].str.lower().str.strip()
    df['brand'] = df['brand'].str.lower().str.strip()
    
    # Remove extra spaces
    df['category_code'] = df['category_code'].str.replace(r'\s+', ' ', regex=True)
    df['brand'] = df['brand'].str.replace(r'\s+', ' ', regex=True)
    
    # Fill empty strings
    df.loc[df['category_code'] == '', 'category_code'] = 'unknown'
    df.loc[df['brand'] == '', 'brand'] = 'unknown'
    
    logger.info(f"  Categories standardized")
    return df


def validate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate và clean sessions"""
    logger.info("Transforming: Validate sessions")
    
    # Remove rows with NULL session
    before_count = len(df)
    df = df[df['user_session'].notna()]
    df = df[df['user_session'] != '']
    
    # Validate session length (should be UUID-like)
    df = df[df['user_session'].str.len() >= 10]
    
    after_count = len(df)
    logger.info(f"  Sessions validated: {before_count:,} -> {after_count:,} rows")
    
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers bằng IQR method"""
    logger.info("Transforming: Remove outliers")
    
    before_count = len(df)
    
    # Price outliers
    Q1 = df['price'].quantile(0.01)
    Q3 = df['price'].quantile(0.99)
    df = df[(df['price'] >= Q1) & (df['price'] <= Q3)]
    
    # Validate timestamp (should be between 2019-10 and 2020-02)
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    df = df[df['event_time'].notna()]
    df = df[(df['event_time'] >= '2019-10-01') & (df['event_time'] <= '2020-02-29')]
    
    after_count = len(df)
    logger.info(f"  Outliers removed: {before_count:,} -> {after_count:,} rows")
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows"""
    logger.info("Transforming: Remove duplicates")
    
    before_count = len(df)
    df = df.drop_duplicates()
    
    after_count = len(df)
    logger.info(f"  Duplicates removed: {before_count:,} -> {after_count:,} rows")
    
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo derived features"""
    logger.info("Transforming: Create features")
    
    # Parse datetime
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # Date features
    df['date'] = df['event_time'].dt.date
    df['hour'] = df['event_time'].dt.hour
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['day_name'] = df['event_time'].dt.day_name()
    df['month'] = df['event_time'].dt.month
    df['year'] = df['event_time'].dt.year
    
    # Parse category hierarchy
    df['category_level1'] = df['category_code'].str.split('.').str[0]
    df['category_level2'] = df['category_code'].str.split('.').str.get(1)
    df['category_level3'] = df['category_code'].str.split('.').str.get(2)
    
    # Clean category level
    df['category_level1'] = df['category_level1'].fillna('unknown')
    df['category_level2'] = df['category_level2'].fillna('unknown')
    df['category_level3'] = df['category_level3'].fillna('unknown')
    
    logger.info(f"  Features created: date, hour, day_of_week, category levels")
    
    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Main transform function - gọi tất cả transform steps"""
    logger.info("=" * 50)
    logger.info("TRANSFORM PHASE")
    logger.info("=" * 50)
    
    initial_count = len(df)
    
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = standardize_categories(df)
    df = validate_sessions(df)
    df = remove_outliers(df)
    df = create_features(df)
    
    final_count = len(df)
    logger.info(f"Transform complete: {initial_count:,} -> {final_count:,} rows ({final_count/initial_count*100:.1f}% kept)")
    
    return df


# ============== LOAD ==============
def create_tables(conn):
    """Tạo tables trong PostgreSQL"""
    cursor = conn.cursor()
    
    # Drop existing tables
    cursor.execute("DROP TABLE IF EXISTS events CASCADE")
    cursor.execute("DROP TABLE IF EXISTS dim_users CASCADE")
    cursor.execute("DROP TABLE IF EXISTS dim_products CASCADE")
    cursor.execute("DROP TABLE IF EXISTS dim_categories CASCADE")
    cursor.execute("DROP TABLE IF EXISTS fact_sales_daily CASCADE")
    
    # Create events table
    cursor.execute("""
        CREATE TABLE events (
            event_id SERIAL PRIMARY KEY,
            event_time TIMESTAMP NOT NULL,
            event_type VARCHAR(20) NOT NULL,
            product_id BIGINT NOT NULL,
            category_id BIGINT,
            category_code VARCHAR(100),
            brand VARCHAR(50),
            price DECIMAL(10, 2),
            user_id BIGINT,
            user_session VARCHAR(100),
            source_file VARCHAR(50),
            date DATE,
            hour INT,
            day_of_week INT,
            day_name VARCHAR(20),
            month INT,
            year INT,
            category_level1 VARCHAR(50),
            category_level2 VARCHAR(50),
            category_level3 VARCHAR(50)
        )
    """)
    
    # Create dim_users
    cursor.execute("""
        CREATE TABLE dim_users (
            user_id BIGINT PRIMARY KEY,
            first_seen TIMESTAMP,
            last_seen TIMESTAMP,
            total_sessions INT,
            total_purchases INT,
            total_spent DECIMAL(12, 2)
        )
    """)
    
    # Create dim_products
    cursor.execute("""
        CREATE TABLE dim_products (
            product_id BIGINT PRIMARY KEY,
            category_id BIGINT,
            category_code VARCHAR(100),
            brand VARCHAR(50),
            avg_price DECIMAL(10, 2),
            min_price DECIMAL(10, 2),
            max_price DECIMAL(10, 2),
            total_views INT,
            total_carts INT,
            total_purchases INT
        )
    """)
    
    # Create dim_categories
    cursor.execute("""
        CREATE TABLE dim_categories (
            category_id BIGINT PRIMARY KEY,
            category_code VARCHAR(100),
            category_level1 VARCHAR(50),
            category_level2 VARCHAR(50),
            category_level3 VARCHAR(50),
            total_products INT,
            total_revenue DECIMAL(12, 2)
        )
    """)
    
    # Create fact_sales_daily
    cursor.execute("""
        CREATE TABLE fact_sales_daily (
            id SERIAL PRIMARY KEY,
            date_key DATE,
            product_id BIGINT,
            brand VARCHAR(50),
            category_level1 VARCHAR(50),
            total_views INT,
            total_carts INT,
            total_purchases INT,
            total_revenue DECIMAL(12, 2),
            unique_users INT
        )
    """)
    
    conn.commit()
    logger.info("Tables created successfully")


def load_to_postgres(df: pd.DataFrame):
    """Load cleaned data vào PostgreSQL"""
    logger.info("=" * 50)
    logger.info("LOAD PHASE - Loading to PostgreSQL")
    logger.info("=" * 50)
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            dbname=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        logger.info("Connected to PostgreSQL")
        
        # Create tables
        create_tables(conn)
        
        # Load events table (batch insert)
        logger.info("Loading events table...")
        
        # Prepare data for insert
        events_df = df[['event_time', 'event_type', 'product_id', 'category_id',
                        'category_code', 'brand', 'price', 'user_id', 'user_session',
                        'source_file', 'date', 'hour', 'day_of_week', 'day_name',
                        'month', 'year', 'category_level1', 'category_level2', 'category_level3']].copy()
        
        # Batch insert
        batch_size = 10000
        for i in range(0, len(events_df), batch_size):
            batch = events_df.iloc[i:i+batch_size]
            for _, row in batch.iterrows():
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO events (event_time, event_type, product_id, category_id,
                        category_code, brand, price, user_id, user_session,
                        source_file, date, hour, day_of_week, day_name,
                        month, year, category_level1, category_level2, category_level3)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    row['event_time'], row['event_type'], row['product_id'], row['category_id'],
                    row['category_code'], row['brand'], row['price'], row['user_id'], row['user_session'],
                    row['source_file'], row['date'], row['hour'], row['day_of_week'], row['day_name'],
                    row['month'], row['year'], row['category_level1'], row['category_level2'], row['category_level3']
                ))
            conn.commit()
            logger.info(f"  Loaded {min(i+batch_size, len(events_df)):,} / {len(events_df):,} events")
        
        logger.info("Events table loaded successfully!")
        
        # Create dimension tables
        logger.info("Creating dimension tables...")
        
        # dim_users aggregation
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dim_users (user_id, first_seen, last_seen, total_sessions, total_purchases, total_spent)
            SELECT 
                user_id,
                MIN(event_time) as first_seen,
                MAX(event_time) as last_seen,
                COUNT(DISTINCT user_session) as total_sessions,
                SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as total_purchases,
                SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) as total_spent
            FROM events
            WHERE user_id IS NOT NULL
            GROUP BY user_id
        """)
        conn.commit()
        logger.info("  dim_users created")
        
        # dim_products aggregation
        cursor.execute("""
            INSERT INTO dim_products (product_id, category_id, category_code, brand, avg_price, min_price, max_price, total_views, total_carts, total_purchases)
            SELECT 
                product_id,
                category_id,
                category_code,
                brand,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price,
                SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) as total_views,
                SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) as total_carts,
                SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as total_purchases
            FROM events
            GROUP BY product_id, category_id, category_code, brand
        """)
        conn.commit()
        logger.info("  dim_products created")
        
        # dim_categories aggregation
        cursor.execute("""
            INSERT INTO dim_categories (category_id, category_code, category_level1, category_level2, category_level3, total_products, total_revenue)
            SELECT 
                category_id,
                category_code,
                category_level1,
                category_level2,
                category_level3,
                COUNT(DISTINCT product_id) as total_products,
                SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) as total_revenue
            FROM events
            WHERE category_id IS NOT NULL
            GROUP BY category_id, category_code, category_level1, category_level2, category_level3
        """)
        conn.commit()
        logger.info("  dim_categories created")
        
        # fact_sales_daily aggregation
        cursor.execute("""
            INSERT INTO fact_sales_daily (date_key, product_id, brand, category_level1, total_views, total_carts, total_purchases, total_revenue, unique_users)
            SELECT 
                date as date_key,
                product_id,
                brand,
                category_level1,
                SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) as total_views,
                SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) as total_carts,
                SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as total_purchases,
                SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) as total_revenue,
                COUNT(DISTINCT user_id) as unique_users
            FROM events
            GROUP BY date, product_id, brand, category_level1
        """)
        conn.commit()
        logger.info("  fact_sales_daily created")
        
        # Get counts
        cursor.execute("SELECT COUNT(*) FROM events")
        events_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM dim_users")
        users_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM dim_products")
        products_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM dim_categories")
        categories_count = cursor.fetchone()[0]
        
        logger.info(f"Load complete: {events_count:,} events, {users_count:,} users, {products_count:,} products, {categories_count:,} categories")
        
        conn.close()
        logger.info("PostgreSQL connection closed")
        
    except Exception as e:
        logger.error(f"Error loading to PostgreSQL: {e}")
        raise


def save_clean_csv(df: pd.DataFrame, output_path: str = "data_clean/events_clean.csv"):
    """Lưu cleaned data ra CSV"""
    logger.info(f"Saving cleaned data to {output_path}")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"  Saved {len(df):,} rows to {output_path}")


# ============== MAIN ETL PIPELINE ==============
def run_etl_pipeline():
    """Main ETL pipeline"""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("ETL PIPELINE STARTED")
    logger.info(f"Start time: {start_time}")
    logger.info("=" * 60)
    
    try:
        # EXTRACT
        raw_df = extract_all_data()
        
        # TRANSFORM
        clean_df = transform_data(raw_df)
        
        # LOAD - Option 1: Save to CSV
        save_clean_csv(clean_df)
        
        # LOAD - Option 2: Load to PostgreSQL (uncomment if PostgreSQL is available)
        # load_to_postgres(clean_df)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Duration: {duration}")
        logger.info(f"Rows processed: {len(clean_df):,}")
        logger.info("=" * 60)
        
        return clean_df
        
    except Exception as e:
        logger.error(f"ETL Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # Chạy pipeline
    df = run_etl_pipeline()
    
    # Print summary
    print("\n" + "=" * 50)
    print("ETL PIPELINE SUMMARY")
    print("=" * 50)
    print(f"Total rows: {len(df):,}")
    print(f"Event types: {df['event_type'].value_counts().to_dict()}")
    print(f"Date range: {df['event_time'].min()} to {df['event_time'].max()}")
    print(f"Unique users: {df['user_id'].nunique():,}")
    print(f"Unique products: {df['product_id'].nunique():,}")
