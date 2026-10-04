# 🛒 E-Commerce Big Data Analytics & Recommendation System

> Xây dựng hệ thống xử lý, phân tích và khai thác dữ liệu hành vi người dùng thương mại điện tử sử dụng Big Data, Machine Learning và Visualization.

---

## 📌 1. Giới thiệu dự án

Dự án xây dựng một nền tảng Big Data nhằm thu thập, xử lý, lưu trữ và phân tích dữ liệu hành vi người dùng trên hệ thống thương mại điện tử.

### Nguồn dữ liệu
- **Dataset:** E-commerce User Behavior (Kaggle)
- **Thời gian:** 2019-Oct → 2020-Feb (5 tháng)
- **Sự kiện:** View, Cart, Purchase
- **Dữ liệu:** User ID, Product ID, Category, Brand, Price, Session, Timestamp

---

## 📊 2. Pipeline Architecture

```
Raw Data (CSV)
     ↓
┌─────────────────┐
│  ETL Pipeline   │  ← ETL_Pipeline.ipynb
│  Extract        │
│  Transform      │
│  Load           │
└─────────────────┘
     ↓
┌─────────────────┐
│  Data Clean     │
│  (CSV/PostgreSQL)│
└─────────────────┘
     ↓
┌─────────────────┐
│  ML Pipeline    │  ← ML_Pipeline.ipynb
│  - Segmentation │
│  - Churn        │
│  - Recommendation│
└─────────────────┘
     ↓
┌─────────────────┐
│  Monitoring     │  ← Monitoring_Dashboard.ipynb
│  - Metrics      │
│  - Alerts       │
│  - Dashboard    │
└─────────────────┘
```

---

## 📁 3. Cấu trúc Project

```
DuAnBigData/
│
├── 📓 BigData_Pipeline_Main.ipynb    # Tổng quan kiến trúc
├── 📓 ETL_Pipeline.ipynb              # Extract → Transform → Load
├── 📓 ML_Pipeline.ipynb                # ML Models
├── 📓 Monitoring_Dashboard.ipynb       # Monitoring Dashboard
│
├── 📁 data/                           # Raw data (CSV files)
├── 📁 data_clean/                     # Cleaned data sau ETL
├── 📁 models/                         # Trained ML models
├── 📁 logs/                           # Pipeline logs
├── 📁 monitoring/                     # Metrics & alerts
│
├── 📓 Database_Design.ipynb           # Thiết kế Database
├── 📓 Data_Pipeline_Design.ipynb      # Thiết kế Pipeline
│
└── 📄 README.md                       # This file
```

---

## 🔄 4. Chi tiết từng thành phần

### 4.1 ETL Pipeline (`ETL_Pipeline.ipynb`)

**Quy trình xử lý dữ liệu thô thành dữ liệu sạch:**

| Bước | Mô tả |
|------|-------|
| **Extract** | Đọc CSV files (2019-Oct → 2020-Feb) theo chunks |
| **Remove Duplicates** | Loại bỏ dòng trùng lặp |
| **Handle Missing Values** | Fill category_code/brand='unknown', price=median |
| **Standardize** | Lowercase, trim spaces |
| **Validate Sessions** | Filter sessions có độ dài hợp lệ |
| **Remove Outliers** | Loại bỏ giá trị bất thường (price 1%-99%) |
| **Create Features** | Tạo date, hour, day_of_week, category_level1/2/3 |
| **Load** | Xuất ra `data_clean/events_clean.csv` |

**Kết quả:**
- ✅ ~7 triệu rows sau ETL
- ✅ Categories chuẩn hóa
- ✅ Outliers removed
- ✅ Features sẵn sàng cho ML

### 4.2 ML Pipeline (`ML_Pipeline.ipynb`)

**3 Models đã train:**

| Model | Algorithm | Mục đích |
|-------|----------|----------|
| **Customer Segmentation** | K-Means (5 clusters) | Phân khách hàng theo hành vi |
| **Churn Prediction** | Random Forest | Dự đoán khách hàng rời bỏ |
| **Product Recommendation** | Popularity-based | Gợi ý sản phẩm theo category |

**Customer Segmentation:**
- Silhouette Score đánh giá chất lượng phân cụm
- Features: total_events, total_spent, unique_products, events_per_day

**Churn Prediction:**
- Churn rate (30 ngày không hoạt động)
- Metrics: Accuracy, Precision, Recall, F1
- Feature importance để hiểu yếu tố churn

**Model Registry:**
- Lưu trữ metadata models (MLflow-like)
- Version tracking

### 4.3 Monitoring Dashboard (`Monitoring_Dashboard.ipynb`)

**Theo dõi 5 metrics chính:**

| Metric | Mô tả |
|--------|-------|
| **Processing Latency** | Thời gian chạy pipeline |
| **Error Rate** | Tỷ lệ lỗi pipeline |
| **Missing Data Rate** | Tỷ lệ missing values |
| **CPU/RAM Usage** | Tài nguyên hệ thống |
| **Model Accuracy** | Performance của ML models |

**Alerts:**
- Tự động cảnh báo khi vượt ngưỡng
- Lưu alert history

---

## ⚙️ 5. Công nghệ sử dụng

| Layer | Công nghệ |
|-------|-----------|
| **Processing** | Python, Pandas, NumPy |
| **ML** | Scikit-learn, K-Means, Random Forest |
| **Storage** | CSV, PostgreSQL (optional) |
| **Notebook** | Jupyter Notebook (.ipynb) |
| **Visualization** | Matplotlib, Seaborn |

---

## 🚀 6. Cách chạy

### Cách 1: Chạy từng notebook theo thứ tự

```bash
# 1. Mở Jupyter Notebook
jupyter notebook

# 2. Chạy theo thứ tự:
#    BigData_Pipeline_Main.ipynb  → Tổng quan
#    ETL_Pipeline.ipynb           → Làm sạch dữ liệu
#    ML_Pipeline.ipynb             → Train models
#    Monitoring_Dashboard.ipynb    → Xem dashboard
```

### Cách 2: Thứ tự nhanh

1. Mở `ETL_Pipeline.ipynb` → Run All
2. Mở `ML_Pipeline.ipynb` → Run All  
3. Mở `Monitoring_Dashboard.ipynb` → Run All

---

## 📈 7. Kết quả đạt được

### Đã hoàn thành:
- ✅ ETL Pipeline hoàn chỉnh
- ✅ Data Quality checks (missing, duplicates, outliers)
- ✅ Customer Segmentation (K-Means)
- ✅ Churn Prediction (Random Forest)
- ✅ Product Recommendation
- ✅ Monitoring Dashboard
- ✅ Model Registry (MLflow-like)
- ✅ Documentation đầy đủ

### Chưa triển khai (tùy chọn):
- ⏳ PostgreSQL integration (code có sẵn, uncomment để dùng)
- ⏳ Apache Spark/Flink (chưa cần - Pandas đủ cho dataset hiện tại)
- ⏳ Cloud deployment (AWS/GCP/Azure)
- ⏳ Real-time streaming

---

## 👥 Team

| Thành viên | Vai trò |
|------------|---------|
| BigData Team | Data Engineer, ML Engineer, Analyst |

---

## 📅 Cập nhật

- **04/10/2026:** Hoàn thành ETL, ML, Monitoring Pipelines (.ipynb)
- **04/10/2026:** Hoàn thành Database Design
- **04/10/2026:** Hoàn thành Data Pipeline Design

---

## 📖 Tham khảo

- [Kaggle E-commerce Dataset](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
