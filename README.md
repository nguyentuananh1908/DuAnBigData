# DuAnBigData

# 🛒 E-Commerce Big Data Analytics & Recommendation System

> Xây dựng hệ thống xử lý, phân tích và khai thác dữ liệu hành vi người dùng thương mại điện tử sử dụng Big Data, Machine Learning và Visualization.

---

## 📌 1. Giới thiệu dự án

Dự án xây dựng một nền tảng Big Data nhằm thu thập, xử lý, lưu trữ và phân tích dữ liệu hành vi người dùng trên hệ thống thương mại điện tử.

Dữ liệu bao gồm các sự kiện phát sinh trong quá trình người dùng tương tác với sản phẩm như:

- View sản phẩm
- Add to Cart
- Purchase
- Thông tin sản phẩm
- Danh mục sản phẩm
- Giá sản phẩm
- Người dùng
- Session của người dùng
- Thời gian xảy ra sự kiện

Hệ thống được thiết kế theo pipeline:

```text
Raw Data
   ↓
Data Ingestion
   ↓
ETL / ELT Processing
   ↓
Data Lake / Data Warehouse
   ↓
Machine Learning
   ↓
Visualization
