# Kiểm chứng đã thực hiện

Cập nhật cùng lần hoàn thiện pipeline (mã nguồn + test + chạy demo có `--reset`).

## Đã kiểm

- Unit/integration: `python -m unittest discover -s tests -v`
  - Nạp lặp, trùng xuyên chunk, tăng mẫu lên toàn bộ
  - Giữ nguyên dữ liệu đầy đủ khi sau đó xin mẫu nhỏ hơn
  - Từ chối thu hẹp sample (ví dụ 2 chunk xuống 1 chunk)
  - Rollback khi chunk lỗi
  - `total_spent` chỉ cộng `purchase`; nhãn churn không dùng sự kiện sau cutoff
  - Giữ mua giá cao hợp lệ; loại giá âm / loại sự kiện sai
  - Parquet khớp số dòng SQLite
  - Huấn luyện + Customer 360
  - HTTP 200/400/404/503
- CLI: `--mode check`, `--reset`, `all` / `etl` / `ml` / `inference` / `monitor` / `serve`
- Demo runtime: `--reset --mode all --sample-chunks 2` trên 5 CSV thật (2 × 100.000 dòng/file)

## Giới hạn còn lại

- Mẫu lấy từ **đầu** mỗi tệp tháng, không đại diện cả tháng
- Chống trùng **giữa hai tệp nguồn** khác nhau chưa làm (các tháng không chồng lấn)
- Recommender theo độ phổ biến danh mục, chưa collaborative filtering / precision@K
- Churn: một cutoff lịch sử, chưa backtest nhiều thời điểm
- Giám sát ghi JSON/log cục bộ, không phát hiện model drift, không gửi email
- API chỉ localhost, không TLS/auth
- DAG Airflow chưa được chạy trên scheduler thật
- Manifest nhận diện tệp bằng kích thước + mtime + phiên bản ETL, không phải checksum nội dung
