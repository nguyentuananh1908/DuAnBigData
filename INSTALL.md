# Hướng dẫn cài đặt và chạy

Pipeline chạy trên **một máy Windows**, Python 3.10+, không cần Spark/Hadoop/Kafka.

## 1. Yêu cầu

- Windows 10/11, PowerShell
- Python **3.10 hoặc mới hơn** (`python --version`)
- RAM khuyến nghị 8 GB; ổ đĩa trống khoảng **10 GB** (CSV ~2,4 GB + SQLite + Parquet)
- Năm tệp CSV Kaggle eCommerce behavior:

  `2019-Oct.csv`, `2019-Nov.csv`, `2019-Dec.csv`, `2020-Jan.csv`, `2020-Feb.csv`

## 2. Mở thư mục dự án

```powershell
cd D:\BigData\DuAnBigData
```

## 3. Môi trường Python

Khuyến nghị dùng venv (không bắt buộc):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Hoặc cài thẳng vào Python hiện tại:

```powershell
python -m pip install -r requirements.txt
```

Phiên bản đã kiểm tra nằm trong `requirements-tested.txt`.

Nếu PowerShell chặn script venv:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 4. Đặt dữ liệu thô

Copy 5 CSV vào `data\`. Không commit CSV lên Git.

```powershell
Get-ChildItem data\*.csv | Select-Object Name, @{N='MB';E={[math]::Round($_.Length/1MB,1)}}
```

## 5. Kiểm tra máy

```powershell
python run_pipeline.py --mode check
```

Lệnh này xác nhận Python, thư viện, và 5 CSV. Exit code 0 = sẵn sàng.

## 6. Chạy demo đầy đủ (khuyến nghị lần đầu)

Nạp 2 chunk đầu (100.000 dòng/chunk) mỗi tháng, huấn luyện, suy luận, giám sát:

```powershell
python run_pipeline.py --reset --mode all --sample-chunks 2
```

`--reset` xóa SQLite/Parquet/mô hình/monitoring cũ, **không** xóa CSV. Lần chạy này thường vài phút.

Kết quả mong đợi:

- `data\bigdata.db` — sự kiện đã làm sạch + manifest
- `data_clean\events.parquet` và `user_features.parquet`
- `models\*.pkl` và `models\model_registry.json`
- `monitoring\latest_health.json`
- In ra một hồ sơ Customer 360

## 7. Dùng hệ thống

```powershell
python run_pipeline.py --mode inference --user-id 474232307
python run_pipeline.py --mode monitor
python run_pipeline.py --mode serve --port 8000
```

API chỉ lắng nghe `127.0.0.1` (cửa sổ PowerShell khác):

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/customers/474232307
```

Dừng API bằng Ctrl+C.

Nếu `474232307` không có trong mẫu, lấy một `user_id` có mua hàng:

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/bigdata.db'); print(c.execute(\"SELECT user_id FROM events WHERE event_type='purchase' LIMIT 1\").fetchone())"
```

## 8. Xử lý toàn bộ dữ liệu

Bỏ `--sample-chunks`. Có thể mất vài chục phút đến hơn một giờ tùy máy:

```powershell
python run_pipeline.py --reset --mode all
```

Chạy lại ETL khi CSV không đổi sẽ **bỏ qua** nguồn đã nạp (manifest). Tăng từ mẫu lên toàn bộ sẽ thay dữ liệu nguồn đó.

Muốn giữ bản đầy đủ và thử mẫu riêng, dùng runtime khác:

```powershell
$env:BIGDATA_RUNTIME_DIR = "$PWD\.demo-sample"
$env:BIGDATA_DATA_DIR = "$PWD\data"
python run_pipeline.py --mode all --sample-chunks 2
```

## 9. Notebook

Từ thư mục gốc dự án:

```powershell
python -m jupyter notebook
```

Bắt đầu với `BigData_Pipeline_Main.ipynb`. Restart kernel nếu vừa sửa `src/`.

## 10. Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Test dùng `.test-runs\`, không ghi đè `data\bigdata.db`.

## 11. Airflow (tùy chọn)

CLI không cần Airflow. DAG: `dags\bigdata_pipeline_dag.py` (Airflow 2.x). Cài Airflow trong môi trường riêng và trỏ cùng thư mục dự án.

## Sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `Missing input files` | Đặt đủ 5 CSV đúng tên vào `data\` |
| `Events table is empty` | Chạy `--mode etl` hoặc `--mode all` |
| `Model feature schema is outdated` | `python run_pipeline.py --mode ml` |
| SQLite bị khóa / file `-wal` | Đóng Jupyter/API rồi chạy lại; hoặc `--reset` |
| Dữ liệu sự kiện bị gấp đôi / manifest trống | `python run_pipeline.py --reset --mode all --sample-chunks 2` |
| Thiếu `pyarrow` khi xuất Parquet | `python -m pip install -r requirements.txt` |
| Port 8000 đã dùng | `--port 8001` |

Phạm vi: pipeline batch + demo ML + API localhost. Không gồm cụm phân tán, TLS, hay đăng nhập người dùng.
