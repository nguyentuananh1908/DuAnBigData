# E-Commerce Big Data Analytics & Recommendation System

Pipeline Python xử lý dữ liệu hành vi thương mại điện tử, phân nhóm khách hàng,
ước lượng nguy cơ không quay lại trong 30 ngày và gợi ý sản phẩm theo danh mục.
Dự án chạy batch trên một máy, sử dụng Pandas, SQLite, Parquet và scikit-learn.

Hướng dẫn cài đặt chi tiết: [INSTALL.md](INSTALL.md).
Những gì đã kiểm chứng: [VALIDATION.md](VALIDATION.md).

## Chạy nhanh trên Windows

Mở PowerShell tại thư mục dự án:

```powershell
cd D:\BigData\DuAnBigData
python -m pip install -r requirements.txt
python run_pipeline.py --mode check
```

Đặt đủ 5 CSV vào `data/` rồi nạp demo sạch (xóa SQLite/Parquet/mô hình cũ, giữ CSV):

```powershell
python run_pipeline.py --reset --mode all --sample-chunks 2
python run_pipeline.py --mode monitor
python run_pipeline.py --mode inference --user-id 474232307
```

Hoặc cài phụ thuộc rồi kiểm tra bằng `.\setup.ps1`.

Xử lý toàn bộ dữ liệu (lâu hơn):

```powershell
python run_pipeline.py --reset --mode all
```

Các tệp đầu vào: `2019-Oct.csv`, `2019-Nov.csv`, `2019-Dec.csv`,
`2020-Jan.csv`, `2020-Feb.csv`. Không đưa CSV và database vào Git.

ETL bỏ qua tệp không đổi đã xử lý với cùng giới hạn mẫu, hoặc đã xử lý toàn bộ.
Tăng từ mẫu lên toàn bộ sẽ thay thế dữ liệu của tệp đó, không nối thêm bản sao.
Chạy mẫu sau khi đã nạp toàn bộ sẽ giữ nguyên dữ liệu đầy đủ. Muốn có một bộ
mẫu riêng, dùng runtime riêng như hướng dẫn phía dưới.

## Các chế độ

- `--mode check`: kiểm tra Python, thư viện và 5 tệp CSV.
- `--mode all`: ETL → Parquet → đặc trưng → huấn luyện → gợi ý → Customer 360 → giám sát.
- `--mode etl`: nạp dữ liệu và xuất toàn bộ sự kiện ra Parquet.
- `--mode ml`: tạo đặc trưng, huấn luyện hai mô hình và cập nhật gợi ý.
- `--mode inference --user-id ID`: trả hồ sơ Customer 360 ra terminal.
- `--mode monitor`: kiểm tra dữ liệu, mô hình, tài nguyên và cảnh báo; ghi `monitoring/latest_health.json`.
- `--mode serve --port 8000`: chạy HTTP API cục bộ; dừng bằng Ctrl+C.
- `--reset`: xóa DB/Parquet/mô hình/monitoring dẫn xuất trước khi chạy; không xóa CSV.

Lỗi thực thi làm CLI trả exit code 1. Đầu vào CLI không hợp lệ trả exit code 2.
Cảnh báo dữ liệu thô không phải lỗi thực thi nếu dữ liệu đã làm sạch hợp lệ.

## HTTP API cục bộ

```powershell
python run_pipeline.py --mode serve
```

Mở PowerShell thứ hai:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/customers/474232307

$body = @{ user_ids = @(474232307, 123) } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/batch -Method Post -ContentType 'application/json' -Body $body
```

API trả JSON: 200 thành công, 400 đầu vào sai, 404 không tìm thấy, 503 thiếu
hoặc không đọc được dữ liệu/mô hình. Batch giới hạn 100 ID, liệt kê ID không có
trong dữ liệu ở `not_found`. Đây là máy chủ demo chỉ lắng nghe localhost;
chưa bao gồm xác thực, TLS hay triển khai dịch vụ công khai.
Không huấn luyện đồng thời với phiên API đang phục vụ.

## Luồng dữ liệu và tính toàn vẹn

1. Đọc CSV theo chunk 100.000 dòng; ID danh mục giữ dạng chuỗi, giá dùng float64.
2. Kiểm tra chất lượng từng chunk thô; lưu báo cáo để truy nguyên cảnh báo.
3. Loại dòng thiếu khóa bắt buộc, loại sự kiện sai, session thiếu/ngắn,
   thời gian ngoài 01/10/2019–29/02/2020, giá không hữu hạn hoặc không dương.
4. Chuẩn hóa UTC và danh mục 3 cấp. Không tự tạo giá mua hàng hoặc cắt bỏ
   giao dịch hợp lệ chỉ vì giá cao hơn phân vị 99%.
5. Kiểm tra lại chunk sạch trước khi ghi. Khử bản ghi trùng trong và giữa
   các chunk của cùng tệp bằng khóa gồm thời gian, sự kiện, sản phẩm,
   người dùng, session, giá, danh mục và thương hiệu.
6. Ghi vào bảng tạm trên đĩa. Chỉ thay thế dữ liệu của nguồn và cập nhật
   manifest trong một giao dịch sau khi đọc thành công toàn tệp/phạm vi mẫu.
   Nếu lỗi giữa chừng, dữ liệu nguồn cũ được giữ nguyên.
7. Xuất Parquet theo chunk qua tệp tạm rồi thay thế đầu ra hoàn chỉnh.

Manifest nhận diện thay đổi bằng kích thước, thời gian sửa tệp và phiên bản ETL;
không phải checksum nội dung. Tệp nguồn phải đứng yên trong lúc nạp.
Chống trùng giữa hai tệp nguồn khác nhau chưa được thực hiện; đầu vào dự kiến
là các tệp tháng không chồng lấn. Tính nguyên tử áp dụng cho từng tệp,
không áp dụng cho toàn bộ 5 tệp cùng lúc. Chạy lại sau lỗi để hoàn thành các nguồn còn lại.

## Đặc trưng và mô hình

### Phân nhóm khách hàng

K-Means trên các đặc trưng hành vi đã chuẩn hóa: số sự kiện, tổng/giá trị
mua trung bình, số sản phẩm và số sự kiện/ngày. Tên nhóm được tạo từ thống kê
thực tế của cụm; không gán sẵn nhãn VIP cho một số hiệu cụm tùy ý.

`total_spent` chỉ cộng giá các sự kiện `purchase`; `avg_spent` chỉ lấy trung bình
các sự kiện mua. Một sự kiện mua được xem là một đơn vị sản phẩm vì dữ liệu
không có số lượng hàng và mã đơn hàng. Do đó đây không phải báo cáo doanh thu kế toán.

### Churn 30 ngày

- Lấy mốc lớn nhất trong dữ liệu trừ 30 ngày làm cutoff.
- Chỉ dùng hành vi **trước cutoff** để tạo đặc trưng huấn luyện.
- Nhãn 1: khách hàng đó không có tương tác từ cutoff đến mốc cuối dữ liệu.
- Chia người dùng thành train/test có phân tầng; cố định random seed.
- Báo cáo accuracy, precision, recall, F1, balanced accuracy, ROC AUC,
  average precision và baseline dự đoán lớp phổ biến nhất.

Đây là kiểm chứng trên người dùng giữ lại ở một cutoff lịch sử; chưa phải
backtest nhiều thời điểm hoặc bảo đảm hiệu quả trên dữ liệu tương lai.
Không suy diễn F1 cao là mô hình tốt nếu không so với baseline và mất cân bằng lớp.
Mẫu lấy từ đầu mỗi tệp không đại diện cho cả tháng; kết quả demo chỉ kiểm tra chức năng.

Khi phục vụ, mô hình nhận lịch sử hiện có của khách hàng để ước lượng rủi ro.
`risk_drivers` là các quy tắc mô tả hành vi, không phải giải thích nhân quả của mô hình.
Mô hình phiên bản cũ bị từ chối nếu khác schema đặc trưng; chạy `--mode ml` để cập nhật.

### Gợi ý

Xếp hạng sản phẩm theo số sự kiện mua trong danh mục yêu thích, bổ sung sản phẩm
phổ biến ở danh mục khác nếu thiếu. Đây là popularity-based recommender;
chưa có collaborative filtering hay đánh giá offline precision@K.

### Quản lý mô hình

Mỗi lần đăng ký có version timestamp, metrics và tham số. Bản mô hình lịch sử
được lưu trong `models/versions/`; mô hình hiện hành nằm trực tiếp trong `models/`.
Registry JSON và bảng SQLite `model_performance` lưu thông tin truy vết.
Cache suy luận tải lại khi tệp mô hình thay đổi. Chỉ nạp pickle do dự án tạo ra.

## Đầu ra

- `data/bigdata.db`: sự kiện, manifest nạp, lịch sử pipeline, chất lượng, metrics, gợi ý.
- `data_clean/events.parquet`: toàn bộ sự kiện sau làm sạch.
- `data_clean/user_features.parquet`: mẫu đặc trưng mô tả dùng trong lần huấn luyện CLI.
- `models/*.pkl`, `models/model_registry.json`: mô hình hiện hành và registry.
- `models/versions/`: bản mô hình theo lần đăng ký.
- `logs/`: log thực thi.
- `monitoring/latest_health.json`: ảnh chụp tình trạng ở lần chạy giám sát gần nhất.
- `backups/`: bản sao trước thay đổi dữ liệu/mô hình trong lần nâng cấp này.

## Runtime riêng và sao lưu

Có thể dùng dữ liệu thô chung nhưng database, mô hình và đầu ra riêng:

```powershell
$env:BIGDATA_RUNTIME_DIR = "$PWD\.test-runs\demo"
$env:BIGDATA_DATA_DIR = "$PWD\data"
python run_pipeline.py --mode all --sample-chunks 2
```

`BIGDATA_DB_PATH` cho phép chỉ định database riêng nếu cần. Đặt biến trước khi
chạy Python hoặc trước khi import `src` trong notebook. Để về cấu hình mặc định:

```powershell
Remove-Item Env:BIGDATA_RUNTIME_DIR -ErrorAction SilentlyContinue
Remove-Item Env:BIGDATA_DATA_DIR -ErrorAction SilentlyContinue
Remove-Item Env:BIGDATA_DB_PATH -ErrorAction SilentlyContinue
```

Không cần xóa database để chạy ETL lại. Khi sao lưu database đang sử dụng WAL,
dùng SQLite backup API; tránh chỉ sao chép tệp `.db` khi còn tiến trình ghi.

## Notebook

```powershell
python -m jupyter notebook
```

- `BigData_Pipeline_Main.ipynb`: tổng quan hệ thống và Customer 360.
- `ETL_Pipeline.ipynb`: đầu vào, chất lượng, nạp và xuất mẫu Parquet.
- `ML_Pipeline.ipynb`: huấn luyện, metrics, biểu đồ và registry.
- `Application_Demo.ipynb`: hồ sơ, gợi ý và batch scoring.
- `Monitoring_Dashboard.ipynb`: tình trạng database và báo cáo vận hành.
- `Database_Design.ipynb`: tài liệu thiết kế tham khảo, không phải migration schema.
- `Data_Pipeline_Design.ipynb`: ví dụ đọc mẫu và làm sạch bằng module chung.

Chạy Jupyter tại thư mục gốc dự án. Notebook ML thay đổi mô hình hiện hành;
restart kernel nếu vừa thay mã nguồn. Các giá trị/biểu đồ cũ đã được xóa để tránh
hiển thị kết quả của mô hình dùng đặc trưng cũ.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Kiểm thử dùng database và mô hình tạm dưới `.test-runs/`, không sửa dữ liệu chính.
Bao phủ nạp lặp, trùng xuyên chunk, tăng mẫu lên toàn bộ, rollback khi đầu vào lỗi,
chi tiêu mua hàng, tách nhãn tương lai, dữ liệu không hợp lệ, Parquet, huấn luyện
và suy luận tích hợp, HTTP status và xác thực đầu vào. Test API cần quyền kết nối localhost.

## Airflow (tùy chọn)

`dags/bigdata_pipeline_dag.py` định nghĩa lịch mỗi ngày, tối đa một DAG run hoạt động.
Các bước huấn luyện chạy nối tiếp để tránh cùng ghi registry JSON. DAG nạp toàn bộ
nguồn thay đổi, xuất Parquet đầy đủ, tạo đặc trưng và mô hình rồi kiểm tra sức khỏe.

CLI và notebook không cần Airflow. Khi chưa cài Airflow, module DAG có
`AIRFLOW_AVAILABLE=False` và `dag=None`, không giả lập rằng lịch đang chạy.
Cài và cấu hình Airflow trong môi trường riêng phù hợp với phiên bản Python;
đảm bảo scheduler/worker thấy cùng thư mục dữ liệu và package `src`.
DAG hiện dùng giao diện Airflow 2.x (`schedule_interval`, `airflow.operators.python`).
Việc kiểm chứng CLI không thay thế kiểm thử trên một scheduler Airflow thực tế.

## Phạm vi triển khai

Dự án cung cấp pipeline batch, demo ML, API localhost và notebook. Không có
Kafka/Spark/Hadoop, cụm phân tán, tự động phát hiện model drift, email cảnh báo,
hay hệ thống production nhiều người dùng. Giám sát hiện ghi log và JSON cục bộ.
Xem `VALIDATION.md` để biết những kiểm chứng đã thực hiện và giới hạn còn lại.
Xem `INSTALL.md` nếu cài máy mới.
