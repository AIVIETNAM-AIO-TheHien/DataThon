# Task 3 figures — hướng dẫn chạy nhanh

Đặt `make_task3_figures.py` vào thư mục gốc repo, cùng cấp với `run_pipeline.py` và thư mục `src/`.

Chạy pipeline trước để có `submission.csv`:

```powershell
python run_pipeline.py
```

Sau đó sinh figure:

```powershell
python make_task3_figures.py --raw-dir data/raw --submission submission.csv --out-dir report_figures
```

Nếu máy chưa cài LightGBM hoặc muốn chạy nhanh, bỏ qua figure cần train model:

```powershell
python make_task3_figures.py --raw-dir data/raw --submission submission.csv --out-dir report_figures --skip-model-figures
```
