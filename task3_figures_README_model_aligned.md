# Task 3 figures — model-aligned version

File chính: `make_task3_figures.py`

## Mục tiêu

Script này đã được chỉnh để figure khớp với source model hiện tại:

- Chạy theo đúng trình tự: `run_tuning.py` → `run_pipeline.py` → `make_task3_figures.py`.
- Dùng `src.data_prep.build_features()` giống model.
- Dùng đúng feature columns: loại `Date`, `Revenue`, `COGS` như trong pipeline.
- Dùng đúng sample weights: năm 2014–2018 có weight `1.0`, các năm còn lại `0.01`.
- Vẽ CV split bằng chính `src.cv_validation.time_series_split()` thay vì split tự chế.
- Validation figure dùng final ensemble giống pipeline: LightGBM base + Q-specialists + Ridge + calibration.
- Output mặc định là folder `reports` và có đúng 16 figure PNG.

## Cách chạy chuẩn

Đặt `make_task3_figures.py` ở thư mục gốc repo, cùng cấp với:

```text
run_tuning.py
run_pipeline.py
src/
data/raw/sales.csv
```

Chạy lần lượt:

```powershell
python run_tuning.py
python run_pipeline.py
python make_task3_figures.py --raw-dir data/raw --submission submission.csv --out-dir reports --clean
```

Sau khi chạy xong, kiểm tra folder:

```text
reports/
```

Trong đó có:

- 16 file `.png`
- `figure_captions.md`
- `figure_manifest.csv`
- `validation_metrics.csv`
- `feature_importance.csv`
- `validation_predictions_revenue.csv`
- `validation_predictions_cogs.csv`

## Chạy nhanh nếu máy yếu

Lệnh này vẫn tạo đủ 16 PNG, nhưng các hình cần train model sẽ là placeholder thông báo:

```powershell
python make_task3_figures.py --raw-dir data/raw --submission submission.csv --out-dir reports --clean --skip-model-figures
```

## 16 figure được tạo

1. `fig_01_monthly_history_and_forecast.png`
2. `fig_02_revenue_year_month_heatmap.png`
3. `fig_03_cogs_year_month_heatmap.png`
4. `fig_04_average_daily_by_month.png`
5. `fig_05_average_revenue_by_weekday.png`
6. `fig_06_calendar_event_impact.png`
7. `fig_07_cv_folds_from_code.png`
8. `fig_08_lgb_internal_split.png`
9. `fig_09_sample_weights.png`
10. `fig_10_feature_importance_revenue.png`
11. `fig_11_feature_importance_cogs.png`
12. `fig_12_validation_actual_pred_revenue.png`
13. `fig_13_validation_actual_pred_cogs.png`
14. `fig_14_validation_residuals_final_ensemble.png`
15. `fig_15_submission_monthly_forecast.png`
16. `fig_16_model_pipeline_overview.png`

## Nên chèn vào report 1–1.5 trang

Chỉ cần chọn khoảng 4–6 hình mạnh nhất:

- `fig_16_model_pipeline_overview.png`
- `fig_07_cv_folds_from_code.png`
- `fig_09_sample_weights.png`
- `fig_10_feature_importance_revenue.png` hoặc `fig_11_feature_importance_cogs.png`
- `fig_12_validation_actual_pred_revenue.png`
- `fig_15_submission_monthly_forecast.png`
