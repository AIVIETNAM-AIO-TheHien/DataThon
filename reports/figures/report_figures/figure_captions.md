# Task 3 figure captions — model-aligned version

This folder intentionally contains exactly 16 PNG figures.

1. **fig_01_monthly_history_and_forecast.png** — Monthly historical Revenue/COGS plus the submitted forecast if `submission.csv` exists.
2. **fig_02_revenue_year_month_heatmap.png** — Revenue month-by-year heatmap from `sales.csv`.
3. **fig_03_cogs_year_month_heatmap.png** — COGS month-by-year heatmap from `sales.csv`.
4. **fig_04_average_daily_by_month.png** — Average daily Revenue/COGS by calendar month; supports yearly/monthly seasonality features.
5. **fig_05_average_revenue_by_weekday.png** — Average Revenue by day of week; supports weekly Fourier/weekday features.
6. **fig_06_calendar_event_impact.png** — Average Revenue/COGS on normal days, promo days, Tet windows, fixed holidays, weekends, and month-end days.
7. **fig_07_cv_folds_from_code.png** — CV folds drawn by calling `src.cv_validation.time_series_split()`, so it matches the actual code.
8. **fig_08_lgb_internal_split.png** — Internal split inside `train_lgb_with_weight()`: dates <= 2022-07-04 for fit and dates > 2022-07-04 for early stopping.
9. **fig_09_sample_weights.png** — Sample weights used in both tuning and pipeline: 2014-2018 have weight 1.0, all other years have weight 0.01.
10. **fig_10_feature_importance_revenue.png** — Base LightGBM feature importance for Revenue using `train_lgb_with_weight()` and tuned params if available.
11. **fig_11_feature_importance_cogs.png** — Base LightGBM feature importance for COGS using the same training function.
12. **fig_12_validation_actual_pred_revenue.png** — Fold A validation: actual Revenue vs final ensemble prediction.
13. **fig_13_validation_actual_pred_cogs.png** — Fold A validation: actual COGS vs final ensemble prediction.
14. **fig_14_validation_residuals_final_ensemble.png** — Fold A residual diagnostics for final ensemble predictions.
15. **fig_15_submission_monthly_forecast.png** — Monthly forecast from `submission.csv` over the Kaggle test horizon.
16. **fig_16_model_pipeline_overview.png** — Diagram of the exact model flow: tuning -> LGB base/Q-specialists -> Ridge -> 80/20 blend -> calibration -> submission.

Recommended figures for a 1–1.5 page report:
- fig_16_model_pipeline_overview.png
- fig_07_cv_folds_from_code.png
- fig_09_sample_weights.png
- fig_10_feature_importance_revenue.png or fig_11_feature_importance_cogs.png
- fig_12_validation_actual_pred_revenue.png
- fig_15_submission_monthly_forecast.png
