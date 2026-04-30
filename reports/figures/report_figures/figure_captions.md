# Suggested figure captions for Task 3 report

Use only the strongest 4-6 figures in the main 4-page report. Put the rest in appendix or GitHub.

1. **fig_01_target_overview.png** — Monthly historical Revenue/COGS and forecast horizon. Use this to introduce the forecasting task and show that the output follows the required 2023-01-01 to 2024-07-01 horizon.
2. **fig_02_revenue_year_month_heatmap.png** — Year-month heatmap of Revenue. Use this to discuss long-term trend, seasonal peaks, and unusual periods.
3. **fig_03_monthly_seasonality.png** — Average daily Revenue/COGS by month. Use this to justify monthly and yearly Fourier/calendar features.
4. **fig_04_weekday_pattern.png** — Average daily Revenue by weekday. Use this to justify weekly seasonality features.
5. **fig_05_event_impact.png** — Average Revenue on normal, promo, Tet, fixed holiday, weekend, and month-end days. Use this to explain why event/promo/holiday features are included.
6. **fig_06_time_series_cv_splits.png** — Time-series validation folds. Use this in the methodology section to show that validation respects chronological order and avoids leakage.
7. **fig_07_sample_weights.png** — Sample weighting scheme. Use this to explain why the model emphasizes the selected historical regime.
8. **fig_08_feature_importance_revenue.png** and **fig_08_feature_importance_cogs.png** — LightGBM feature importances. Use this for the required explainability section.
9. **fig_09_validation_actual_pred_revenue.png** and **fig_09_validation_actual_pred_cogs.png** — 2022 holdout actual vs predicted. Use this to report internal validation behavior before Kaggle test submission.
10. **fig_10_validation_residuals_revenue.png** and **fig_10_validation_residuals_cogs.png** — Residual diagnostics. Use this to discuss where the model over/under-predicts.
11. **fig_11_submission_monthly_forecast.png** — Aggregated Kaggle test forecast. Use this to describe the submitted forecast shape.
12. **fig_12_pipeline_overview.png** — Pipeline diagram. Use this as the main method figure.

Recommended main-report set:
- Pipeline overview
- Time-series CV splits
- Feature importance for Revenue
- Validation actual vs predicted for Revenue
- Event/calendar impact
- Submission monthly forecast
