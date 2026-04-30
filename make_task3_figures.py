"""
make_task3_figures.py
Generate report-ready figures for Datathon 2026 Task 3 forecasting report.

How to use:
    1) Put this file in your repo root, next to run_pipeline.py and the src/ folder.
    2) Make sure data/raw/sales.csv exists.
    3) Run your pipeline first if you want forecast figures:
           python run_pipeline.py
    4) Generate figures:
           python make_task3_figures.py --raw-dir data/raw --submission submission.csv --out-dir report_figures

Outputs:
    report_figures/*.png
    report_figures/figure_captions.md
    report_figures/validation_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Allow running from repo root or from scripts/ subfolder.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.data_prep import build_features
except Exception as exc:
    raise ImportError(
        "Cannot import src.data_prep.build_features. Put this script in the repo root "
        "where the src/ folder exists, then run again."
    ) from exc

try:
    from src.cv_validation import time_series_split
except Exception:
    time_series_split = None


def ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")


def load_sales(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "sales.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Expected sales.csv with Date, Revenue, COGS.")
    df = pd.read_csv(path, parse_dates=["Date"])
    needed = {"Date", "Revenue", "COGS"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"sales.csv is missing columns: {missing}")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_submission(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        print("[WARN] submission.csv not found. Forecast-specific figures will be skipped.")
        return None
    sub = pd.read_csv(path, parse_dates=["Date"])
    needed = {"Date", "Revenue", "COGS"}
    missing = needed - set(sub.columns)
    if missing:
        print(f"[WARN] submission file is missing {missing}; forecast figures skipped.")
        return None
    return sub.sort_values("Date").reset_index(drop=True)


def add_engineered_features(sales: pd.DataFrame) -> pd.DataFrame:
    feat = build_features(sales["Date"])
    feat["Revenue"] = sales["Revenue"].values
    feat["COGS"] = sales["COGS"].values
    feat["GrossProfit"] = feat["Revenue"] - feat["COGS"]
    feat["Margin"] = np.where(feat["Revenue"] != 0, feat["GrossProfit"] / feat["Revenue"], np.nan)
    promo_cols = [c for c in feat.columns if c.startswith("promo_") and not c.endswith(("_since", "_until", "_disc"))]
    hol_cols = [c for c in feat.columns if c.startswith("hol_")]
    feat["has_promo"] = feat[promo_cols].max(axis=1) if promo_cols else 0
    feat["has_fixed_holiday"] = feat[hol_cols].max(axis=1) if hol_cols else 0
    return feat


def fig_01_target_overview(sales: pd.DataFrame, sub: Optional[pd.DataFrame], out_dir: Path) -> None:
    monthly = sales.set_index("Date")[["Revenue", "COGS"]].resample("MS").sum()
    plt.figure(figsize=(11, 5.5))
    plt.plot(monthly.index, monthly["Revenue"], label="Train Revenue, monthly total")
    plt.plot(monthly.index, monthly["COGS"], label="Train COGS, monthly total")
    if sub is not None:
        pred_monthly = sub.set_index("Date")[["Revenue", "COGS"]].resample("MS").sum()
        plt.plot(pred_monthly.index, pred_monthly["Revenue"], linestyle="--", label="Forecast Revenue, monthly total")
        plt.plot(pred_monthly.index, pred_monthly["COGS"], linestyle="--", label="Forecast COGS, monthly total")
        plt.axvline(pd.Timestamp("2023-01-01"), linestyle=":", linewidth=1.5, label="Forecast start")
    plt.title("Historical and Forecast Monthly Revenue / COGS")
    plt.xlabel("Date")
    plt.ylabel("Monthly total")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_fig(out_dir / "fig_01_target_overview.png")


def fig_02_year_month_heatmap(sales: pd.DataFrame, out_dir: Path) -> None:
    tmp = sales.copy()
    tmp["year"] = tmp["Date"].dt.year
    tmp["month"] = tmp["Date"].dt.month
    pivot = tmp.pivot_table(index="year", columns="month", values="Revenue", aggfunc="sum")
    plt.figure(figsize=(10, 5.5))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Monthly Revenue")
    plt.title("Revenue Seasonality Heatmap by Year and Month")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.xticks(range(12), [str(i) for i in range(1, 13)])
    plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
    save_fig(out_dir / "fig_02_revenue_year_month_heatmap.png")


def fig_03_calendar_patterns(feat: pd.DataFrame, out_dir: Path) -> None:
    month_avg = feat.groupby("month", as_index=False)[["Revenue", "COGS"]].mean()
    plt.figure(figsize=(9, 5))
    plt.plot(month_avg["month"], month_avg["Revenue"], marker="o", label="Avg Revenue")
    plt.plot(month_avg["month"], month_avg["COGS"], marker="o", label="Avg COGS")
    plt.title("Average Daily Revenue / COGS by Month")
    plt.xlabel("Month")
    plt.ylabel("Average daily value")
    plt.xticks(range(1, 13))
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_fig(out_dir / "fig_03_monthly_seasonality.png")

    dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_avg = feat.groupby("dow", as_index=False)["Revenue"].mean()
    plt.figure(figsize=(8, 5))
    plt.bar(dow_avg["dow"], dow_avg["Revenue"])
    plt.title("Average Daily Revenue by Day of Week")
    plt.xlabel("Day of week")
    plt.ylabel("Average daily Revenue")
    plt.xticks(range(7), dow_order)
    plt.grid(True, axis="y", alpha=0.25)
    save_fig(out_dir / "fig_04_weekday_pattern.png")


def fig_04_event_impact(feat: pd.DataFrame, out_dir: Path) -> None:
    groups = []
    labels = []
    normal_mask = (feat["has_promo"] == 0) & (feat["tet_in_14"] == 0) & (feat["has_fixed_holiday"] == 0)
    definitions = [
        ("Normal days", normal_mask),
        ("Promo days", feat["has_promo"] == 1),
        ("Tet ±14 days", feat["tet_in_14"] == 1),
        ("Fixed holidays", feat["has_fixed_holiday"] == 1),
        ("Weekend", feat["is_weekend"] == 1),
        ("Month-end 3d", feat["is_last3"] == 1),
    ]
    for name, mask in definitions:
        s = feat.loc[mask, "Revenue"]
        if len(s) > 0:
            labels.append(name)
            groups.append(s.mean())
    plt.figure(figsize=(9.5, 5))
    plt.bar(labels, groups)
    plt.title("Average Revenue on Calendar / Promotion Events")
    plt.xlabel("Event type")
    plt.ylabel("Average daily Revenue")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    save_fig(out_dir / "fig_05_event_impact.png")


def fig_05_cv_splits(sales: pd.DataFrame, out_dir: Path) -> None:
    dates = sales["Date"]
    folds = ["A", "B", "C"]
    rows = []
    for fold in folds:
        if time_series_split is not None:
            train_mask, val_mask = time_series_split(dates, fold)
        else:
            if fold == "A":
                train_mask = dates < pd.Timestamp("2022-01-01")
                val_mask = (dates >= pd.Timestamp("2022-01-01")) & (dates <= pd.Timestamp("2022-12-31"))
            elif fold == "B":
                train_mask = dates < pd.Timestamp("2021-01-01")
                val_mask = (dates >= pd.Timestamp("2021-01-01")) & (dates <= pd.Timestamp("2021-12-31"))
            else:
                train_mask = (dates >= pd.Timestamp("2020-07-01")) & (dates < pd.Timestamp("2021-07-01"))
                val_mask = (dates >= pd.Timestamp("2021-07-01")) & (dates <= pd.Timestamp("2022-06-30"))
        rows.append((fold, "train", dates[train_mask].min(), dates[train_mask].max()))
        rows.append((fold, "validation", dates[val_mask].min(), dates[val_mask].max()))

    fig, ax = plt.subplots(figsize=(10, 3.6))
    y_positions = {"A": 2, "B": 1, "C": 0}
    style = {"train": 7, "validation": 13}
    for fold, typ, start, end in rows:
        if pd.isna(start) or pd.isna(end):
            continue
        y = y_positions[fold]
        ax.plot([start, end], [y, y], linewidth=style[typ], solid_capstyle="butt", label=typ if fold == "A" else None)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Fold C", "Fold B", "Fold A"])
    ax.set_title("Time-Series Cross-Validation Splits")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.25)
    save_fig(out_dir / "fig_06_time_series_cv_splits.png")


def fig_06_sample_weights(feat: pd.DataFrame, out_dir: Path) -> None:
    tmp = feat[["Date", "year"]].copy()
    tmp["weight"] = 0.01
    tmp.loc[(tmp["year"] >= 2014) & (tmp["year"] <= 2018), "weight"] = 1.0
    monthly = tmp.set_index("Date")["weight"].resample("MS").mean()
    plt.figure(figsize=(10, 4))
    plt.plot(monthly.index, monthly.values)
    plt.title("Training Sample Weights Used by the Model")
    plt.xlabel("Date")
    plt.ylabel("Sample weight")
    plt.grid(True, alpha=0.25)
    save_fig(out_dir / "fig_07_sample_weights.png")


def _try_train_lightgbm_importance(feat: pd.DataFrame, target: str, best_params_path: Optional[Path]) -> Optional[pd.DataFrame]:
    try:
        import lightgbm as lgb
    except Exception:
        print("[WARN] lightgbm is not installed. Skipping feature importance.")
        return None

    non_features = {"Date", "Revenue", "COGS", "GrossProfit", "Margin"}
    feature_cols = [c for c in feat.columns if c not in non_features]
    X = feat[feature_cols].values.astype(float)
    y = np.log1p(feat[target].values)
    years = feat["year"].values
    w = np.full(len(years), 0.01)
    w[(years >= 2014) & (years <= 2018)] = 1.0

    params = dict(
        objective="regression",
        metric="mae",
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=30,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        lambda_l2=1.0,
        seed=42,
        verbosity=-1,
    )
    if best_params_path is not None and best_params_path.exists():
        try:
            with open(best_params_path, "r", encoding="utf-8") as f:
                params.update(json.load(f))
        except Exception as exc:
            print(f"[WARN] could not read {best_params_path}: {exc}")

    train_data = lgb.Dataset(X, label=y, weight=w, feature_name=feature_cols)
    model = lgb.train(params, train_data, num_boost_round=700)
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.feature_importance(importance_type="gain"),
        "importance_split": model.feature_importance(importance_type="split"),
        "target": target,
    })
    imp = imp.sort_values("importance_gain", ascending=False)
    return imp


def fig_07_feature_importance(feat: pd.DataFrame, out_dir: Path, best_params_path: Optional[Path]) -> None:
    all_imp = []
    for target in ["Revenue", "COGS"]:
        imp = _try_train_lightgbm_importance(feat, target, best_params_path)
        if imp is not None:
            all_imp.append(imp)
            top = imp.head(20).iloc[::-1]
            plt.figure(figsize=(9, 6))
            plt.barh(top["feature"], top["importance_gain"])
            plt.title(f"Top 20 LightGBM Feature Importances for {target}")
            plt.xlabel("Importance by gain")
            plt.ylabel("Feature")
            plt.grid(True, axis="x", alpha=0.25)
            save_fig(out_dir / f"fig_08_feature_importance_{target.lower()}.png")
    if all_imp:
        pd.concat(all_imp, ignore_index=True).to_csv(out_dir / "feature_importance.csv", index=False)
        print(f"[OK] saved {out_dir / 'feature_importance.csv'}")


def _train_validation_model(feat: pd.DataFrame, target: str):
    try:
        import lightgbm as lgb
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except Exception:
        print("[WARN] lightgbm or sklearn not installed. Skipping validation residual figures.")
        return None

    non_features = {"Date", "Revenue", "COGS", "GrossProfit", "Margin"}
    feature_cols = [c for c in feat.columns if c not in non_features]
    train_mask = feat["Date"] < pd.Timestamp("2022-01-01")
    val_mask = (feat["Date"] >= pd.Timestamp("2022-01-01")) & (feat["Date"] <= pd.Timestamp("2022-12-31"))
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        return None

    X_train = feat.loc[train_mask, feature_cols].values.astype(float)
    X_val = feat.loc[val_mask, feature_cols].values.astype(float)
    y_train = np.log1p(feat.loc[train_mask, target].values)
    y_val_real = feat.loc[val_mask, target].values
    years = feat.loc[train_mask, "year"].values
    w = np.full(len(years), 0.01)
    w[(years >= 2014) & (years <= 2018)] = 1.0

    params = dict(
        objective="regression",
        metric="mae",
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=30,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        lambda_l2=1.0,
        seed=42,
        verbosity=-1,
    )
    train_data = lgb.Dataset(X_train, label=y_train, weight=w, feature_name=feature_cols)
    model = lgb.train(params, train_data, num_boost_round=700)
    pred = np.expm1(model.predict(X_val))
    val_dates = feat.loc[val_mask, "Date"].reset_index(drop=True)
    actual = pd.Series(y_val_real).reset_index(drop=True)
    pred = pd.Series(pred).reset_index(drop=True)
    metrics = {
        "target": target,
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "R2": r2_score(actual, pred),
    }
    return val_dates, actual, pred, metrics


def fig_08_validation_residuals(feat: pd.DataFrame, out_dir: Path) -> None:
    metrics_rows = []
    for target in ["Revenue", "COGS"]:
        result = _train_validation_model(feat, target)
        if result is None:
            continue
        val_dates, actual, pred, metrics = result
        metrics_rows.append(metrics)
        residual = pred - actual

        plt.figure(figsize=(10, 5))
        plt.plot(val_dates, actual, label=f"Actual {target}")
        plt.plot(val_dates, pred, label=f"Predicted {target}")
        plt.title(f"Validation Actual vs Predicted for {target} (2022 Holdout)")
        plt.xlabel("Date")
        plt.ylabel(target)
        plt.legend()
        plt.grid(True, alpha=0.25)
        save_fig(out_dir / f"fig_09_validation_actual_pred_{target.lower()}.png")

        plt.figure(figsize=(8, 5))
        plt.scatter(actual, residual, s=12, alpha=0.65)
        plt.axhline(0, linestyle=":", linewidth=1.5)
        plt.title(f"Validation Residuals for {target} (2022 Holdout)")
        plt.xlabel(f"Actual {target}")
        plt.ylabel("Prediction - Actual")
        plt.grid(True, alpha=0.25)
        save_fig(out_dir / f"fig_10_validation_residuals_{target.lower()}.png")

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(out_dir / "validation_metrics.csv", index=False)
        print(f"[OK] saved {out_dir / 'validation_metrics.csv'}")


def fig_09_submission_forecast(sub: Optional[pd.DataFrame], out_dir: Path) -> None:
    if sub is None:
        return
    monthly = sub.set_index("Date")[["Revenue", "COGS"]].resample("MS").sum()
    plt.figure(figsize=(10, 5))
    plt.plot(monthly.index, monthly["Revenue"], marker="o", label="Forecast Revenue")
    plt.plot(monthly.index, monthly["COGS"], marker="o", label="Forecast COGS")
    plt.title("Monthly Forecast on Kaggle Test Horizon")
    plt.xlabel("Date")
    plt.ylabel("Monthly forecast total")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_fig(out_dir / "fig_11_submission_monthly_forecast.png")


def fig_10_pipeline_diagram(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    boxes = [
        (0.05, 0.72, 0.18, 0.16, "Input\nsales.csv\nDate, Revenue, COGS"),
        (0.30, 0.72, 0.20, 0.16, "Calendar features\ntrend, Fourier, weekday,\nTet, holidays, promo windows"),
        (0.58, 0.72, 0.17, 0.16, "Targets\nlog1p(Revenue)\nlog1p(COGS)"),
        (0.05, 0.40, 0.18, 0.16, "Sample weights\n2014-2018: 1.0\nothers: 0.01"),
        (0.30, 0.40, 0.20, 0.16, "LightGBM layer\nbase model + Q1-Q4\nspecialists"),
        (0.58, 0.40, 0.17, 0.16, "Ridge layer\nstandardized linear\ncalendar fallback"),
        (0.82, 0.40, 0.14, 0.16, "Blend\n80% LGB\n20% Ridge"),
        (0.58, 0.12, 0.17, 0.16, "Calibration\nRevenue x 1.26\nCOGS x 1.32"),
        (0.82, 0.12, 0.14, 0.16, "Output\nsubmission.csv"),
    ]
    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.23, 0.80), (0.30, 0.80)),
        ((0.50, 0.80), (0.58, 0.80)),
        ((0.14, 0.72), (0.14, 0.56)),
        ((0.23, 0.48), (0.30, 0.48)),
        ((0.50, 0.48), (0.58, 0.48)),
        ((0.75, 0.48), (0.82, 0.48)),
        ((0.89, 0.40), (0.70, 0.28)),
        ((0.75, 0.20), (0.82, 0.20)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.set_title("Task 3 Forecasting Pipeline Overview", fontsize=14, pad=12)
    save_fig(out_dir / "fig_12_pipeline_overview.png")


def write_captions(out_dir: Path) -> None:
    captions = """# Suggested figure captions for Task 3 report

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
"""
    path = out_dir / "figure_captions.md"
    path.write_text(captions, encoding="utf-8")
    print(f"[OK] saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw", help="Folder containing sales.csv")
    parser.add_argument("--submission", default="submission.csv", help="submission.csv path; optional")
    parser.add_argument("--out-dir", default="report_figures", help="Output folder")
    parser.add_argument("--best-params", default="best_lgb_params.json", help="Optional best LightGBM params JSON")
    parser.add_argument("--skip-model-figures", action="store_true", help="Skip LightGBM feature importance and validation residual figures")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    ensure_out(out_dir)

    sales = load_sales(raw_dir)
    sub = load_submission(Path(args.submission))
    feat = add_engineered_features(sales)

    fig_01_target_overview(sales, sub, out_dir)
    fig_02_year_month_heatmap(sales, out_dir)
    fig_03_calendar_patterns(feat, out_dir)
    fig_04_event_impact(feat, out_dir)
    fig_05_cv_splits(sales, out_dir)
    fig_06_sample_weights(feat, out_dir)
    if not args.skip_model_figures:
        fig_07_feature_importance(feat, out_dir, Path(args.best_params))
        fig_08_validation_residuals(feat, out_dir)
    fig_09_submission_forecast(sub, out_dir)
    fig_10_pipeline_diagram(out_dir)
    write_captions(out_dir)

    print("\nDone. Figures are in:", out_dir.resolve())


if __name__ == "__main__":
    main()
