"""
make_task3_figures.py
Model-aligned figure generator for Datathon 2026 Task 3.

This script is aligned with the current model flow:
    1) python run_tuning.py      -> writes best_lgb_params.json
    2) python run_pipeline.py    -> writes submission.csv
    3) python make_task3_figures.py --raw-dir data/raw --submission submission.csv

Default output:
    reports/                 # exactly 16 PNG figures
    reports/figure_captions.md
    reports/validation_metrics.csv
    reports/figure_manifest.csv
    reports/feature_importance.csv

Important alignment choices:
    - Uses src.data_prep.build_features() exactly like run_tuning.py and run_pipeline.py.
    - Uses the same NON_FEATURES = ['Date', 'Revenue', 'COGS'] feature selection.
    - Uses the same sample weights: 2014-2018 => 1.0, all other years => 0.01.
    - Uses src.cv_validation.time_series_split() for CV fold visualization.
    - Uses src.train_model.train_lgb_with_weight(), train_q_specialist(), train_ridge(),
      and predict_ridge() for model-based validation/importance figures.
    - Uses the same final blend:
          LGB blend = 0.60 * Q-specialist + 0.40 * LGB base
          raw       = 0.20 * Ridge + 0.80 * LGB blend
          final     = Revenue * 1.26, COGS * 1.32

Run:
    python make_task3_figures.py --raw-dir data/raw --submission submission.csv --out-dir reports

Optional quick mode, keeps 16 output files but creates notice placeholders for expensive
model-based figures:
    python make_task3_figures.py --raw-dir data/raw --submission submission.csv --out-dir reports --skip-model-figures
"""

from __future__ import annotations

import matplotlib.dates as mdates
import argparse
import json
import shutil
import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.data_prep import build_features
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Cannot import src.data_prep.build_features. Put this file in the repo root, "
        "next to run_pipeline.py and the src/ folder."
    ) from exc

try:
    from src.cv_validation import time_series_split
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Cannot import src.cv_validation.time_series_split. This figure file is intended "
        "to match your current Task 3 repo structure."
    ) from exc

try:
    from src.train_model import (
        LGB_PARAMS,
        train_lgb_with_weight,
        train_q_specialist,
        train_ridge,
        predict_ridge,
    )
    HAS_MODEL_IMPORTS = True
except Exception:
    LGB_PARAMS = None
    train_lgb_with_weight = None
    train_q_specialist = None
    train_ridge = None
    predict_ridge = None
    HAS_MODEL_IMPORTS = False


# ==============================
# Constants copied from pipeline
# ==============================
NON_FEATURES = ["Date", "Revenue", "COGS"]
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2024-07-01")
INTERNAL_LGB_SPLIT = pd.Timestamp("2022-07-04")
WEIGHT_START_YEAR = 2014
WEIGHT_END_YEAR = 2018
WEIGHT_MAIN = 1.0
WEIGHT_OTHER = 0.01
Q_SPECIALIST_ALPHA = 0.60
RIDGE_WEIGHT = 0.20
LGB_BLEND_WEIGHT = 0.80
CALIBRATION = {"Revenue": 1.26, "COGS": 1.32}
TARGETS = ["Revenue", "COGS"]
FOLD_NAMES = ["A", "B", "C"]

EXPECTED_FIGURES = [
    "fig_01_monthly_history_and_forecast.png",
    "fig_02_revenue_year_month_heatmap.png",
    "fig_03_cogs_year_month_heatmap.png",
    "fig_04_average_daily_by_month.png",
    "fig_05_average_revenue_by_weekday.png",
    "fig_06_calendar_event_impact.png",
    "fig_07_cv_folds_from_code.png",
    "fig_08_lgb_internal_split.png",
    "fig_09_sample_weights.png",
    "fig_10_feature_importance_revenue.png",
    "fig_11_feature_importance_cogs.png",
    "fig_12_validation_actual_pred_revenue.png",
    "fig_13_validation_actual_pred_cogs.png",
    "fig_14_validation_residuals_final_ensemble.png",
    "fig_15_submission_monthly_forecast.png",
    "fig_16_model_pipeline_overview.png",
]


# ==============================
# Utilities
# ==============================
def ensure_clean_out(out_dir: Path, clean: bool = False) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path}")


def save_notice_figure(path: Path, title: str, message: str) -> None:
    plt.figure(figsize=(10, 5.5))
    plt.axis("off")
    plt.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, fontweight="bold")
    plt.text(0.5, 0.42, message, ha="center", va="center", fontsize=11, wrap=True)
    save_fig(path)


def load_sales(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "sales.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Expected data/raw/sales.csv with Date, Revenue, COGS.")
    df = pd.read_csv(path, parse_dates=["Date"])
    needed = {"Date", "Revenue", "COGS"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"sales.csv is missing columns: {sorted(missing)}")
    return df.sort_values("Date").reset_index(drop=True)


def load_submission(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        print("[WARN] submission.csv not found. Forecast figures will use notice placeholders where needed.")
        return None
    sub = pd.read_csv(path, parse_dates=["Date"])
    needed = {"Date", "Revenue", "COGS"}
    missing = needed - set(sub.columns)
    if missing:
        print(f"[WARN] submission file is missing columns {sorted(missing)}. Forecast figures will use placeholders.")
        return None
    return sub.sort_values("Date").reset_index(drop=True)


def load_best_params(path: Path) -> Optional[dict]:
    if not path.exists():
        print(f"[WARN] {path} not found. Figures will use default LGB params from train_model.py.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    print(f"[OK] loaded tuned LightGBM params from {path}")
    return params


def add_engineered_features(sales: pd.DataFrame) -> pd.DataFrame:
    # Exactly the same feature entry point as run_tuning.py and run_pipeline.py.
    feat = build_features(sales["Date"])
    feat["Revenue"] = sales["Revenue"].values
    feat["COGS"] = sales["COGS"].values

    # Extra columns below are only for EDA figures and are deliberately excluded
    # from model feature lists by get_model_matrices().
    feat["GrossProfit"] = feat["Revenue"] - feat["COGS"]
    feat["Margin"] = np.where(feat["Revenue"] != 0, feat["GrossProfit"] / feat["Revenue"], np.nan)

    promo_cols = [
        c for c in feat.columns
        if c.startswith("promo_") and not c.endswith(("_since", "_until", "_disc"))
    ]
    hol_cols = [c for c in feat.columns if c.startswith("hol_")]
    feat["has_promo"] = feat[promo_cols].max(axis=1) if promo_cols else 0
    feat["has_fixed_holiday"] = feat[hol_cols].max(axis=1) if hol_cols else 0
    return feat


def get_model_matrices(feat: pd.DataFrame) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray], pd.Series, np.ndarray, np.ndarray]:
    # Match run_pipeline.py / run_tuning.py exactly:
    # cols = [c for c in feat_train.columns if c not in ['Date', 'Revenue', 'COGS']]
    feature_cols = [c for c in feat.columns if c not in NON_FEATURES and c not in {"GrossProfit", "Margin", "has_promo", "has_fixed_holiday"}]
    X = feat[feature_cols].values.astype(float)
    y = {target: np.log1p(feat[target].values) for target in TARGETS}
    dates = feat["Date"]
    years = feat["year"].values
    quarters = feat["quarter"].values
    return feature_cols, X, y, dates, years, quarters


def make_weights(years: np.ndarray) -> np.ndarray:
    w = np.full(len(years), WEIGHT_OTHER, dtype=float)
    w[(years >= WEIGHT_START_YEAR) & (years <= WEIGHT_END_YEAR)] = WEIGHT_MAIN
    return w


def check_submission_horizon(sub: Optional[pd.DataFrame]) -> str:
    if sub is None or sub.empty:
        return "submission.csv not available"
    d0 = sub["Date"].min()
    d1 = sub["Date"].max()
    n = len(sub)
    expected_n = len(pd.date_range(TEST_START, TEST_END, freq="D"))
    ok = (d0 == TEST_START) and (d1 == TEST_END) and (n == expected_n)
    return f"{d0.date()} to {d1.date()}, {n} rows" + (" (OK)" if ok else f" (expected {expected_n} rows)")


# ==============================
# EDA figures
# ==============================
def fig_01_monthly_history_and_forecast(sales: pd.DataFrame, sub: Optional[pd.DataFrame], out_dir: Path) -> None:
    monthly = sales.set_index("Date")[["Revenue", "COGS"]].resample("MS").sum()
    plt.figure(figsize=(11.5, 5.5))
    plt.plot(monthly.index, monthly["Revenue"], label="Train Revenue")
    plt.plot(monthly.index, monthly["COGS"], label="Train COGS")
    if sub is not None:
        pred_monthly = sub.set_index("Date")[["Revenue", "COGS"]].resample("MS").sum()
        plt.plot(pred_monthly.index, pred_monthly["Revenue"], linestyle="--", label="Forecast Revenue")
        plt.plot(pred_monthly.index, pred_monthly["COGS"], linestyle="--", label="Forecast COGS")
        plt.axvline(TEST_START, linestyle=":", linewidth=1.5, label="Forecast start")
    plt.title("Historical and Forecast Monthly Revenue / COGS")
    plt.xlabel("Date")
    plt.ylabel("Monthly total")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_fig(out_dir / EXPECTED_FIGURES[0])


def _year_month_heatmap(sales: pd.DataFrame, target: str, out_path: Path) -> None:
    tmp = sales.copy()
    tmp["year"] = tmp["Date"].dt.year
    tmp["month"] = tmp["Date"].dt.month
    pivot = tmp.pivot_table(index="year", columns="month", values=target, aggfunc="sum").sort_index()
    plt.figure(figsize=(10.5, 5.5))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label=f"Monthly {target}")
    plt.title(f"{target} Seasonality Heatmap by Year and Month")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.xticks(range(12), [str(i) for i in range(1, 13)])
    plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
    save_fig(out_path)


def fig_02_03_heatmaps(sales: pd.DataFrame, out_dir: Path) -> None:
    _year_month_heatmap(sales, "Revenue", out_dir / EXPECTED_FIGURES[1])
    _year_month_heatmap(sales, "COGS", out_dir / EXPECTED_FIGURES[2])


def fig_04_average_daily_by_month(feat: pd.DataFrame, out_dir: Path) -> None:
    month_avg = feat.groupby("month", as_index=False)[["Revenue", "COGS"]].mean()
    plt.figure(figsize=(9.5, 5.2))
    plt.plot(month_avg["month"], month_avg["Revenue"], marker="o", label="Avg Revenue")
    plt.plot(month_avg["month"], month_avg["COGS"], marker="o", label="Avg COGS")
    plt.title("Average Daily Revenue / COGS by Month")
    plt.xlabel("Month")
    plt.ylabel("Average daily value")
    plt.xticks(range(1, 13))
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_fig(out_dir / EXPECTED_FIGURES[3])


def fig_05_average_revenue_by_weekday(feat: pd.DataFrame, out_dir: Path) -> None:
    dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_avg = feat.groupby("dow", as_index=False)["Revenue"].mean()
    plt.figure(figsize=(8.5, 5.2))
    plt.bar(dow_avg["dow"], dow_avg["Revenue"])
    plt.title("Average Daily Revenue by Day of Week")
    plt.xlabel("Day of week")
    plt.ylabel("Average daily Revenue")
    plt.xticks(range(7), dow_order)
    plt.grid(True, axis="y", alpha=0.25)
    save_fig(out_dir / EXPECTED_FIGURES[4])


def fig_06_calendar_event_impact(feat: pd.DataFrame, out_dir: Path) -> None:
    normal_mask = (feat["has_promo"] == 0) & (feat["tet_in_14"] == 0) & (feat["has_fixed_holiday"] == 0)
    definitions = [
        ("Normal days", normal_mask),
        ("Promo days", feat["has_promo"] == 1),
        ("Tet ±14 days", feat["tet_in_14"] == 1),
        ("Fixed holidays", feat["has_fixed_holiday"] == 1),
        ("Weekend", feat["is_weekend"] == 1),
        ("Month-end 3d", feat["is_last3"] == 1),
    ]
    rows = []
    for name, mask in definitions:
        sub = feat.loc[mask]
        if len(sub) > 0:
            rows.append((name, sub["Revenue"].mean(), sub["COGS"].mean(), len(sub)))
    impact = pd.DataFrame(rows, columns=["event", "Revenue", "COGS", "n_days"])

    x = np.arange(len(impact))
    width = 0.38
    plt.figure(figsize=(10.8, 5.3))
    plt.bar(x - width / 2, impact["Revenue"], width=width, label="Revenue")
    plt.bar(x + width / 2, impact["COGS"], width=width, label="COGS")
    plt.title("Average Revenue / COGS on Calendar and Promotion Events")
    plt.xlabel("Event type")
    plt.ylabel("Average daily value")
    plt.xticks(x, impact["event"], rotation=20, ha="right")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    save_fig(out_dir / EXPECTED_FIGURES[5])


# ==============================
# Split / weight figures
# ==============================
import matplotlib.dates as mdates

def fig_07_cv_folds_from_code(feat: pd.DataFrame, out_dir: Path) -> None:
    """
    Vẽ đúng CV folds bằng src.cv_validation.time_series_split(),
    format lại để:
    - không bị đè chữ lên tick năm
    - title và legend không dính nhau
    """
    dates = pd.to_datetime(feat["Date"]).reset_index(drop=True)

    rows = []
    for fold in FOLD_NAMES:
        train_mask, val_mask = time_series_split(dates, fold)

        if train_mask.sum() > 0:
            rows.append({
                "fold": fold,
                "kind": "train",
                "start": dates[train_mask].min(),
                "end": dates[train_mask].max(),
                "n": int(train_mask.sum())
            })

        if val_mask.sum() > 0:
            rows.append({
                "fold": fold,
                "kind": "validation",
                "start": dates[val_mask].min(),
                "end": dates[val_mask].max(),
                "n": int(val_mask.sum())
            })

    fig, ax = plt.subplots(figsize=(12.5, 4.8))

    # A trên, B giữa, C dưới
    y_pos = {
        "A": 2,
        "B": 1,
        "C": 0
    }

    colors = {
        "train": "#1f77b4",
        "validation": "#ff7f0e"
    }

    bar_height = 0.22
    added_labels = set()

    for row in rows:
        fold = row["fold"]
        kind = row["kind"]
        start = row["start"]
        end = row["end"]
        n = row["n"]

        y = y_pos[fold]
        start_num = mdates.date2num(start)
        end_num = mdates.date2num(end + pd.Timedelta(days=1))
        width = end_num - start_num

        label = kind if kind not in added_labels else None

        ax.barh(
            y=y,
            width=width,
            left=start_num,
            height=bar_height,
            color=colors[kind],
            edgecolor="white",
            linewidth=1.1,
            label=label,
            zorder=3
        )

        added_labels.add(kind)

        # Bar dài: chữ nằm giữa
        if width >= 650:
            ax.text(
                start_num + width / 2,
                y,
                f"{kind}: {n}d",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
                zorder=4
            )

        # Bar vừa: chữ nằm giữa, 2 dòng
        elif width >= 300:
            ax.text(
                start_num + width / 2,
                y,
                f"{kind}\n{n}d",
                ha="center",
                va="center",
                fontsize=8.5,
                color="black" if kind == "validation" else "white",
                fontweight="bold",
                zorder=4
            )

        # Bar ngắn: đẩy text ra ngoài
        else:
            if kind == "train":
                x_offset = 75
                y_offset = 0.20
            else:
                x_offset = 75
                y_offset = -0.20

            ax.text(
                end_num + x_offset,
                y + y_offset,
                f"{kind}: {n}d",
                ha="left",
                va="center",
                fontsize=8.8,
                color="#222222",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.9,
                    pad=1.6
                ),
                clip_on=False,
                zorder=5
            )

    # Chừa khoảng hai bên để text không bị cắt
    min_date = dates.min() - pd.Timedelta(days=180)
    max_date = dates.max() + pd.Timedelta(days=480)

    ax.set_xlim(
        mdates.date2num(min_date),
        mdates.date2num(max_date)
    )

    ax.set_ylim(-0.55, 2.55)

    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["Fold A", "Fold B", "Fold C"], fontsize=11)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.set_title(
        "Time-Series CV Folds Used by cv_validation.py",
        fontsize=15,
        pad=20
    )
    ax.set_xlabel("Date", fontsize=11)

    ax.grid(True, axis="x", linestyle="--", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Hạ legend xuống 1 chút để không dính title
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        frameon=False,
        fontsize=10
    )

    save_fig(out_dir / EXPECTED_FIGURES[6])

def fig_08_lgb_internal_split(feat: pd.DataFrame, out_dir: Path) -> None:
    dates = feat["Date"]
    fit_mask = dates <= INTERNAL_LGB_SPLIT
    ins_mask = dates > INTERNAL_LGB_SPLIT

    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    if fit_mask.sum() > 0:
        ax.plot([dates[fit_mask].min(), dates[fit_mask].max()], [1, 1], linewidth=13, solid_capstyle="butt", label="fit_idx: dates <= 2022-07-04")
    if ins_mask.sum() > 0:
        ax.plot([dates[ins_mask].min(), dates[ins_mask].max()], [1, 1], linewidth=13, solid_capstyle="butt", label="ins_idx: dates > 2022-07-04")
    ax.axvline(INTERNAL_LGB_SPLIT, linestyle=":", linewidth=1.5, label="internal split")
    ax.set_yticks([])
    ax.set_ylim(0.7, 1.3)
    ax.set_title("Internal LightGBM Early-Stopping Split in train_lgb_with_weight()")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left")
    ax.grid(True, axis="x", alpha=0.25)
    save_fig(out_dir / EXPECTED_FIGURES[7])


def fig_09_sample_weights(feat: pd.DataFrame, out_dir: Path) -> None:
    tmp = feat[["Date", "year"]].copy()
    tmp["weight"] = make_weights(tmp["year"].values)
    monthly = tmp.set_index("Date")[["weight"]].resample("MS").mean()

    plt.figure(figsize=(10.5, 4.2))
    plt.plot(monthly.index, monthly["weight"])
    plt.axhspan(WEIGHT_MAIN - 0.02, WEIGHT_MAIN + 0.02, alpha=0.08, label="2014-2018 high-weight regime")
    plt.title("Training Sample Weights Used by run_tuning.py and run_pipeline.py")
    plt.xlabel("Date")
    plt.ylabel("Sample weight")
    plt.ylim(-0.05, 1.1)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.25)
    save_fig(out_dir / EXPECTED_FIGURES[8])


# ==============================
# Model-aligned training helpers
# ==============================
def _require_model_imports() -> None:
    if not HAS_MODEL_IMPORTS:
        raise RuntimeError("Could not import required functions from src.train_model.")


def fit_base_lgb_for_importance(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    dates: pd.Series,
    feature_cols: Sequence[str],
    best_params: Optional[dict],
) -> pd.DataFrame:
    _require_model_imports()
    model = train_lgb_with_weight(X, y, weights, dates, custom_params=best_params)
    importance = pd.DataFrame({
        "feature": list(feature_cols),
        "importance_gain": model.feature_importance(importance_type="gain"),
        "importance_split": model.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    return importance


def predict_pipeline_like_for_target(
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    dates_train: pd.Series,
    years_train: np.ndarray,
    quarters_train: np.ndarray,
    X_pred: np.ndarray,
    quarters_pred: np.ndarray,
    target: str,
    best_params: Optional[dict],
) -> Dict[str, np.ndarray]:
    _require_model_imports()
    w_train = make_weights(years_train)

    base = train_lgb_with_weight(X_train, y_train_log, w_train, dates_train, custom_params=best_params)
    p_lgb_base = np.expm1(base.predict(X_pred))

    spec_pred_by_q = {}
    for q in [1, 2, 3, 4]:
        model_q = train_q_specialist(
            X_train,
            y_train_log,
            w_train,
            quarters_train,
            dates_train,
            q,
            custom_params=best_params,
        )
        spec_pred_by_q[q] = np.expm1(model_q.predict(X_pred))

    p_q_specialist = np.zeros(len(X_pred), dtype=float)
    for q in [1, 2, 3, 4]:
        mask = quarters_pred == q
        p_q_specialist[mask] = spec_pred_by_q[q][mask]

    lgb_blend = Q_SPECIALIST_ALPHA * p_q_specialist + (1 - Q_SPECIALIST_ALPHA) * p_lgb_base

    ridge, stats = train_ridge(X_train, y_train_log)
    p_ridge = np.expm1(predict_ridge(ridge, X_pred, stats))

    raw = RIDGE_WEIGHT * p_ridge + LGB_BLEND_WEIGHT * lgb_blend
    final = CALIBRATION[target] * raw

    return {
        "lgb_base": p_lgb_base,
        "q_specialist": p_q_specialist,
        "lgb_blend": lgb_blend,
        "ridge": p_ridge,
        "raw_before_calibration": raw,
        "final": final,
    }


def run_model_aligned_validation(
    feat: pd.DataFrame,
    feature_cols: Sequence[str],
    X: np.ndarray,
    y: Dict[str, np.ndarray],
    dates: pd.Series,
    years: np.ndarray,
    quarters: np.ndarray,
    best_params: Optional[dict],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Chạy validation theo đúng các fold trong src.cv_validation.time_series_split().

    - validation_metrics.csv sẽ có đủ Fold A, B, C cho cả Revenue và COGS.
    - Bổ sung metric R^2.
    - validation_predictions_*.csv và các figure validation vẫn dùng Fold A
      để không làm thay đổi số lượng figure đang cố định là 16.
    """

    results = {}
    metric_rows = []

    figure_fold = "A"  # giữ figure validation theo Fold A như caption/report hiện tại

    for fold in FOLD_NAMES:
        train_mask, val_mask = time_series_split(dates, fold)

        train_mask = np.asarray(train_mask, dtype=bool)
        val_mask = np.asarray(val_mask, dtype=bool)

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            print(f"[WARN] Fold {fold} has empty train/validation set. Skipping.")
            continue

        val_dates = dates[val_mask].reset_index(drop=True)

        for target in TARGETS:
            preds = predict_pipeline_like_for_target(
                X_train=X[train_mask],
                y_train_log=y[target][train_mask],
                dates_train=dates[train_mask].reset_index(drop=True),
                years_train=years[train_mask],
                quarters_train=quarters[train_mask],
                X_pred=X[val_mask],
                quarters_pred=quarters[val_mask],
                target=target,
                best_params=best_params,
            )

            actual = feat.loc[val_mask, target].reset_index(drop=True)
            pred = pd.Series(preds["final"]).reset_index(drop=True)
            residual = pred - actual

            mae = float(np.mean(np.abs(residual)))
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            smape = float(
                np.mean(
                    2 * np.abs(pred - actual)
                    / np.maximum(np.abs(actual) + np.abs(pred), 1e-9)
                )
            )

            # R^2 = 1 - SS_res / SS_tot
            ss_res = float(np.sum((actual - pred) ** 2))
            ss_tot = float(np.sum((actual - actual.mean()) ** 2))
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

            metric_rows.append({
                "fold": fold,
                "target": target,
                "train_start": pd.Timestamp(dates[train_mask].min()).date().isoformat(),
                "train_end": pd.Timestamp(dates[train_mask].max()).date().isoformat(),
                "val_start": pd.Timestamp(dates[val_mask].min()).date().isoformat(),
                "val_end": pd.Timestamp(dates[val_mask].max()).date().isoformat(),
                "n_train": int(train_mask.sum()),
                "n_val": int(val_mask.sum()),
                "MAE": mae,
                "RMSE": rmse,
                "sMAPE": smape,
                "R^2": r2,
            })

            # Chỉ lưu prediction chi tiết cho Fold A để các hàm vẽ figure phía sau không bị lỗi
            if fold == figure_fold:
                results[target] = pd.DataFrame({
                    "Date": val_dates,
                    "actual": actual,
                    "pred_final_ensemble": pred,
                    "residual": residual,
                    "lgb_base": preds["lgb_base"],
                    "q_specialist": preds["q_specialist"],
                    "lgb_blend": preds["lgb_blend"],
                    "ridge": preds["ridge"],
                    "raw_before_calibration": preds["raw_before_calibration"],
                })

    metrics = pd.DataFrame(metric_rows)

    if not metrics.empty:
        metrics = metrics.sort_values(["fold", "target"]).reset_index(drop=True)

    if not results:
        raise RuntimeError("No validation prediction results were generated for figure fold A.")

    return results, metrics

    metrics = pd.DataFrame(metric_rows)
    return results, metrics


# ==============================
# Model figures
# ==============================
def fig_10_11_feature_importance(
    feat: pd.DataFrame,
    feature_cols: Sequence[str],
    X: np.ndarray,
    y: Dict[str, np.ndarray],
    dates: pd.Series,
    years: np.ndarray,
    best_params: Optional[dict],
    out_dir: Path,
) -> pd.DataFrame:
    weights = make_weights(years)
    all_imp = []
    for i, target in enumerate(TARGETS, start=10):
        imp = fit_base_lgb_for_importance(X, y[target], weights, dates, feature_cols, best_params)
        imp["target"] = target
        all_imp.append(imp)
        top = imp.head(20).iloc[::-1]
        plt.figure(figsize=(9.5, 6.3))
        plt.barh(top["feature"], top["importance_gain"])
        plt.title(f"Top 20 Base LightGBM Feature Importances — {target}")
        plt.xlabel("Importance by gain")
        plt.ylabel("Feature")
        plt.grid(True, axis="x", alpha=0.25)
        save_fig(out_dir / EXPECTED_FIGURES[i - 1])
    output = pd.concat(all_imp, ignore_index=True)
    output.to_csv(out_dir / "feature_importance.csv", index=False)
    print(f"[OK] saved {out_dir / 'feature_importance.csv'}")
    return output


def fig_12_13_validation_actual_pred(validation_results: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    for idx, target in enumerate(TARGETS, start=12):
        df = validation_results[target]
        plt.figure(figsize=(11, 5.2))
        plt.plot(df["Date"], df["actual"], label=f"Actual {target}")
        plt.plot(df["Date"], df["pred_final_ensemble"], label="Predicted final ensemble")
        plt.title(f"Fold A Validation: Actual vs Final Ensemble Prediction — {target}")
        plt.xlabel("Date")
        plt.ylabel(target)
        plt.legend()
        plt.grid(True, alpha=0.25)
        save_fig(out_dir / EXPECTED_FIGURES[idx - 1])


def fig_14_validation_residuals(validation_results: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    plt.figure(figsize=(9.5, 5.5))
    for target in TARGETS:
        df = validation_results[target]
        plt.scatter(df["actual"], df["residual"], s=13, alpha=0.55, label=target)
    plt.axhline(0, linestyle=":", linewidth=1.4)
    plt.title("Fold A Validation Residuals — Final Ensemble")
    plt.xlabel("Actual value")
    plt.ylabel("Prediction - Actual")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_fig(out_dir / EXPECTED_FIGURES[13])


# ==============================
# Forecast + pipeline figures
# ==============================
def fig_15_submission_monthly_forecast(sub: Optional[pd.DataFrame], out_dir: Path) -> None:
    if sub is None:
        save_notice_figure(
            out_dir / EXPECTED_FIGURES[14],
            "Submission forecast unavailable",
            "Run python run_pipeline.py first to create submission.csv, then run this figure script again.",
        )
        return
    monthly = sub.set_index("Date")[["Revenue", "COGS"]].resample("MS").sum()
    plt.figure(figsize=(10.5, 5.2))
    plt.plot(monthly.index, monthly["Revenue"], marker="o", label="Forecast Revenue")
    plt.plot(monthly.index, monthly["COGS"], marker="o", label="Forecast COGS")
    plt.title("Monthly Forecast on Kaggle Test Horizon")
    plt.xlabel("Date")
    plt.ylabel("Monthly forecast total")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_fig(out_dir / EXPECTED_FIGURES[14])


def fig_16_model_pipeline_overview(out_dir: Path, submission_status: str, has_best_params: bool) -> None:
    """
    Draw a clean model-aligned pipeline figure.

    Giữ nguyên tham số submission_status và has_best_params để không cần sửa main(),
    nhưng không hiển thị chúng trên figure vì dễ làm hình bị rối.
    """
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axis("off")

    boxes = [
        # row 1
        (0.04, 0.72, 0.18, 0.14, "Input\nsales.csv\nDate, Revenue, COGS"),
        (0.29, 0.72, 0.22, 0.14, "Feature engineering\nbuild_features(Date)\ncalendar, Fourier,\nTet, holidays, promo"),
        (0.59, 0.72, 0.18, 0.14, "Targets\nlog1p(Revenue)\nlog1p(COGS)"),

        # row 2
        (0.04, 0.45, 0.18, 0.14, "Tuning\nrun_tuning.py\nOptuna on Fold A"),
        (0.29, 0.45, 0.22, 0.14, "Sample weights\n2014–2018: 1.0\nother years: 0.01"),
        (0.59, 0.45, 0.18, 0.14, "LightGBM\nbase model\n+ Q1–Q4 specialists"),
        (0.82, 0.45, 0.14, 0.14, "LGB blend\n0.60 specialist\n0.40 base"),

        # row 3
        (0.29, 0.18, 0.22, 0.14, "Ridge layer\nstandardized X\nlinear fallback"),
        (0.59, 0.18, 0.18, 0.14, "Raw ensemble\n0.20 Ridge\n0.80 LGB blend"),
        (0.82, 0.18, 0.14, 0.14, "Calibration\nRevenue × 1.26\nCOGS × 1.32"),

        # output
        (0.82, 0.01, 0.14, 0.11, "Output\nsubmission.csv"),
    ]

    for x, y, w, h, text in boxes:
        rect = plt.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            linewidth=1.6,
            edgecolor="black"
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=9.5
        )

    arrows = [
        # top row
        ((0.22, 0.79), (0.29, 0.79)),
        ((0.51, 0.79), (0.59, 0.79)),

        # input to tuning
        ((0.13, 0.72), (0.13, 0.59)),

        # middle row
        ((0.22, 0.52), (0.29, 0.52)),
        ((0.51, 0.52), (0.59, 0.52)),
        ((0.77, 0.52), (0.82, 0.52)),

        # ridge to raw ensemble
        ((0.51, 0.25), (0.59, 0.25)),

        # LGB blend to raw ensemble
        ((0.89, 0.45), (0.68, 0.32)),

        # raw ensemble to calibration
        ((0.77, 0.25), (0.82, 0.25)),

        # calibration to output
        ((0.89, 0.18), (0.89, 0.12)),
    ]

    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=1.6)
        )

    ax.set_title(
        "Task 3 Model-Aligned Forecasting Pipeline",
        fontsize=15,
        pad=16
    )

    save_fig(out_dir / EXPECTED_FIGURES[15])


# ==============================
# Metadata
# ==============================
def write_captions(out_dir: Path) -> None:
    captions = """# Task 3 figure captions — model-aligned version

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
"""
    path = out_dir / "figure_captions.md"
    path.write_text(captions, encoding="utf-8")
    print(f"[OK] saved {path}")


def write_manifest(out_dir: Path) -> None:
    rows = []
    for filename in EXPECTED_FIGURES:
        path = out_dir / filename
        rows.append({
            "figure": filename,
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "figure_manifest.csv", index=False)
    n = int(manifest["exists"].sum())
    print(f"[OK] saved {out_dir / 'figure_manifest.csv'}")
    print(f"[CHECK] {n}/16 expected PNG figures exist in {out_dir}")
    if n != 16:
        missing = manifest.loc[~manifest["exists"], "figure"].tolist()
        raise RuntimeError(f"Expected 16 figures, but missing: {missing}")


def create_model_placeholders(out_dir: Path, reason: str) -> None:
    placeholders = {
        9: ("Feature importance unavailable", reason),
        10: ("Feature importance unavailable", reason),
        11: ("Validation figure unavailable", reason),
        12: ("Validation figure unavailable", reason),
        13: ("Validation residual figure unavailable", reason),
    }
    for idx, (title, message) in placeholders.items():
        path = out_dir / EXPECTED_FIGURES[idx]
        if path.exists():
            # Do not overwrite a model figure that was already generated successfully.
            continue
        save_notice_figure(path, title, message)


# ==============================
# Main
# ==============================
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exactly 16 model-aligned Task 3 report figures.")
    parser.add_argument("--raw-dir", default="data/raw", help="Folder containing sales.csv")
    parser.add_argument("--submission", default="submission.csv", help="Path to submission.csv produced by run_pipeline.py")
    parser.add_argument("--out-dir", default="reports", help="Output folder; default is reports")
    parser.add_argument("--best-params", default="best_lgb_params.json", help="Path to best_lgb_params.json from run_tuning.py")
    parser.add_argument("--skip-model-figures", action="store_true", help="Skip expensive model-based training figures but still create 16 files via notice placeholders")
    parser.add_argument("--clean", action="store_true", help="Delete the output folder before regenerating figures")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    ensure_clean_out(out_dir, clean=args.clean)

    sales = load_sales(raw_dir)
    sub = load_submission(Path(args.submission))
    best_params_path = Path(args.best_params)
    best_params = load_best_params(best_params_path)

    feat = add_engineered_features(sales)
    feature_cols, X, y, dates, years, quarters = get_model_matrices(feat)

    print(f"[INFO] training rows: {len(sales)}")
    print(f"[INFO] model feature count: {len(feature_cols)}")
    print(f"[INFO] submission horizon: {check_submission_horizon(sub)}")
    print(f"[INFO] output dir: {out_dir.resolve()}")

    # Non-model figures: always generated.
    fig_01_monthly_history_and_forecast(sales, sub, out_dir)
    fig_02_03_heatmaps(sales, out_dir)
    fig_04_average_daily_by_month(feat, out_dir)
    fig_05_average_revenue_by_weekday(feat, out_dir)
    fig_06_calendar_event_impact(feat, out_dir)
    fig_07_cv_folds_from_code(feat, out_dir)
    fig_08_lgb_internal_split(feat, out_dir)
    fig_09_sample_weights(feat, out_dir)

    # Model figures: generated with the same training functions as the model.
    if args.skip_model_figures:
        create_model_placeholders(out_dir, "Skipped by --skip-model-figures. Remove that flag to train model-aligned figures.")
    else:
        try:
            fig_10_11_feature_importance(feat, feature_cols, X, y, dates, years, best_params, out_dir)
            validation_results, metrics = run_model_aligned_validation(
                feat=feat,
                feature_cols=feature_cols,
                X=X,
                y=y,
                dates=dates,
                years=years,
                quarters=quarters,
                best_params=best_params,
            )
            metrics.to_csv(out_dir / "validation_metrics.csv", index=False)
            print(f"[OK] saved {out_dir / 'validation_metrics.csv'}")
            for target, df in validation_results.items():
                df.to_csv(out_dir / f"validation_predictions_{target.lower()}.csv", index=False)
                print(f"[OK] saved {out_dir / f'validation_predictions_{target.lower()}.csv'}")
            fig_12_13_validation_actual_pred(validation_results, out_dir)
            fig_14_validation_residuals(validation_results, out_dir)
        except Exception as exc:
            print("[WARN] Could not generate model-based figures. Creating placeholders instead.")
            print("[WARN]", str(exc))
            traceback.print_exc()
            create_model_placeholders(out_dir, f"Model-based figure generation failed: {exc}")

    fig_15_submission_monthly_forecast(sub, out_dir)
    fig_16_model_pipeline_overview(out_dir, check_submission_horizon(sub), has_best_params=best_params is not None)

    write_captions(out_dir)
    write_manifest(out_dir)

    print("\nDone. Figures are in:", out_dir.resolve())


if __name__ == "__main__":
    main()
