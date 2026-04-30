import pandas as pd
import numpy as np
import json
from src.data_prep import build_features
from src.train_model import (
    train_lgb_with_weight, train_q_specialist,
    train_ridge, predict_ridge
)
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    print("--- PIPELINE DATATHON 2026 (RIDGE + LGB BLEND, NO PROPHET) ---")
    
    # 0. Load best params nếu có
    try:
        with open('best_lgb_params.json', 'r') as f:
            best_lgb_params = json.load(f)
        print("Loaded best LGB params from file.")
    except FileNotFoundError:
        best_lgb_params = None
        print("No best_lgb_params.json found, using default params.")
    
    # 1. Load và feature engineering
    print("1. Đang tạo đặc trưng Calendar-only...")
    sales = pd.read_csv('data/raw/sales.csv', parse_dates=['Date'])
    test_dates = pd.date_range('2023-01-01', '2024-07-01', freq='D')
    
    feat_train = build_features(sales['Date'])
    feat_train['Revenue'] = sales['Revenue'].values
    feat_train['COGS'] = sales['COGS'].values
    feat_test = build_features(test_dates)
    
    NON_FEATURES = ['Date', 'Revenue', 'COGS']
    cols = [c for c in feat_train.columns if c not in NON_FEATURES]
    
    X_tr = feat_train[cols].values.astype(float)
    X_te = feat_test[cols].values.astype(float)
    y_rev = np.log1p(feat_train['Revenue'].values)
    y_cog = np.log1p(feat_train['COGS'].values)
    
    dates_tr = feat_train['Date']
    years_tr = feat_train['year'].values
    quarters_tr = feat_train['quarter'].values
    quarters_te = feat_test['quarter'].values
    
    # Sample weight (ưu tiên 2014-2018)
    w_full = np.full(len(years_tr), 0.01)
    w_full[(years_tr >= 2014) & (years_tr <= 2018)] = 1.0
    
    # ========== TẦNG 1: LGB BASE & Q-SPECIALISTS ==========
    print("\n2. Tầng 1: LightGBM Base...")
    lgb_base_rev = train_lgb_with_weight(X_tr, y_rev, w_full, dates_tr, custom_params=best_lgb_params)
    lgb_base_cog = train_lgb_with_weight(X_tr, y_cog, w_full, dates_tr, custom_params=best_lgb_params)
    p_lgb_base_rev = np.expm1(lgb_base_rev.predict(X_te))
    p_lgb_base_cog = np.expm1(lgb_base_cog.predict(X_te))
    
    print("3. Tầng 1: Q-Specialists (4 quý)...")
    spec_rev, spec_cog = {}, {}
    for q in [1, 2, 3, 4]:
        print(f"   -> Q{q}")
        model_rev = train_q_specialist(X_tr, y_rev, w_full, quarters_tr, dates_tr, q, custom_params=best_lgb_params)
        spec_rev[q] = np.expm1(model_rev.predict(X_te))
        model_cog = train_q_specialist(X_tr, y_cog, w_full, quarters_tr, dates_tr, q, custom_params=best_lgb_params)
        spec_cog[q] = np.expm1(model_cog.predict(X_te))
    
    p_spec_rev = np.zeros(len(test_dates))
    p_spec_cog = np.zeros(len(test_dates))
    for q in [1, 2, 3, 4]:
        mask = (quarters_te == q)
        p_spec_rev[mask] = spec_rev[q][mask]
        p_spec_cog[mask] = spec_cog[q][mask]
    
    ALPHA = 0.60
    lgb_blend_rev = ALPHA * p_spec_rev + (1 - ALPHA) * p_lgb_base_rev
    lgb_blend_cog = ALPHA * p_spec_cog + (1 - ALPHA) * p_lgb_base_cog
    
    # ========== TẦNG 2: RIDGE ==========
    print("\n4. Tầng 2: Ridge Regression...")
    ridge_rev, stats_rev = train_ridge(X_tr, y_rev)
    ridge_cog, stats_cog = train_ridge(X_tr, y_cog)
    p_rd_rev = np.expm1(predict_ridge(ridge_rev, X_te, stats_rev))
    p_rd_cog = np.expm1(predict_ridge(ridge_cog, X_te, stats_cog))
    
    # ========== BỎ PROPHET, DÙNG RIDGE FALLBACK (DO LỖI WINDOWS) ==========
    p_pr_rev = p_rd_rev.copy()
    p_pr_cog = p_rd_cog.copy()
    
    # ========== TẦNG 3: ENSEMBLE & CALIBRATION ==========
    print("\n6. Ensemble 2 tầng (Ridge + LGB blend)...")
    # Blend: Ridge (20%) + LGB blend (80%)
    raw_rev = 0.20 * p_rd_rev + 0.80 * lgb_blend_rev
    raw_cog = 0.20 * p_rd_cog + 0.80 * lgb_blend_cog
    
    # Tầng 3: Calibration
    CR, CC = 1.26, 1.32
    final_rev = CR * raw_rev
    final_cog = CC * raw_cog
    
    # ========== XUẤT FILE ==========
    print("\n7. Xuất file submission...")
    submission = pd.DataFrame({
        'Date': test_dates.strftime('%Y-%m-%d'),
        'Revenue': final_rev,
        'COGS': final_cog
    })
    out_path = 'submission.csv'
    submission.to_csv(out_path, index=False)
    print(f"\n[THÀNH CÔNG] File đã lưu tại: {out_path}")