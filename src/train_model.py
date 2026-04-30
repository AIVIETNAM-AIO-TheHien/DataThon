import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import warnings
import logging

# Tắt log rác
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')

# ================== LIGHTGBM ==================
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.03,
    'num_leaves': 63,
    'min_data_in_leaf': 30,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'lambda_l2': 1.0,
    'seed': 42,
    'verbosity': -1,
}

def train_lgb_with_weight(X, y, w, dates, num_boost_es=5000, early_stop=300, custom_params=None):
    params = LGB_PARAMS.copy()
    if custom_params:
        params.update(custom_params)
    
    intern = pd.Timestamp('2022-07-04')
    fit_idx = (dates <= intern).values
    ins_idx = (dates > intern).values

    # Nếu không có mẫu validation -> train toàn bộ không early stopping
    if np.sum(ins_idx) == 0:
        print("   [WARN] No validation data available, training without early stopping.")
        full_data = lgb.Dataset(X, y, weight=w)
        bf = lgb.train(params, full_data, num_boost_round=num_boost_es)
        return bf

    train_data = lgb.Dataset(X[fit_idx], y[fit_idx], weight=w[fit_idx])
    val_data = lgb.Dataset(X[ins_idx], y[ins_idx])
    callbacks = [lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)]
    
    bk = lgb.train(params, train_data, num_boost_round=num_boost_es,
                   valid_sets=[val_data], callbacks=callbacks)
    
    full_data = lgb.Dataset(X, y, weight=w)
    bf = lgb.train(params, full_data, num_boost_round=bk.best_iteration)
    return bf

def train_q_specialist(X, y, w_base, quarters, dates, target_q, q_boost=2.0, custom_params=None):
    w = w_base.copy()
    mask = (quarters == target_q)
    w[mask] = w[mask] * q_boost
    return train_lgb_with_weight(X, y, w, dates, num_boost_es=3000, early_stop=200, custom_params=custom_params)

# ================== RIDGE ==================
def train_ridge(X_train, y_train, alpha=3.0):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X_train - mu) / sigma
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(Xs, y_train)
    return model, (mu, sigma)

def predict_ridge(model, X_test, stats):
    mu, sigma = stats
    return model.predict((X_test - mu) / sigma)

# ================== PROPHET (xử lý lỗi) ==================
def safe_fit_prophet(df, y_col='y', promo_cols=None, post_regime_only=True):
    """
    Huấn luyện Prophet an toàn, trả về None nếu lỗi.
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("   [WARN] Prophet chưa được cài đặt. Bỏ qua.")
        return None, None

    train_df = df.copy()
    if post_regime_only:
        train_df = train_df[train_df['ds'] >= '2020-01-01']
    
    # Xử lý missing
    train_df = train_df.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_df) < 100:
        print("   [WARN] Không đủ dữ liệu cho Prophet sau khi lọc.")
        return None, None

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                seasonality_mode='additive', changepoint_prior_scale=0.05)
    if promo_cols:
        for col in promo_cols:
            if col in train_df.columns:
                m.add_regressor(col)
    m.fit(train_df[['ds', y_col] + (promo_cols if promo_cols else [])])
    return m, promo_cols

def safe_predict_prophet(model, test_df, promo_cols):
    if model is None:
        return np.full(len(test_df), np.nan)
    try:
        forecast = model.predict(test_df[['ds'] + (promo_cols if promo_cols else [])])
        return forecast['yhat'].values
    except:
        return np.full(len(test_df), np.nan)