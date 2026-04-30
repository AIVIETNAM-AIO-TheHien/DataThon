# run_tuning.py
import pandas as pd
import numpy as np
import json
from src.data_prep import build_features
from src.train_model import train_lgb_with_weight
from src.tune_hyperparams import tune_lgb

if __name__ == "__main__":
    print("=== TUNING LIGHTGBM PARAMETERS ===")
    # Load dữ liệu
    sales = pd.read_csv('data/raw/sales.csv', parse_dates=['Date'])
    feat = build_features(sales['Date'])
    feat['Revenue'] = sales['Revenue'].values
    NON_FEATURES = ['Date', 'Revenue', 'COGS']
    cols = [c for c in feat.columns if c not in NON_FEATURES]
    X = feat[cols].values.astype(float)
    y_rev = np.log1p(feat['Revenue'].values)
    dates = feat['Date']
    years = feat['year'].values
    w = np.full(len(years), 0.01)
    w[(years >= 2014) & (years <= 2018)] = 1.0
    
    best_params = tune_lgb(X, y_rev, w, dates, n_trials=30)
    
    # Lưu lại best_params
    with open('best_lgb_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    print("Saved best params to best_lgb_params.json")