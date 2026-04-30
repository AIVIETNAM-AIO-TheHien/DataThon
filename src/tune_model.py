import optuna
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error
import warnings

# Tắt cảnh báo linh tinh để terminal sạch sẽ
warnings.filterwarnings('ignore')

def objective(trial, X_train, y_train, X_val, y_val):
    # Định nghĩa không gian tìm kiếm siêu tham số
    param = {
        'objective': 'regression',
        'metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1  # Tắt log quá trình chạy của LightGBM
    }

    # Huấn luyện mô hình
    model = lgb.LGBMRegressor(**param)
    
    # Thiết lập dừng sớm nếu mô hình không cải thiện
    callbacks_list = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks_list
    )

    # Dự báo trên không gian Log
    preds_log = model.predict(X_val)
    
    # Kéo ngược về giá trị thực để tính MAE chuẩn
    y_val_real = np.expm1(y_val)
    preds_real = np.expm1(preds_log)
    
    # Tính toán sai số thực tế
    mae = mean_absolute_error(y_val_real, preds_real)
    return mae

def run_tuning(X_train, y_train, X_val, y_val, target_name="Target", n_trials=30):
    print(f"\n[OPTUNA] Bắt đầu tự động dò tìm siêu tham số cho {target_name}...")
    
    # Tắt log mặc định của Optuna cho đỡ rối mắt
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
    
    print(f"--- Kết quả tối ưu cho {target_name} ---")
    print("Bộ tham số tốt nhất:")
    print(study.best_params)
    print(f"MAE kịch sàn đạt được: {study.best_value:.2f}")
    
    return study.best_params