# src/cv_validation.py
import pandas as pd
import numpy as np

def time_series_split(dates, fold_name):
    """
    Chia train/val theo thời gian cho CV.
    dates: pandas Series hoặc array các datetime
    fold_name: 'A' (val 2022), 'B' (val 2021), 'C' (rolling 12 months)
    Trả về: (train_mask, val_mask) là boolean array cùng length với dates
    """
    if isinstance(dates, pd.Series):
        dates = dates.values
    # Chuyển sang datetime nếu chưa
    if not isinstance(dates[0], pd.Timestamp):
        dates = pd.to_datetime(dates)
    
    if fold_name == 'A':
        train_mask = dates < pd.Timestamp('2022-01-01')
        val_mask = (dates >= pd.Timestamp('2022-01-01')) & (dates <= pd.Timestamp('2022-12-31'))
    elif fold_name == 'B':
        train_mask = dates < pd.Timestamp('2021-01-01')
        val_mask = (dates >= pd.Timestamp('2021-01-01')) & (dates <= pd.Timestamp('2021-12-31'))
    elif fold_name == 'C':
        train_mask = (dates >= pd.Timestamp('2020-07-01')) & (dates < pd.Timestamp('2021-07-01'))
        val_mask = (dates >= pd.Timestamp('2021-07-01')) & (dates <= pd.Timestamp('2022-06-30'))
    else:
        raise ValueError("fold_name must be A, B, or C")
    
    return train_mask, val_mask

def evaluate_on_fold(X, y, dates, w, model_func, **kwargs):
    """
    Đánh giá một mô hình (ví dụ LightGBM) trên một fold cụ thể.
    model_func: hàm train có signature (X, y, w, dates, ...) và trả về model đã train.
    Trả về: MAE trên validation fold.
    """
    train_mask, val_mask = time_series_split(dates, kwargs.pop('fold_name', 'A'))
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    w_train = w[train_mask]
    dates_train = dates[train_mask]
    
    model = model_func(X_train, y_train, w_train, dates_train, **kwargs)
    preds_log = model.predict(X_val)
    preds_real = np.expm1(preds_log)
    y_real = np.expm1(y_val)
    mae = np.mean(np.abs(preds_real - y_real))
    return mae