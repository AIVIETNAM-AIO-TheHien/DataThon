# src/tune_hyperparams.py
import optuna
import numpy as np
import lightgbm as lgb
from src.cv_validation import evaluate_on_fold
from src.train_model import train_lgb_with_weight

def objective_lgb(trial, X, y, w, dates, base_params):
    """
    Tune các hyperparameter của LightGBM.
    """
    params = base_params.copy()
    params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    params['num_leaves'] = trial.suggest_int('num_leaves', 20, 150)
    params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 10, 100)
    params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.6, 1.0)
    params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.6, 1.0)
    params['lambda_l2'] = trial.suggest_float('lambda_l2', 0.1, 10.0, log=True)
    
    # Định nghĩa hàm train với params mới - truyền qua custom_params
    def train_with_custom_params(X_tr, y_tr, w_tr, dates_tr, **kwargs):
        return train_lgb_with_weight(X_tr, y_tr, w_tr, dates_tr, 
                                     custom_params=params, **kwargs)
    
    mae = evaluate_on_fold(X, y, dates, w, train_with_custom_params, 
                           fold_name='A', num_boost_es=1000, early_stop=100)
    return mae

def tune_lgb(X, y, w, dates, n_trials=30):
    """
    Chạy Optuna tuning cho LightGBM.
    """
    base_params = {
        'objective': 'regression',
        'metric': 'mae',
        'seed': 42,
        'verbosity': -1,
    }
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_lgb(trial, X, y, w, dates, base_params), 
                   n_trials=n_trials)
    print("Best params:", study.best_params)
    print("Best MAE on fold A:", study.best_value)
    return study.best_params