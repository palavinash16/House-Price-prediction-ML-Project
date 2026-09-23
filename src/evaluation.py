"""
Evaluation metrics and diagnostic tools for House Price Prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute full suite of regression performance metrics.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    # Adjusted R2
    n = len(y_true)
    p = 10  # approximate feature count
    adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else r2
    
    # MAPE (Mean Absolute Percentage Error)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    
    return {
        'mse': round(mse, 2),
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'r2': round(r2, 4),
        'adj_r2': round(adj_r2, 4),
        'mape': round(mape, 2)
    }


def evaluate_residuals(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate residual homoscedasticity and distribution.
    """
    residuals = y_true - y_pred
    return {
        'mean_residual': float(round(np.mean(residuals), 2)),
        'std_residual': float(round(np.std(residuals), 2)),
        'max_positive_error': float(round(np.max(residuals), 2)),
        'max_negative_error': float(round(np.min(residuals), 2))
    }


def get_feature_importances(model_obj: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importances or linear coefficients.
    """
    if hasattr(model_obj, 'coef_'):
        imp = model_obj.coef_
    elif hasattr(model_obj, 'feature_importances_'):
        imp = model_obj.feature_importances_
    else:
        imp = np.zeros(len(feature_names))
        
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': imp,
        'Abs_Importance': np.abs(imp)
    }).sort_values(by='Abs_Importance', ascending=False).reset_index(drop=True)
    
    return df_imp
