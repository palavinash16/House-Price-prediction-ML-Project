"""
Model training, serialization, and prediction module for House Price Prediction.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from .evaluation import compute_regression_metrics


class HousePriceModel:
    """
    Wrapper class for training, evaluating, saving, and deploying ML models.
    """
    def __init__(self, model_type: str = 'linear', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=random_state)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=100.0, random_state=random_state)
        elif model_type == 'elasticnet':
            self.model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=random_state)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=random_state)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=random_state)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit scaler and regression model.
        """
        self.feature_names = list(X_train.columns)
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict target house prices.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict()")
        
        # Ensure column alignment
        X_aligned = X[self.feature_names]
        X_scaled = self.scaler.transform(X_aligned)
        return self.model.predict(X_scaled)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Compute regression evaluation metrics.
        """
        y_pred = self.predict(X_test)
        metrics = compute_regression_metrics(y_test, y_pred)
        return metrics

    def save(self, filepath: str):
        """
        Serialize model and scaler to disk.
        """
        artifact = {
            'model_type': self.model_type,
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(artifact, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'HousePriceModel':
        """
        Load serialized model from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        artifact = joblib.load(filepath)
        instance = cls(model_type=artifact['model_type'])
        instance.model = artifact['model']
        instance.scaler = artifact['scaler']
        instance.feature_names = artifact['feature_names']
        instance.is_trained = artifact['is_trained']
        return instance


def train_and_evaluate_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Dict[str, HousePriceModel], Dict[str, Dict[str, float]], str]:
    """
    Train all model architectures and return models, metrics, and best model name.
    """
    model_types = ['linear', 'ridge', 'lasso', 'elasticnet', 'random_forest', 'gradient_boosting']
    models = {}
    metrics_summary = {}
    best_model_name = 'linear'
    best_r2 = -float('inf')

    for m_type in model_types:
        model_inst = HousePriceModel(model_type=m_type)
        model_inst.fit(X_train, y_train)
        metrics = model_inst.evaluate(X_test, y_test)
        
        models[m_type] = model_inst
        metrics_summary[m_type] = metrics
        
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model_name = m_type

    return models, metrics_summary, best_model_name


def export_weights_json(model: HousePriceModel, filepath: str, metrics: Dict[str, float]):
    """
    Export model parameters and scaler weights as JSON for JavaScript web predictor.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    scaler_mean = model.scaler.mean_.tolist()
    scaler_scale = model.scaler.scale_.tolist()
    
    if hasattr(model.model, 'coef_'):
        coefficients = model.model.coef_.tolist()
        intercept = float(model.model.intercept_)
    else:
        # Fallback for tree models
        coefficients = [0.0] * len(model.feature_names)
        intercept = float(np.mean(model.scaler.mean_))

    export_data = {
        'model_type': model.model_type,
        'feature_names': model.feature_names,
        'coefficients': coefficients,
        'intercept': intercept,
        'scaler_mean': scaler_mean,
        'scaler_scale': scaler_scale,
        'metrics': metrics
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
