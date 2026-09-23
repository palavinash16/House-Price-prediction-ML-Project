"""
Data processing and loading module for House Price Prediction.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV dataset from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def clean_missing_values(df: pd.DataFrame, imputation_values: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Clean missing values in dataset. Imputes missing continuous features with median.
    
    Returns:
        df_cleaned: Cleaned DataFrame
        imputation_dict: Dictionary of median values used for imputation
    """
    df_clean = df.copy()
    
    if imputation_values is None:
        imputation_dict = {}
        if 'Area_SqFt' in df_clean.columns:
            imputation_dict['Area_SqFt'] = float(df_clean['Area_SqFt'].median())
        if 'House_Age_Years' in df_clean.columns:
            imputation_dict['House_Age_Years'] = float(df_clean['House_Age_Years'].median())
    else:
        imputation_dict = imputation_values
        
    for col, val in imputation_dict.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(val)
            
    return df_clean, imputation_dict


def preprocess_data(
    df: pd.DataFrame,
    target_column: str = 'Price',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series], Dict[str, Any]]:
    """
    Full data loading & split pipeline.
    """
    df_clean, imputation_dict = clean_missing_values(df)
    
    if target_column in df_clean.columns:
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        metadata = {
            'imputation': imputation_dict,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': list(X.columns)
        }
        
        return X_train, y_train, X_test, y_test, metadata
    else:
        metadata = {
            'imputation': imputation_dict,
            'features': list(df_clean.columns)
        }
        return df_clean, None, None, None, metadata
