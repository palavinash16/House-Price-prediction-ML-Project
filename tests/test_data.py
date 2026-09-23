"""
Unit tests for data module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data import clean_missing_values, preprocess_data


def test_clean_missing_values():
    df_raw = pd.DataFrame({
        'Area_SqFt': [1500.0, np.nan, 2000.0],
        'House_Age_Years': [10.0, 20.0, np.nan],
        'Neighborhood': ['Suburbs', 'Downtown', 'Waterfront']
    })
    
    df_clean, imp = clean_missing_values(df_raw)
    assert df_clean['Area_SqFt'].isna().sum() == 0
    assert df_clean['House_Age_Years'].isna().sum() == 0
    assert imp['Area_SqFt'] == 1750.0
    assert imp['House_Age_Years'] == 15.0


def test_preprocess_data():
    df_raw = pd.DataFrame({
        'Area_SqFt': [1500.0, 1800.0, 2000.0, 2200.0, 2500.0],
        'Bedrooms': [2, 3, 3, 4, 4],
        'Bathrooms': [1, 2, 2, 3, 3],
        'Stories': [1, 2, 2, 2, 3],
        'House_Age_Years': [10.0, 15.0, 20.0, 5.0, 2.0],
        'Garage_Cars': [1, 2, 2, 2, 3],
        'Neighborhood': ['Suburbs', 'Downtown', 'Waterfront', 'Highland', 'Oldtown'],
        'Furnishing': ['Furnished', 'Semi-Furnished', 'Unfurnished', 'Furnished', 'Unfurnished'],
        'Has_Pool': ['No', 'Yes', 'No', 'Yes', 'No'],
        'Price': [300000.0, 450000.0, 500000.0, 600000.0, 700000.0]
    })
    
    X_train, y_train, X_test, y_test, meta = preprocess_data(df_raw, test_size=0.2, random_state=42)
    assert len(X_train) == 4
    assert len(X_test) == 1
    assert 'Price' not in X_train.columns
    assert len(y_train) == 4
