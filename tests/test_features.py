"""
Unit tests for features module.
"""

import pytest
import pandas as pd
from src.features import add_engineered_features, encode_categorical_features, engineer_features


def test_add_engineered_features():
    df = pd.DataFrame({
        'Area_SqFt': [2000.0],
        'Bedrooms': [4],
        'Bathrooms': [2],
        'Stories': [2],
        'Garage_Cars': [2],
        'Has_Pool': ['Yes']
    })
    df_feat = add_engineered_features(df)
    
    assert 'SqFt_per_Bedroom' in df_feat.columns
    assert 'Bath_Bed_Ratio' in df_feat.columns
    assert 'Luxury_Score' in df_feat.columns
    assert round(df_feat['SqFt_per_Bedroom'].iloc[0], 2) == round(2000.0 / 4.1, 2)


def test_encode_categorical_features():
    df = pd.DataFrame({
        'Area_SqFt': [2000.0],
        'Neighborhood': ['Suburbs'],
        'Furnishing': ['Furnished'],
        'Has_Pool': ['Yes']
    })
    df_enc, cols = encode_categorical_features(df)
    assert isinstance(cols, list)
    assert 'Neighborhood_Waterfront' not in df_enc.columns or df_enc['Neighborhood_Waterfront'].iloc[0] == 0
