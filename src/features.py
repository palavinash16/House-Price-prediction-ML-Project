"""
Feature engineering and transformation pipeline for House Price Prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

EXPECTED_CATEGORICAL = {
    'Neighborhood': ['Suburbs', 'Downtown', 'Waterfront', 'Highland', 'Oldtown'],
    'Furnishing': ['Semi-Furnished', 'Unfurnished', 'Furnished'],
    'Has_Pool': ['No', 'Yes']
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific derived features.
    """
    df_feat = df.copy()
    
    # Area per bedroom ratio
    if 'Area_SqFt' in df_feat.columns and 'Bedrooms' in df_feat.columns:
        df_feat['SqFt_per_Bedroom'] = df_feat['Area_SqFt'] / (df_feat['Bedrooms'] + 0.1)
        
    # Bathroom to Bedroom ratio
    if 'Bathrooms' in df_feat.columns and 'Bedrooms' in df_feat.columns:
        df_feat['Bath_Bed_Ratio'] = df_feat['Bathrooms'] / (df_feat['Bedrooms'] + 0.1)
        
    # Luxury Index Score
    if 'Garage_Cars' in df_feat.columns and 'Has_Pool' in df_feat.columns and 'Stories' in df_feat.columns:
        pool_num = df_feat['Has_Pool'].apply(lambda x: 1.0 if str(x).strip().lower() in ['yes', '1', 'true'] else 0.0)
        df_feat['Luxury_Score'] = (df_feat['Garage_Cars'] * 1.5) + (pool_num * 2.0) + (df_feat['Stories'] * 0.5)
        
    return df_feat


def encode_categorical_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-Hot Encode categorical columns consistently.
    If feature_columns is provided, aligns columns to match training schema.
    """
    df_encoded = pd.get_dummies(df, columns=['Neighborhood', 'Furnishing', 'Has_Pool'], drop_first=True)
    
    # Convert all dummy boolean columns to float/int (0 and 1)
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool:
            df_encoded[col] = df_encoded[col].astype(int)
            
    if feature_columns is not None:
        # Reindex to ensure identical feature layout for prediction
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0.0
        df_encoded = df_encoded[feature_columns]
        cols = feature_columns
    else:
        cols = list(df_encoded.columns)
        
    return df_encoded, cols


def engineer_features(
    df: pd.DataFrame,
    training_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Full feature engineering & encoding pipeline.
    """
    df_eng = add_engineered_features(df)
    df_final, final_cols = encode_categorical_features(df_eng, feature_columns=training_columns)
    return df_final, final_cols
