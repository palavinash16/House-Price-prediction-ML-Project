"""
Unit tests for models module.
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.models import HousePriceModel


def test_house_price_model_training_and_prediction(tmp_path):
    df_train = pd.DataFrame({
        'Area_SqFt': [1500.0, 1800.0, 2000.0, 2200.0, 2500.0],
        'Bedrooms': [2, 3, 3, 4, 4],
        'Bathrooms': [1, 2, 2, 3, 3],
        'Stories': [1, 2, 2, 2, 3],
        'House_Age_Years': [10.0, 15.0, 20.0, 5.0, 2.0]
    })
    y_train = pd.Series([300000.0, 450000.0, 500000.0, 600000.0, 700000.0])
    
    model = HousePriceModel(model_type='linear')
    model.fit(df_train, y_train)
    
    preds = model.predict(df_train)
    assert len(preds) == 5
    assert preds[0] > 0
    
    save_path = os.path.join(tmp_path, 'model.joblib')
    model.save(save_path)
    assert os.path.exists(save_path)
    
    loaded_model = HousePriceModel.load(save_path)
    loaded_preds = loaded_model.predict(df_train)
    np.testing.assert_almost_equal(preds, loaded_preds)
