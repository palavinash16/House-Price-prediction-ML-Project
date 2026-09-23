"""
FastAPI REST Server for House Price Prediction ML System.
Run with:
    uvicorn app_api:app --reload --port 8000
"""

import os
import json
import pandas as pd
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models import HousePriceModel
from src.data import clean_missing_values
from src.features import engineer_features

app = FastAPI(
    title="House Price Prediction API",
    description="Machine Learning REST API service for property valuation.",
    version="1.0.0"
)

# Enable CORS for browser frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/house_price_model.joblib"
META_PATH = "models/metadata.json"

# Global lazy model loader
model_instance: Optional[HousePriceModel] = None
meta_instance: Dict[str, Any] = {}


def get_model_and_meta():
    global model_instance, meta_instance
    if model_instance is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please train model first.")
        model_instance = HousePriceModel.load(MODEL_PATH)
        
        if os.path.exists(META_PATH):
            with open(META_PATH, 'r', encoding='utf-8') as f:
                meta_instance = json.load(f)
    return model_instance, meta_instance


class HousePredictionInput(BaseModel):
    Area_SqFt: float = Field(..., example=2200.0, description="Total property square footage")
    Bedrooms: int = Field(..., example=3, description="Number of bedrooms")
    Bathrooms: int = Field(..., example=2, description="Number of bathrooms")
    Stories: int = Field(..., example=2, description="Number of stories / floors")
    House_Age_Years: float = Field(..., example=10.0, description="Age of the house in years")
    Garage_Cars: int = Field(..., example=2, description="Garage car capacity")
    Neighborhood: str = Field(..., example="Suburbs", description="Neighborhood (Suburbs, Downtown, Waterfront, Highland, Oldtown)")
    Furnishing: str = Field(..., example="Semi-Furnished", description="Furnishing status (Furnished, Semi-Furnished, Unfurnished)")
    Has_Pool: str = Field(..., example="No", description="Has swimming pool ('Yes' or 'No')")


class PredictionOutput(BaseModel):
    estimated_price: float
    formatted_price: str
    confidence_interval_95: Dict[str, float]
    model_used: str
    features_received: Dict[str, Any]


@app.get("/")
def root():
    return {
        "message": "Welcome to House Price Prediction ML API",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health_check():
    try:
        model, meta = get_model_and_meta()
        return {
            "status": "healthy",
            "model_loaded": True,
            "best_model": meta.get("best_model", "unknown")
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/metrics")
def get_metrics():
    _, meta = get_model_and_meta()
    return meta


@app.post("/predict", response_model=PredictionOutput)
def predict_price(house: HousePredictionInput):
    try:
        model, meta = get_model_and_meta()
        input_dict = house.dict()
        
        df_raw = pd.DataFrame([input_dict])
        df_clean, _ = clean_missing_values(df_raw, imputation_values=meta.get('imputation_values', {}))
        
        training_cols = meta.get('feature_columns', [])
        df_eng, _ = engineer_features(df_clean, training_columns=training_cols)
        
        pred = float(model.predict(df_eng)[0])
        rmse = meta.get('test_metrics', {}).get('rmse', 28000.0)
        
        lower_bound = max(0.0, pred - (1.96 * rmse))
        upper_bound = pred + (1.96 * rmse)
        
        return {
            "estimated_price": round(pred, 2),
            "formatted_price": f"${pred:,.2f}",
            "confidence_interval_95": {
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2)
            },
            "model_used": meta.get("best_model", model.model_type),
            "features_received": input_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
