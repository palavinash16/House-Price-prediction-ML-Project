"""
Command Line Script for predicting house prices using trained ML model.
Usage:
    python predict.py --input '{"Area_SqFt": 2200, "Bedrooms": 3, "Bathrooms": 2, "Stories": 2, "House_Age_Years": 10, "Garage_Cars": 2, "Neighborhood": "Suburbs", "Furnishing": "Semi-Furnished", "Has_Pool": "No"}'
    python predict.py --csv input.csv --output output.csv
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.models import HousePriceModel
from src.data import clean_missing_values
from src.features import engineer_features


def predict_single(model: HousePriceModel, input_dict: dict, metadata: dict) -> dict:
    """
    Predict price for a single property dictionary.
    """
    df_raw = pd.DataFrame([input_dict])
    
    # Clean missing values using saved imputation
    df_clean, _ = clean_missing_values(df_raw, imputation_values=metadata.get('imputation_values', {}))
    
    # Feature engineering & column alignment
    training_cols = metadata.get('feature_columns', [])
    df_engineered, _ = engineer_features(df_clean, training_columns=training_cols)
    
    pred_price = float(model.predict(df_engineered)[0])
    
    # Calculate approx 95% confidence interval using RMSE from test metrics
    rmse = metadata.get('test_metrics', {}).get('rmse', 28000.0)
    lower_bound = max(0.0, pred_price - (1.96 * rmse))
    upper_bound = pred_price + (1.96 * rmse)
    
    return {
        'estimated_price': round(pred_price, 2),
        'confidence_interval_95': {
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2)
        },
        'formatted_price': f"${pred_price:,.2f}"
    }


def main():
    parser = argparse.ArgumentParser(description="House Price Prediction CLI Tool")
    parser.add_argument("--model", type=str, default="models/house_price_model.joblib", help="Path to trained model file.")
    parser.add_argument("--meta", type=str, default="models/metadata.json", help="Path to metadata JSON file.")
    parser.add_argument("--input", type=str, help="JSON string representing single house features.")
    parser.add_argument("--csv", type=str, help="Input CSV file for batch predictions.")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV path for batch predictions.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Trained model not found at {args.model}. Please run 'python train.py' first.")
        sys.exit(1)

    model = HousePriceModel.load(args.model)

    metadata = {}
    if os.path.exists(args.meta):
        with open(args.meta, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    if args.input:
        input_data = json.loads(args.input)
        result = predict_single(model, input_data, metadata)
        print("\n==========================================")
        print("🏠 House Price Prediction Result")
        print("==========================================")
        print(f"Estimated Market Value : {result['formatted_price']}")
        print(f"95% Confidence Range   : ${result['confidence_interval_95']['lower_bound']:,.2f} - ${result['confidence_interval_95']['upper_bound']:,.2f}")
        print("==========================================\n")

    elif args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: Input CSV file {args.csv} not found.")
            sys.exit(1)

        df_input = pd.read_csv(args.csv)
        df_clean, _ = clean_missing_values(df_input, imputation_values=metadata.get('imputation_values', {}))
        training_cols = metadata.get('feature_columns', [])
        df_eng, _ = engineer_features(df_clean, training_columns=training_cols)

        preds = model.predict(df_eng)
        df_input['Predicted_Price'] = np.round(preds, 2)
        df_input.to_csv(args.output, index=False)
        print(f"✅ Batch predictions for {len(df_input)} records saved to: {args.output}")

    else:
        # Default sample run
        sample = {
            "Area_SqFt": 2248.4,
            "Bedrooms": 3,
            "Bathrooms": 2,
            "Stories": 3,
            "House_Age_Years": 4.0,
            "Garage_Cars": 2,
            "Neighborhood": "Suburbs",
            "Furnishing": "Semi-Furnished",
            "Has_Pool": "No"
        }
        result = predict_single(model, sample, metadata)
        print("\n[Sample Prediction Run]")
        print(f"Input Features : {json.dumps(sample, indent=2)}")
        print(f"Predicted Price: {result['formatted_price']}\n")


if __name__ == "__main__":
    main()
