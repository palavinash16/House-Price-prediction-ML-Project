"""
Command Line Script to train, evaluate, and export House Price Prediction ML models.
Usage:
    python train.py --data House_Price_Data.csv --model_dir models
"""

import os
import sys
import json
import argparse
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


from src.data import load_dataset, clean_missing_values, preprocess_data
from src.features import engineer_features
from src.models import train_and_evaluate_all, export_weights_json, HousePriceModel
from src.evaluation import evaluate_residuals, get_feature_importances


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate House Price Prediction models.")
    parser.add_argument("--data", type=str, default="House_Price_Data.csv", help="Path to input dataset CSV.")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save trained model artifacts.")
    args = parser.parse_args()

    print("==================================================")
    print("🏠 House Price Prediction - Training Pipeline")
    print("==================================================")

    # 1. Load dataset
    print(f"--> Loading raw dataset from: {args.data}")
    df_raw = load_dataset(args.data)
    print(f"    Dataset loaded with shape: {df_raw.shape}")

    # 2. Clean missing values & feature engineering
    print("--> Preprocessing & Feature Engineering...")
    df_clean, imp_dict = clean_missing_values(df_raw)
    
    X_raw = df_clean.drop(columns=['Price'])
    y_raw = df_clean['Price']

    X_engineered, feature_cols = engineer_features(X_raw)
    print(f"    Generated {len(feature_cols)} features: {feature_cols}")

    # 3. Train/Test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y_raw, test_size=0.2, random_state=42
    )
    print(f"    Train sample count: {len(X_train)} | Test sample count: {len(X_test)}")

    # 4. Train and benchmark all models
    print("\n--> Training & Benchmarking Models...")
    models_dict, metrics_summary, best_name = train_and_evaluate_all(X_train, y_train, X_test, y_test)

    print("\n==========================================================================================")
    print(f"{'Model Architecture':<22} | {'Test R²':<10} | {'RMSE ($)':<12} | {'MAE ($)':<12} | {'MAPE (%)':<10}")
    print("==========================================================================================")
    for m_type, m_metrics in metrics_summary.items():
        is_best = " (BEST)" if m_type == best_name else ""
        print(f"{m_type + is_best:<22} | {m_metrics['r2']:<10.4f} | ${m_metrics['rmse']:<11.2f} | ${m_metrics['mae']:<11.2f} | {m_metrics['mape']:<9.2f}%")
    print("==========================================================================================")

    best_model = models_dict[best_name]
    best_metrics = metrics_summary[best_name]
    print(f"\n🌟 Selected Best Performing Model: {best_name.upper()} (R² = {best_metrics['r2']:.4f})")

    # 5. Export model artifacts
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "house_price_model.joblib")
    meta_path = os.path.join(args.model_dir, "metadata.json")
    weights_path = os.path.join(args.model_dir, "weights.json")

    best_model.save(model_path)
    print(f"--> Saved best model pipeline to: {model_path}")

    # Residuals analysis
    residuals_info = evaluate_residuals(y_test, best_model.predict(X_test))

    # Save metadata
    metadata = {
        'best_model': best_name,
        'feature_columns': feature_cols,
        'imputation_values': imp_dict,
        'test_metrics': best_metrics,
        'residuals_info': residuals_info,
        'all_metrics': metrics_summary
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"--> Saved training metadata to: {meta_path}")

    # Export weights for JS frontend
    export_weights_json(best_model, weights_path, best_metrics)
    print(f"--> Exported web predictor weights to: {weights_path}")

    print("\n✅ Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
