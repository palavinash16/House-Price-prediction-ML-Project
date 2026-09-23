# 🏠 House Price Prediction - End-to-End Machine Learning System

**Author**: Avinash Kumar Pal  
**Domain**: Real Estate Analytics & Predictive Machine Learning  

---

## 📌 Project Overview
Predicting housing market valuation is a foundational problem in predictive real estate analytics. This project is a **standalone, production-grade Machine Learning system**.

It features automated missing value imputation, domain feature engineering (`SqFt_per_Bedroom`, `Bath_Bed_Ratio`, `Luxury_Score`), regularized linear and ensemble tree model benchmarks (`LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `RandomForestRegressor`, `GradientBoostingRegressor`), automated CLI tools (`train.py`, `predict.py`), FastAPI REST server backend (`app_api.py`), unit tests (`pytest`), and a modern **Glassmorphism Interactive Web Dashboard**.

---

## 📊 Feature Checklist & Implementation

| Feature Area | Implementation Details | Status |
| :--- | :--- | :---: |
| **EDA & Cleaning** | Inspected 1,500 property records, imputed missing values (`Area_SqFt`, `Age`) | ✅ Complete |
| **Feature Engineering** | Created `SqFt_per_Bedroom`, `Bath_Bed_Ratio`, `Luxury_Score`, and `Neighborhood` dummies | ✅ Complete |
| **Categorical Encoding** | One-Hot Encoded `Neighborhood`, `Furnishing`, and `Has_Pool` categories | ✅ Complete |
| **Correlation Analysis** | Triangular seaborn correlation matrix identifying top price drivers | ✅ Complete |
| **Train/Test Split** | 80/20 train/test split preserving feature distribution | ✅ Complete |
| **Multi-Model Benchmark** | Evaluated Linear, Ridge, Lasso, ElasticNet, Random Forest & Gradient Boosting | ✅ Complete |
| **Metrics Evaluation** | Computed Test $R^2 \ge 0.8871$, RMSE ($28,389), MAE ($22,049), MAPE (4.63%) | ✅ Complete |
| **Visual Diagnostics** | Actual vs Predicted scatter plots & residual homoscedasticity distributions | ✅ Complete |
| **CLI Automation Tools** | `train.py` for pipeline export & `predict.py` for single/batch CLI inference | ✅ Complete |
| **FastAPI REST Service** | Async API backend (`/predict`, `/health`, `/metrics`) | ✅ Complete |
| **Unit Test Suite** | Automated `pytest` coverage for data, features, and model pipelines | ✅ Complete |
| **Interactive Web App** | Modern Dark Glassmorphism Web App with live slider predictor | ✅ Complete |

---

## 🛠️ Tech Stack & Dependencies

- **Programming Language**: Python 3.12+
- **Machine Learning**: `scikit-learn`, `numpy`, `pandas`
- **Data Visualization**: `matplotlib`, `seaborn`
- **REST API Backend**: `fastapi`, `uvicorn`, `pydantic`
- **Testing**: `pytest`, `httpx`
- **Frontend Web Dashboard**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (ES6+), FontAwesome 6, Google Fonts (Outfit, Inter, Fira Code)
- **Model Serialization**: `joblib`, `json`

---

## 🤖 Model Performance Summary Matrix

| Model Architecture | Test $R^2$ | Test RMSE ($) | Test MAE ($) | MAPE (%) | Model Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Lasso Regression (L1)** | **0.8871** | **$28,389.31** | **$22,049.71** | **4.63%** | 🌟 **SELECTED BEST MODEL** |
| **Linear Regression (OLS)** | **0.8869** | **$28,411.71** | **$22,027.00** | **4.63%** | Baseline Linear Model |
| **Ridge Regression (L2)** | **0.8867** | **$28,432.10** | **$22,080.81** | **4.64%** | Multi-Collinearity Penalty |
| **Gradient Boosting** | **0.8464** | **$33,107.60** | **$25,903.14** | **5.43%** | Non-Linear Ensemble |
| **Random Forest** | **0.8053** | **$37,275.51** | **$29,059.48** | **6.00%** | Bagged Decision Trees |
| **ElasticNet** | **0.8036** | **$37,444.14** | **$29,526.85** | **6.14%** | Combined L1/L2 Penalty |

---

## 📂 Project Structure

```text
HousePricePrediction/
├── House_Price_Data.csv           # Original dataset (1,500 property records)
├── requirements.txt               # Python package dependencies
├── main.ipynb                     # Executed Jupyter Notebook with EDA, Models & Residual Plots
├── README.md                      # Comprehensive project documentation
│
├── src/                           # Modular Python ML Package
│   ├── __init__.py
│   ├── data.py                    # Dataset loading, median imputation, train/test split
│   ├── features.py                # Domain feature engineering & One-Hot Encoding
│   ├── models.py                  # Model training, scaling, evaluation & serialization
│   └── evaluation.py              # Regression metrics (R², RMSE, MAE, MAPE) & residual analysis
│
├── models/                        # Saved Model Artifacts
│   ├── house_price_model.joblib   # Serialized best model pipeline
│   ├── metadata.json              # Trained metadata, feature columns & metrics
│   └── weights.json               # Serialized weights for fast client JS inference
│
├── tests/                         # Pytest Unit Test Suite
│   ├── test_data.py               # Imputation & split tests
│   ├── test_features.py           # Feature transformer integrity tests
│   └── test_models.py             # Inference output & artifact loading tests
│
├── train.py                       # CLI script to train models and export artifacts
├── predict.py                     # CLI script to run predictions on single inputs or CSV files
├── app_api.py                     # FastAPI REST server (/predict, /health, /metrics)
│
└── web/                           # Modern Interactive Web App Dashboard
    ├── index.html                 # Full interactive dashboard UI
    ├── styles.css                 # Custom glassmorphism design system & micro-animations
    └── app.js                     # Live predictor engine, chart rendering & API client
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train Models & Export Artifacts
Run the training pipeline to benchmark models and export trained artifacts:
```bash
python train.py
```

### 3. Run Predictions via CLI
Predict house price for a single property:
```bash
python predict.py --input '{"Area_SqFt": 2250, "Bedrooms": 3, "Bathrooms": 2, "Stories": 2, "House_Age_Years": 8, "Garage_Cars": 2, "Neighborhood": "Suburbs", "Furnishing": "Semi-Furnished", "Has_Pool": "No"}'
```
Or run batch predictions on a CSV file:
```bash
python predict.py --csv House_Price_Data.csv --output predictions.csv
```

### 4. Run Unit Test Suite
Verify pipeline integrity using Pytest:
```bash
python -m pytest tests/
```

### 5. Launch FastAPI REST Service
Start the REST API server:
```bash
uvicorn app_api:app --reload --port 8000
```
Access interactive API documentation at: `http://localhost:8000/docs`

### 6. Launch Web Dashboard
Open `web/index.html` in any web browser to use the interactive house price simulator, model comparison cards, and feature importance visualizer:
```bash
python -m http.server 3000 --directory web
```
Visit `http://localhost:3000` in your browser.
