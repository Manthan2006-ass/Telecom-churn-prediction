# Telecom Customer Churn Prediction Pipeline

This project is a Telco churn prediction pipeline built with Python and Streamlit. It includes data preprocessing, LightGBM and CatBoost model training, pseudo-labeling, ensembling, and a Streamlit inference app.

## Repository Structure

- `app.py` - Streamlit web app for single-customer inference and batch CSV scoring.
- `ensemble.py` - Blends out-of-fold and submission predictions from multiple models.
- `traincat.py` - Trains a CatBoost churn model and saves predictions.
- `trainlgb.py` - Trains a LightGBM churn model and saves predictions.
- `trainpsuedo.py` - Trains a LightGBM model using pseudo-labeled test examples.
- `utils.py` - Common utilities for loading data, feature engineering, encoding, model persistence, and submission export.
- `outputs/` - Generated model artifacts and CSV outputs.
- sample_churn_template.csv` - example batch input format.

## Data requirements

This repository does not include the full dataset files. To run training, you must provide:

- `train.csv`
- `test.csv`
- `Telco_customer_churn.csv`

Place these files in the project root or a local data folder before running the scripts.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Train models

```bash
python trainlgb.py
python traincat.py
python trainpsuedo.py
```

### Create blend

```bash
python ensemble.py
```

### Run the Streamlit app

```bash
streamlit run app.py
```

## Output files

The `outputs/` directory contains:

- `model_lgb.pkl`, `model_cat.pkl`, `model_pseudo.pkl` - saved models
- `oof_lgb.csv`, `oof_cat.csv`, `oof_pseudo.csv` - out-of-fold prediction scores
- `submission_lgb.csv`, `submission_cat.csv`, `submission_pseudo.csv` - test predictions
- `final_master_submission.csv` - blended final submission

## Notes

- `app.py` expects saved model artifacts in `outputs/` to use the real inference pipeline. If those files are missing, it falls back to a heuristic risk estimate.
- `trainpsuedo.py` depends on `outputs/submission_lgb.csv` from `trainlgb.py`.
- Ensure the input CSV for batch scoring includes required columns such as `gender`, `tenure`, `MonthlyCharges`, and `TotalCharges`.
