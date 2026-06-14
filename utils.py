import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42
N_SPLITS = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
ORIG_PATH = os.path.join(BASE_DIR, "Telco_customer_churn.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def load_data():
    print("Reading CSV source files from disk...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    orig = pd.read_csv(ORIG_PATH)
    
    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()
    orig.columns = orig.columns.str.strip()
    
    if "Churn Label" in orig.columns:
        orig = orig.rename(columns={"Churn Label": "Churn"})
    elif "Churn Value" in orig.columns:
        orig = orig.rename(columns={"Churn Value": "Churn"})
        
    if "CustomerID" in orig.columns:
        orig = orig.rename(columns={"CustomerID": "id"})
        
    feature_rename_map = {
        "Tenure Months": "tenure",
        "Monthly Charges": "MonthlyCharges",
        "Total Charges": "TotalCharges"
    }
    for ibm_col, kaggle_col in feature_rename_map.items():
        if ibm_col in orig.columns and kaggle_col not in orig.columns:
            orig = orig.rename(columns={ibm_col: kaggle_col})
            
    target_col = "Churn"
    id_col = "id"
    
    for name, df in [("train.csv", train), ("Telco_customer_churn.csv", orig)]:
        if target_col not in df.columns:
            print(f"CRITICAL CRASH: '{target_col}' column missing in {name}!")
            print(f"Available header list: {list(df.columns)}")
            raise KeyError(f"Could not resolve key '{target_col}' in file {name}")
            
        if df[target_col].dtype == 'O':
            df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})

    common_features = list(set(train.columns).intersection(set(orig.columns)))
    orig_cleaned = orig[common_features].copy()
    
    combined_train = pd.concat([train, orig_cleaned], axis=0).drop_duplicates().reset_index(drop=True)

    X = combined_train.drop(columns=[target_col, id_col], errors='ignore')
    y = combined_train[target_col]
    
    return X, y, test

def add_interaction_features(X, test):
    X = X.copy()
    test = test.copy()
    
    tenure_col = next((c for c in X.columns if c.lower() == 'tenure'), None)
    monthly_col = next((c for c in X.columns if c.lower() in ['monthlycharges', 'monthly_charges']), None)
    total_col = next((c for c in X.columns if c.lower() in ['totalcharges', 'total_charges']), None)
    
    for col in [tenure_col, monthly_col, total_col]:
        if col is not None:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            test[col] = pd.to_numeric(test[col], errors='coerce').fillna(0)
    
    if tenure_col and monthly_col:
        X['TotalCharges_Estimate'] = X[tenure_col] * X[monthly_col]
        test['TotalCharges_Estimate'] = test[tenure_col] * test[monthly_col]
        
    if monthly_col and total_col:
        X['Charges_Ratio'] = X[monthly_col] / (X[total_col] + 1)
        test['Charges_Ratio'] = test[monthly_col] / (test[total_col] + 1)
        
    return X, test

def encode_categorical(X, test):
    X = X.copy()
    test = test.copy()
    
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in cat_cols:
        X[col] = X[col].astype(str)
        test[col] = test[col].astype(str)
        
        joint_vocab = list(set(X[col].unique()).union(set(test[col].unique())))
        vocab_mapping = {token: index for index, token in enumerate(joint_vocab)}
        
        X[col] = X[col].map(vocab_mapping)
        test[col] = test[col].map(vocab_mapping)
        
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    test = test.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    test = test[X.columns]
    
    return X, test

def get_folds(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(skf.split(X, y))

def save_predictions(oof_predictions, test_predictions, test_ids, name):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    oof_df = pd.DataFrame({"oof": oof_predictions})
    oof_path = os.path.join(OUTPUT_DIR, f"oof_{name}.csv")
    oof_df.to_csv(oof_path, index=False)
    
    sub_df = pd.DataFrame({
        "id": test_ids,
        "Churn": test_predictions
    })
    sub_path = os.path.join(OUTPUT_DIR, f"submission_{name}.csv")
    sub_df.to_csv(sub_path, index=False)
    
    print(f"File export successful for [{name.upper()}]:")
    print(f"   -> OOF Saved: {oof_path}")
    print(f"   -> Sub Saved: {sub_path}")

def save_model_artifact(model_object, name):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    model_path = os.path.join(OUTPUT_DIR, f"model_{name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_object, f)
    print(f"   -> Saved model binary: {model_path}")

def load_model_artifact(name):
    model_path = os.path.join(OUTPUT_DIR, f"model_{name}.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)