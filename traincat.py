import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from utils import (
    load_data, 
    add_interaction_features, 
    encode_categorical, 
    get_folds, 
    save_predictions,
    save_model_artifact,
    N_SPLITS,
    RANDOM_STATE
)

def train_catboost():
    print("Loading datasets...")
    X, y, test = load_data()
    test_ids = test["id"].values  
    
    print("Engineering interaction features...")
    X, test = add_interaction_features(X, test)
    
    print("Encoding categorical columns...")
    X, test = encode_categorical(X, test)
    
    folds = get_folds(X, y)
    
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(test))
    
    cat_params = {
        "iterations": 1500,
        "learning_rate": 0.03,
        "eval_metric": "AUC",
        "random_seed": RANDOM_STATE,
        "verbose": 100
    }
    
    print("\nStarting 10-Fold Cross-Validation for CatBoost...")
    for fold, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = CatBoostClassifier(**cat_params)
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        
        oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
        test_predictions += model.predict_proba(test)[:, 1] / N_SPLITS
        
        fold_auc = roc_auc_score(y_val, oof_predictions[val_idx])
        print(f"-> Fold {fold + 1} CatBoost AUC: {fold_auc:.5f}")

    save_model_artifact(model, "cat")
    save_predictions(oof_predictions, test_predictions, test_ids, name="cat")

if __name__ == "__main__":
    train_catboost()