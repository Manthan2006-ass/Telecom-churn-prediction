import numpy as np
import pandas as pd
import lightgbm as lgb
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

def train_lightgbm():
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
    
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "random_state": RANDOM_STATE,
        "n_estimators": 1500,
        "verbose": -1
    }
    
    print("\nStarting 10-Fold Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**lgb_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
        test_predictions += model.predict_proba(test)[:, 1] / N_SPLITS
        
        fold_auc = roc_auc_score(y_val, oof_predictions[val_idx])
        print(f"-> Fold {fold + 1} AUC: {fold_auc:.5f}")

    save_model_artifact(model, "lgb")
    save_predictions(oof_predictions, test_predictions, test_ids, name="lgb")

if __name__ == "__main__":
    train_lightgbm()