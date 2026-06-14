import os
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
    RANDOM_STATE,
    OUTPUT_DIR
)

def train_pseudo_labeling():
    print("Loading datasets...")
    X, y, test = load_data()
    test_ids = test["id"].values  
    
    print("Engineering interaction features...")
    X, test = add_interaction_features(X, test)
    
    print("Encoding categorical columns...")
    X, test = encode_categorical(X, test)
    
    lgb_pred_path = f"{OUTPUT_DIR}/submission_lgb.csv"
    if not os.path.exists(lgb_pred_path):
        raise FileNotFoundError("Please run train_lgb.py first to generate test predictions!")
    
    print("Loading LightGBM predictions to find confident test rows...")
    sub_df = pd.read_csv(lgb_pred_path)
    test_preds = sub_df["Churn"].values

    HIGH_THRESH = 0.92  
    LOW_THRESH = 0.08   

    high_risk_mask = test_preds >= HIGH_THRESH
    low_risk_mask = test_preds <= LOW_THRESH
    pseudo_mask = high_risk_mask | low_risk_mask

    pseudo_X = test[pseudo_mask].copy()
    pseudo_y = np.where(test_preds[pseudo_mask] >= HIGH_THRESH, 1, 0)
    pseudo_y_series = pd.Series(pseudo_y, index=pseudo_X.index)

    print(f"-> Found {len(pseudo_X)} highly confident test rows to use as pseudo-labels.")
    print(f"   (Churn: {sum(pseudo_y)}, Stay: {len(pseudo_y) - sum(pseudo_y)})")

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
    
    print("\nStarting 10-Fold Pseudo-Labeling Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        X_train_large = pd.concat([X_train, pseudo_X], axis=0).reset_index(drop=True)
        y_train_large = pd.concat([y_train, pseudo_y_series], axis=0).reset_index(drop=True)
        
        model = lgb.LGBMClassifier(**lgb_params)
        
        model.fit(
            X_train_large, y_train_large,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
        test_predictions += model.predict_proba(test)[:, 1] / N_SPLITS
        
        fold_auc = roc_auc_score(y_val, oof_predictions[val_idx])
        print(f"-> Fold {fold + 1} Pseudo-LGBM AUC: {fold_auc:.5f}")

    save_model_artifact(model, "pseudo")
    save_predictions(oof_predictions, test_predictions, test_ids, name="pseudo")

if __name__ == "__main__":
    train_pseudo_labeling()