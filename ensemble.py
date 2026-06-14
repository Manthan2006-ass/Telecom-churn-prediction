import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from utils import OUTPUT_DIR, load_data

def run_ensembling():
    models = {
        "lgb": {
            "oof": f"{OUTPUT_DIR}/oof_lgb.csv",
            "sub": f"{OUTPUT_DIR}/submission_lgb.csv",
            "weight": 0.40,
        },
        "cat": {
            "oof": f"{OUTPUT_DIR}/oof_cat.csv",
            "sub": f"{OUTPUT_DIR}/submission_cat.csv",
            "weight": 0.30,
        },
        "pseudo": {
            "oof": f"{OUTPUT_DIR}/oof_pseudo.csv",
            "sub": f"{OUTPUT_DIR}/submission_pseudo.csv",
            "weight": 0.30,
        },
    }

    for name, files in models.items():
        if not os.path.exists(files["oof"]) or not os.path.exists(files["sub"]):
            raise FileNotFoundError(
                f"Missing files for model '{name}'. Please run train_{name}.py first!"
            )

    print("Analyzing validation shapes for broadcasting...")
    oof_data = {}
    min_rows = float("inf")

    for name, config in models.items():
        oof_data[name] = pd.read_csv(config["oof"])["oof"].values
        row_count = len(oof_data[name])
        print(f"   -> {name.upper()} OOF rows found: {row_count}")
        if row_count < min_rows:
            min_rows = row_count

    print("Loading master ground-truth target from pipeline engine...")
    _, y_true_series, _ = load_data()
    y_true = y_true_series.values

    print(f"   -> Ground truth y_true rows found: {len(y_true)}")

    if len(y_true) < min_rows:
        min_rows = len(y_true)

    print(f"Broadcast Safe Mode Activated. Aligning all vectors to {min_rows} rows.")

    y_true = y_true[:min_rows]

    print("Blending Out-of-Fold (OOF) predictions...")
    final_oof = np.zeros(min_rows)

    for name, config in models.items():
        aligned_oof_vals = oof_data[name][:min_rows]
        final_oof += aligned_oof_vals * config["weight"]

        individual_auc = roc_auc_score(y_true, aligned_oof_vals)
        print(f"-> Individual {name.upper()} OOF AUC: {individual_auc:.5f}")

    ensemble_auc = roc_auc_score(y_true, final_oof)
    print(f"FINAL BLENDED ENSEMBLE OOF AUC: {ensemble_auc:.5f}")

    print("Blending Final Test Submissions...")
    sample_sub = pd.read_csv(models["lgb"]["sub"])
    final_test_preds = np.zeros(len(sample_sub))

    for name, config in models.items():
        sub_df = pd.read_csv(config["sub"])
        final_test_preds += sub_df["Churn"].values * config["weight"]

    final_submission = pd.DataFrame(
        {"id": sample_sub["id"], "Churn": final_test_preds}
    )

    final_sub_path = f"{OUTPUT_DIR}/final_master_submission.csv"
    final_submission.to_csv(final_sub_path, index=False)

    print(f"Success! Absolute final file saved to: {final_sub_path}")

if __name__ == "__main__":
    run_ensembling()