"""
train_model.py
==============
Phase 6: Train LightGBM Binary Classifier on Pairwise Features.

Trains a single LightGBM model:
  - num_leaves = 31
  - learning_rate = 0.05
  - n_estimators = 200
  - early stopping using validation data
  - evaluation metric: binary_logloss / auc

Saves:
  - model.txt

Prints ONLY:
  - validation ROC-AUC
  - validation PR-AUC
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

FEATURE_COLS = [
    "exact_name_match",
    "exact_address_match",
    "name_token_jaccard",
    "address_token_jaccard",
    "name_jaro_winkler",
    "source_is_s2",
    "country_match",
]


def load_feature_df(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        if os.path.exists(path):
            return pd.read_parquet(path)
        csv_fallback = path.replace(".parquet", ".csv")
        if os.path.exists(csv_fallback):
            return pd.read_csv(csv_fallback)
    elif os.path.exists(path):
        return pd.read_csv(path)
    raise FileNotFoundError(f"Cannot find feature file at {path} or fallback CSV.")


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM on pairwise features")
    parser.add_argument("--train", default="features_train.parquet")
    parser.add_argument("--val", default="features_val.parquet")
    parser.add_argument("--out_model", default="model.txt")
    parser.add_argument("--early_stopping_rounds", type=int, default=20)
    args = parser.parse_args()

    # Verify lightgbm is installed
    try:
        import lightgbm as lgb
    except ImportError:
        print("\n[ERROR] LightGBM is not installed in your Python environment.")
        print("Install it with:")
        print("    pip install lightgbm\n")
        sys.exit(1)

    # 1. Load datasets
    df_train = load_feature_df(args.train)
    df_val = load_feature_df(args.val)

    X_train = df_train[FEATURE_COLS]
    y_train = df_train["label"]

    X_val = df_val[FEATURE_COLS]
    y_val = df_val["label"]

    # 2. Configure and train LightGBM
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False),
    ]

    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # 3. Save model
    gbm.save_model(args.out_model)

    # 4. Predict on validation set
    y_pred_val = gbm.predict(X_val, num_iteration=gbm.best_iteration)

    roc_auc = roc_auc_score(y_val, y_pred_val)
    pr_auc = average_precision_score(y_val, y_pred_val)

    # Print ONLY validation ROC-AUC and PR-AUC as requested
    print(f"validation ROC-AUC: {roc_auc:.4f}")
    print(f"validation PR-AUC: {pr_auc:.4f}")


if __name__ == "__main__":
    main()
