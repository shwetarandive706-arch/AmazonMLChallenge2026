"""
threshold_tune.py
=================
Phase 7: Threshold Tuning for Macro F0.5 on Validation Data.

Loads:
  - model.txt
  - features_val.parquet
  - dataset/train/train_ground_truth.tsv

Sweeps thresholds:
  0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90

Computes for each threshold:
  - Entity-level Macro F0.5
  - Entity-level Macro Precision
  - Entity-level Macro Recall
  - Pairwise Precision & Recall
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Set

FEATURE_COLS = [
    "exact_name_match",
    "exact_address_match",
    "name_token_jaccard",
    "address_token_jaccard",
    "name_jaro_winkler",
    "source_is_s2",
    "country_match",
]

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def load_val_features(path: str) -> pd.DataFrame:
    if path.endswith(".parquet") and os.path.exists(path):
        return pd.read_parquet(path)
    csv_fallback = path.replace(".parquet", ".csv")
    if os.path.exists(csv_fallback):
        return pd.read_csv(csv_fallback)
    raise FileNotFoundError(f"Cannot find validation features at {path} or {csv_fallback}")


def load_gt_map(gt_path: str) -> Dict[str, Set[str]]:
    df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    col_s1 = next((c for c in df.columns if "source1" in c.lower() or "s1" in c.lower()), None)
    col_m = next((c for c in df.columns if "match" in c.lower()), None)
    gt_map = {}
    for s1_id, match_str in zip(df[col_s1].astype(str), df[col_m].fillna("").astype(str)):
        if match_str:
            ids = {x.strip() for x in match_str.split(",") if x.strip()}
            gt_map[s1_id] = ids
    return gt_map


def compute_f_beta(p: float, r: float, beta: float = 0.5) -> float:
    b2 = beta ** 2
    if p + r == 0:
        return 0.0
    return (1.0 + b2) * (p * r) / (b2 * p + r)


def evaluate_entity_macro_f05(
    val_df: pd.DataFrame,
    scores: np.ndarray,
    gt_map: Dict[str, Set[str]],
    all_val_s1_ids: Set[str],
    threshold: float
) -> Dict[str, float]:
    """
    Evaluates entity-level Macro F0.5 across all validation S1 entities.
    Handles singletons (where true matches == 0) appropriately:
      - Empty true and empty pred -> P=1.0, R=1.0, F0.5=1.0 (correct singleton)
      - Empty true and non-empty pred -> P=0.0, R=0.0, F0.5=0.0 (false merge)
      - Non-empty true and empty pred -> P=0.0, R=0.0, F0.5=0.0 (missed match)
    """
    # Filter candidate predictions by threshold
    pass_mask = scores >= threshold
    passing_df = val_df[pass_mask]

    # Group passing predictions by S1 ID
    pred_map = passing_df.groupby("source1_entity_id")["target_entity_id"].apply(set).to_dict()

    f05_scores = []
    precisions = []
    recalls = []

    for s1_id in all_val_s1_ids:
        true_set = gt_map.get(s1_id, set())
        pred_set = pred_map.get(s1_id, set())

        if len(true_set) == 0 and len(pred_set) == 0:
            # Correct singleton
            p, r, f = 1.0, 1.0, 1.0
        elif len(true_set) == 0 and len(pred_set) > 0:
            # False merge into singleton
            p, r, f = 0.0, 0.0, 0.0
        elif len(true_set) > 0 and len(pred_set) == 0:
            # Missed all matches
            p, r, f = 0.0, 0.0, 0.0
        else:
            tp = len(true_set & pred_set)
            p = tp / len(pred_set)
            r = tp / len(true_set)
            f = compute_f_beta(p, r, beta=0.5)

        precisions.append(p)
        recalls.append(r)
        f05_scores.append(f)

    return {
        "macro_f05": float(np.mean(f05_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "passing_pairs": int(pass_mask.sum())
    }


def main():
    parser = argparse.ArgumentParser(description="Tune probability threshold for Macro F0.5")
    parser.add_argument("--model", default="model.txt")
    parser.add_argument("--val_features", default="features_val.parquet")
    parser.add_argument("--gt", default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--val_ids", default="splits/val_subset_0k_ids.txt")
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] lightgbm is not installed. Install with: pip install lightgbm")
        sys.exit(1)

    # 1. Load model and features
    print(f"Loading LightGBM model from {args.model} ...")
    gbm = lgb.Booster(model_file=args.model)

    print(f"Loading validation features from {args.val_features} ...")
    val_df = load_val_features(args.val_features)
    X_val = val_df[FEATURE_COLS]

    # 2. Score candidate pairs
    print(f"Scoring {len(val_df):,} validation candidate pairs ...")
    scores = gbm.predict(X_val)

    # 3. Load ground truth and validation S1 IDs
    print(f"Loading ground truth and validation entity IDs ...")
    gt_map = load_gt_map(args.gt)

    if os.path.exists(args.val_ids):
        with open(args.val_ids, "r", encoding="utf-8") as f:
            all_val_s1_ids = {line.strip() for line in f if line.strip()}
    else:
        all_val_s1_ids = set(val_df["source1_entity_id"].unique())

    print(f"Evaluating across {len(all_val_s1_ids):,} validation S1 entities.\n")

    # 4. Sweep thresholds
    print("=" * 72)
    print(f"{'Threshold':<11} {'Macro F0.5':>12} {'Precision':>12} {'Recall':>12} {'Pairs Kept':>14}")
    print("-" * 72)

    best_thresh = 0.50
    best_f05 = -1.0
    best_stats = {}

    table_rows = []
    for th in THRESHOLDS:
        res = evaluate_entity_macro_f05(val_df, scores, gt_map, all_val_s1_ids, th)
        table_rows.append((th, res))
        print(f"  {th:<9.2f} {res['macro_f05']:>12.4f} {res['macro_precision']:>12.4f} {res['macro_recall']:>12.4f} {res['passing_pairs']:>14,d}")
        if res["macro_f05"] > best_f05:
            best_f05 = res["macro_f05"]
            best_thresh = th
            best_stats = res

    print("=" * 72)
    print(f"\nBest Validation Threshold : {best_thresh:.2f}")
    print(f"Best Validation Macro F0.5: {best_f05:.4f}")
    print(f"  Macro Precision         : {best_stats['macro_precision']:.4f}")
    print(f"  Macro Recall            : {best_stats['macro_recall']:.4f}")


if __name__ == "__main__":
    main()
