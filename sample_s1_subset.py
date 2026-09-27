"""
sample_s1_subset.py
===================
Deterministic, stratified sampling of S1 entity IDs for fast training & validation.

Samples:
  - 25,000 S1 IDs from splits/train_s1_ids.txt -> splits/train_subset_25k_ids.txt
  -  5,000 S1 IDs from splits/val_s1_ids.txt   -> splits/val_subset_5k_ids.txt

Preserves:
  - India vs. US country distribution (~40% / ~60%)
  - 4-bucket match-count distribution:
      singleton  (~5.58%)
      exactly-1  (~5.40%)
      2-5        (~77.58%)
      6+         (~11.43%)
  - Zero overlap between train and validation subsets

Usage:
  python sample_s1_subset.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Re-use vetted Phase 3 logic
try:
    from phase3_split import load_s1, load_gt, build_strata, BUCKET_ORDER
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    from phase3_split import load_s1, load_gt, build_strata, BUCKET_ORDER


def load_id_set(path: str) -> set:
    """Read entity IDs from text file (one per line, whitespace stripped)."""
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def print_distribution_table(df_subset: pd.DataFrame, title: str, original_total: int):
    """Print formatting verification table for a sampled subset."""
    total = len(df_subset)
    print("=" * 72)
    print(f"{title} (Total: {total:,} IDs, sampled from {original_total:,})")
    print("=" * 72)

    # 1. Country breakdown
    print("Country Distribution:")
    country_counts = df_subset["country"].value_counts()
    for country in sorted(country_counts.index):
        cnt = country_counts[country]
        pct = (cnt / total) * 100
        print(f"  {country:<15} : {cnt:>6,}  ({pct:>6.2f}%)")

    # 2. Bucket breakdown
    print("\nMatch-Count Bucket Distribution:")
    bucket_counts = df_subset["_bucket"].value_counts()
    for b in BUCKET_ORDER:
        cnt = bucket_counts.get(b, 0)
        pct = (cnt / total) * 100
        print(f"  {b:<15} : {cnt:>6,}  ({pct:>6.2f}%)")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Sample stratified train/val subsets of S1 entity IDs"
    )
    parser.add_argument(
        "--s1", default="dataset/train/train_source1.tsv",
        help="Path to train_source1.tsv"
    )
    parser.add_argument(
        "--gt", default="dataset/train/train_ground_truth.tsv",
        help="Path to train_ground_truth.tsv"
    )
    parser.add_argument(
        "--train_ids", default="splits/train_s1_ids.txt",
        help="Path to full train_s1_ids.txt"
    )
    parser.add_argument(
        "--val_ids", default="splits/val_s1_ids.txt",
        help="Path to full val_s1_ids.txt"
    )
    parser.add_argument(
        "--out_dir", default="splits",
        help="Directory to save sampled subsets (default: splits)"
    )
    parser.add_argument(
        "--train_n", type=int, default=25000,
        help="Number of train S1 IDs to sample (default: 25000)"
    )
    parser.add_argument(
        "--val_n", type=int, default=5000,
        help="Number of validation S1 IDs to sample (default: 5000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # 1. Load S1 and GT to build strata ----------------------------------------
    print(f"[1/5] Loading S1 master table: {args.s1} ...")
    s1 = load_s1(args.s1)
    print(f"      Loaded {len(s1):,} S1 entities.")

    print(f"[2/5] Loading Ground Truth: {args.gt} ...")
    gt = load_gt(args.gt)
    print(f"      Loaded {len(gt):,} GT rows.")

    print(f"[3/5] Building stratification strata (country x match bucket) ...")
    s1_aug = build_strata(s1, gt, country_col="country")

    # 2. Filter pools using existing split files -------------------------------
    print(f"[4/5] Reading existing split ID lists ...")
    full_train_ids = load_id_set(args.train_ids)
    full_val_ids = load_id_set(args.val_ids)
    print(f"      Full Train pool : {len(full_train_ids):,} IDs")
    print(f"      Full Val pool   : {len(full_val_ids):,} IDs")

    # Filter augmented dataframe into train and val pools
    train_pool = s1_aug[s1_aug["entity_id"].isin(full_train_ids)].copy()
    val_pool = s1_aug[s1_aug["entity_id"].isin(full_val_ids)].copy()

    # 3. Stratified sampling with seed=42 --------------------------------------
    print(f"[5/5] Performing deterministic stratified sampling (seed={args.seed}) ...")
    
    # Stratified sample from train pool
    train_subset, _ = train_test_split(
        train_pool,
        train_size=args.train_n,
        stratify=train_pool["_stratum"],
        random_state=args.seed
    )

    # Stratified sample from val pool
    val_subset, _ = train_test_split(
        val_pool,
        train_size=args.val_n,
        stratify=val_pool["_stratum"],
        random_state=args.seed
    )

    # 4. Assertions & verification ---------------------------------------------
    sampled_train_ids = set(train_subset["entity_id"])
    sampled_val_ids = set(val_subset["entity_id"])

    assert len(sampled_train_ids) == args.train_n, f"Expected {args.train_n} train IDs, got {len(sampled_train_ids)}"
    assert len(sampled_val_ids) == args.val_n, f"Expected {args.val_n} val IDs, got {len(sampled_val_ids)}"

    overlap = sampled_train_ids & sampled_val_ids
    assert len(overlap) == 0, f"CRITICAL: Overlap of {len(overlap)} IDs between train and val subsets!"

    # 5. Write output text files ----------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out_train_path = os.path.join(args.out_dir, f"train_subset_{args.train_n // 1000}k_ids.txt")
    out_val_path = os.path.join(args.out_dir, f"val_subset_{args.val_n // 1000}k_ids.txt")

    with open(out_train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(sampled_train_ids)) + "\n")

    with open(out_val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(sampled_val_ids)) + "\n")

    print("\n" * 1)
    print_distribution_table(train_subset, f"TRAIN SUBSET ({args.train_n // 1000}k)", len(full_train_ids))
    print()
    print_distribution_table(val_subset, f"VAL SUBSET ({args.val_n // 1000}k)", len(full_val_ids))

    print(f"\nVerification:")
    print(f"  [OK] Train subset count : {len(sampled_train_ids):,} IDs -> {out_train_path}")
    print(f"  [OK] Val subset count   : {len(sampled_val_ids):,} IDs -> {out_val_path}")
    print(f"  [OK] Overlap check      : 0 overlapping IDs between train and val subsets")
    print(f"  [OK] Deterministic seed : {args.seed}")
    print("\nSubsets successfully generated.")


if __name__ == "__main__":
    main()
