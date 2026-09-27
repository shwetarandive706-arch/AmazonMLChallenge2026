"""
phase3_split.py  --  Phase 3: Entity-Level Train/Val Split
===========================================================

Splits S1 entities into train / val using stratified sampling over
match-count buckets.  Derives which ground-truth rows belong to each
split by following train_ground_truth.tsv S1-entity links.

Bucket definitions  (match-count = # GT rows for an S1 entity):
  0     -> singleton   (no matches in ground truth)
  1     -> exactly-1
  2-5   -> bulk
  6+    -> long-tail

Usage
-----
  python phase3_split.py \\
      --s1       dataset/train/train_source1.tsv \\
      --gt       dataset/train/train_ground_truth.tsv \\
      --out_dir  splits/

Optional flags
--------------
  --val_frac    0.10          fraction for val  (default 0.10 -> 90/10)
  --seed        42
  --country_col country       S1 column for secondary country stratification
                              (set to empty string "" to disable)

Outputs (all written to --out_dir)
------------------------------------
  train_s1_ids.txt    one entity_id per line
  val_s1_ids.txt
  train_gt_rows.tsv   GT rows whose S1 entity is in train
  val_gt_rows.tsv     GT rows whose S1 entity is in val
  split_stats.txt     bucket-level verification table
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------

BUCKET_ORDER = ["singleton", "exactly-1", "2-5", "6+"]


def assign_bucket(match_count: int) -> str:
    if match_count == 0:
        return "singleton"
    if match_count == 1:
        return "exactly-1"
    if match_count <= 5:
        return "2-5"
    return "6+"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _sniff_sep(path: str) -> str:
    """Return '\t' for .tsv, ',' otherwise."""
    return "\t" if path.lower().endswith(".tsv") else ","


def load_s1(path: str) -> pd.DataFrame:
    """Load S1 table; normalise entity_id column name."""
    sep = _sniff_sep(path)
    df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()

    # Accept common column name variants
    for cand in ["entity_id", "id", "s1_entity_id", "s1_id"]:
        if cand in df.columns:
            if cand != "entity_id":
                df = df.rename(columns={cand: "entity_id"})
            break
    else:
        raise ValueError(
            f"Cannot find entity_id column in S1.\n"
            f"Columns found: {df.columns.tolist()}"
        )
    return df


def load_gt(path: str) -> pd.DataFrame:
    """Load ground-truth file; normalise column names."""
    sep = _sniff_sep(path)
    df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()

    rename = {}
    for col in df.columns:
        lc = col.lower()
        if "source1" in lc or ("s1" in lc and "entity" in lc):
            rename[col] = "source1_entity_id"
        elif "match" in lc:
            rename[col] = "matched_entity_ids"
        elif "s2" in lc and ("record" in lc or "id" in lc):
            rename[col] = "s2_record_id"
        elif "s3" in lc and ("record" in lc or "id" in lc):
            rename[col] = "s3_record_id"
    df = df.rename(columns=rename)

    if "source1_entity_id" not in df.columns:
        raise ValueError(
            f"Ground-truth file missing source1_entity_id column.\n"
            f"Columns found: {df.columns.tolist()}"
        )
    return df


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------

def compute_match_counts(s1_ids: pd.Series, gt: pd.DataFrame) -> pd.Series:
    """Number of matched entities per S1 entity (0 = singleton)."""
    col_s1 = "source1_entity_id"
    col_match = "matched_entity_ids"

    if col_match in gt.columns:
        # Competition format: matched_entity_ids is comma-separated list
        match_vals = gt[col_match].fillna("")
        counts_arr = [
            sum(1 for x in str(v).split(",") if x.strip()) if v else 0
            for v in match_vals
        ]
        s1_series = gt[col_s1].astype(str)
        if s1_series.is_unique:
            s1_to_count = dict(zip(s1_series, counts_arr))
        else:
            temp_df = pd.DataFrame({"s1": s1_series, "c": counts_arr})
            s1_to_count = temp_df.groupby("s1")["c"].sum().to_dict()
        return s1_ids.astype(str).map(s1_to_count).fillna(0).astype(int)
    else:
        # Fallback if row-per-match format
        counts = gt[col_s1].value_counts()
        return s1_ids.astype(str).map(counts).fillna(0).astype(int)


def build_strata(s1_df: pd.DataFrame, gt: pd.DataFrame,
                 country_col: str) -> pd.DataFrame:
    """
    Return s1_df augmented with:
      _match_count  int
      _bucket       str  one of BUCKET_ORDER
      _stratum      str  bucket  (or  country|bucket  if country_col is set)
    """
    s1_df = s1_df.copy()
    s1_df["_match_count"] = compute_match_counts(s1_df["entity_id"], gt)
    s1_df["_bucket"]      = s1_df["_match_count"].apply(assign_bucket)

    if country_col and country_col in s1_df.columns:
        s1_df["_stratum"] = (
            s1_df[country_col].fillna("Unknown") + "|" + s1_df["_bucket"]
        )
    else:
        s1_df["_stratum"] = s1_df["_bucket"]

    return s1_df


def stratified_split(s1_aug: pd.DataFrame, val_frac: float, seed: int):
    """
    Stratified 1-level split returning (train_ids, val_ids) as numpy arrays.

    Falls back gracefully when a stratum has < 2 members (forces them to train).
    """
    strata     = s1_aug["_stratum"].values
    entity_ids = s1_aug["entity_id"].values

    try:
        train_ids, val_ids = train_test_split(
            entity_ids,
            test_size=val_frac,
            stratify=strata,
            random_state=seed,
        )
        return train_ids, val_ids

    except ValueError as exc:
        print(
            f"[WARN] Full stratified split failed ({exc}).\n"
            "       Falling back to per-stratum split; "
            "singletons in small strata go to train."
        )
        rng = np.random.default_rng(seed)
        train_list, val_list = [], []

        for _label, grp in s1_aug.groupby("_stratum"):
            ids = grp["entity_id"].values
            if len(ids) < 2:
                train_list.extend(ids)
                continue
            n_val = max(1, round(len(ids) * val_frac))
            n_val = min(n_val, len(ids) - 1)
            chosen = rng.choice(len(ids), size=n_val, replace=False)
            mask = np.zeros(len(ids), dtype=bool)
            mask[chosen] = True
            val_list.extend(ids[mask])
            train_list.extend(ids[~mask])

        return np.array(train_list), np.array(val_list)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def bucket_stats(entity_ids, s1_aug: pd.DataFrame):
    """Return ({bucket: (count, pct)}, total)."""
    id_set = set(entity_ids)
    sub    = s1_aug[s1_aug["entity_id"].isin(id_set)]
    total  = len(sub)
    vc     = sub["_bucket"].value_counts()
    result = {}
    for b in BUCKET_ORDER:
        n = int(vc.get(b, 0))
        result[b] = (n, 100.0 * n / total if total else 0.0)
    return result, total


def format_stats_table(train_ids, val_ids, s1_aug: pd.DataFrame) -> str:
    train_bs, n_train = bucket_stats(train_ids, s1_aug)
    val_bs,   n_val   = bucket_stats(val_ids,   s1_aug)
    total = n_train + n_val

    lines = [
        "=" * 72,
        "Phase 3 Train/Val Split -- Verification Statistics",
        "=" * 72,
        f"  Total S1 entities  : {total:>10,}",
        f"  Train S1 entities  : {n_train:>10,}  ({100*n_train/total:.2f}%)",
        f"  Val   S1 entities  : {n_val:>10,}  ({100*n_val/total:.2f}%)",
        "",
        f"{'Bucket':<12} {'Train N':>10} {'Train%':>8} {'Val N':>8} {'Val%':>8}",
        "-" * 54,
    ]
    for b in BUCKET_ORDER:
        tn, tp = train_bs[b]
        vn, vp = val_bs[b]
        lines.append(f"{b:<12} {tn:>10,} {tp:>7.2f}%  {vn:>8,} {vp:>7.2f}%")
    lines += ["=" * 72]
    return "\n".join(lines)


def country_breakdown(train_ids, val_ids, s1_aug: pd.DataFrame,
                      country_col: str) -> str:
    if country_col not in s1_aug.columns:
        return ""
    train_set = set(train_ids)
    val_set   = set(val_ids)
    lines = ["\nCountry breakdown:"]
    for country, grp in s1_aug.groupby(country_col):
        eids  = set(grp["entity_id"].values)
        n_t   = len(train_set & eids)
        n_v   = len(val_set   & eids)
        lines.append(
            f"  {str(country):<20}  total={n_t+n_v:>8,}  "
            f"train={n_t:>8,}  val={n_v:>8,}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: entity-level stratified train/val split"
    )
    parser.add_argument("--s1",          required=True,
                        help="Path to S1 CSV/TSV (entity master table)")
    parser.add_argument("--gt",          required=True,
                        help="Path to train_ground_truth.tsv")
    parser.add_argument("--out_dir",     required=True,
                        help="Directory to write output files")
    parser.add_argument("--val_frac",    type=float, default=0.10,
                        help="Val fraction (default 0.10 => 90/10 split)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--country_col", default="country",
                        help="S1 column for secondary country stratification. "
                             "Set to '' to disable. Default: 'country'")
    args = parser.parse_args()

    country_col = args.country_col.strip() if args.country_col else ""

    # 1. Load ----------------------------------------------------------------
    print(f"[1/6] Loading S1  : {args.s1}")
    s1 = load_s1(args.s1)
    print(f"      {len(s1):,} S1 entities.")

    print(f"[2/6] Loading GT  : {args.gt}")
    gt = load_gt(args.gt)
    print(f"      {len(gt):,} GT rows.")

    # 2. Build strata --------------------------------------------------------
    print("[3/6] Building strata (match-count buckets) ...")
    s1_aug = build_strata(s1, gt, country_col)

    total_s1 = len(s1_aug)
    print("      Bucket distribution (full corpus):")
    vc = s1_aug["_bucket"].value_counts()
    for b in BUCKET_ORDER:
        n = vc.get(b, 0)
        print(f"        {b:<12}  {n:>10,}  ({100*n/total_s1:.2f}%)")

    # 3. Split ---------------------------------------------------------------
    print(f"[4/6] Stratified split (val_frac={args.val_frac}, seed={args.seed}) ...")
    train_ids, val_ids = stratified_split(s1_aug, args.val_frac, args.seed)
    print(f"      Train: {len(train_ids):,}  |  Val: {len(val_ids):,}")

    # Sanity assertions
    overlap = len(set(train_ids) & set(val_ids))
    assert overlap == 0, f"BUG: {overlap} IDs appear in both splits!"
    total_check = len(train_ids) + len(val_ids)
    assert total_check == total_s1, (
        f"BUG: train({len(train_ids)}) + val({len(val_ids)}) = {total_check} "
        f"!= total_s1({total_s1})"
    )
    print("      Sanity checks passed (no overlap, counts match).")

    # 4. Verify --------------------------------------------------------------
    print("[5/6] Computing verification statistics ...")
    stats_text = format_stats_table(train_ids, val_ids, s1_aug)
    print("\n" + stats_text)

    cc_text = ""
    if country_col:
        cc_text = country_breakdown(train_ids, val_ids, s1_aug, country_col)
        if cc_text:
            print(cc_text)

    # 5. Write outputs -------------------------------------------------------
    print(f"\n[6/6] Writing outputs to {args.out_dir} ...")
    os.makedirs(args.out_dir, exist_ok=True)

    # Entity ID lists
    with open(os.path.join(args.out_dir, "train_s1_ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_ids) + "\n")
    with open(os.path.join(args.out_dir, "val_s1_ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_ids) + "\n")

    # GT rows: follow S1 entity assignment
    # val_gt_rows  = rows where S1 entity is in val
    # train_gt_rows = everything else (S1 entity in train, OR unmatched)
    val_id_set = set(val_ids)
    col_s1 = "source1_entity_id" if "source1_entity_id" in gt.columns else "s1_entity_id"
    gt["_split"] = gt[col_s1].map(
        lambda eid: "val" if eid in val_id_set else "train"
    )
    train_gt = gt[gt["_split"] == "train"].drop(columns=["_split"])
    val_gt   = gt[gt["_split"] == "val"].drop(columns=["_split"])

    train_gt.to_csv(
        os.path.join(args.out_dir, "train_gt_rows.tsv"), sep="\t", index=False
    )
    val_gt.to_csv(
        os.path.join(args.out_dir, "val_gt_rows.tsv"), sep="\t", index=False
    )

    # Stats file
    stats_path = os.path.join(args.out_dir, "split_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(stats_text + "\n")
        if cc_text:
            f.write(cc_text + "\n")

    # Summary
    print(f"\n  [OK] train_s1_ids.txt   : {len(train_ids):,} entity IDs")
    print(f"  [OK] val_s1_ids.txt     : {len(val_ids):,} entity IDs")
    print(f"  [OK] train_gt_rows.tsv  : {len(train_gt):,} GT rows")
    print(f"  [OK] val_gt_rows.tsv    : {len(val_gt):,} GT rows")
    print(f"  [OK] split_stats.txt    : {stats_path}")
    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()

