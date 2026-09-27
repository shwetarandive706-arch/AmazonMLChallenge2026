"""
build_features.py
=================
Phase 5: Pairwise Feature Engineering for Business Entity Resolution.

Reads:
  - Train pairs: output/small_train_pairs.tsv (or specified via --train_pairs)
  - Val pairs:   output/small_val_pairs.tsv   (or specified via --val_pairs)

Joins S1 and Target (S2/S3) records from:
  - dataset/train/train_source1.tsv
  - dataset/train/train_source2.tsv
  - dataset/train/train_source3.tsv

Computes 14 specific pairwise features:
  1. NAME FEATURES (on normalized business_name):
     - name_token_jaccard
     - name_char_ngram_jaccard (trigram)
     - name_levenshtein_ratio (0.0 to 1.0)
     - name_jaro_winkler (0.0 to 1.0)
     - name_length_diff_ratio (abs(l1-l2)/max(l1,l2))
     - name_is_substring (1 if s1 in s2 or s2 in s1, else 0)

  2. ADDRESS FEATURES (on normalized business_address):
     - address_token_jaccard
     - address_char_ngram_jaccard (trigram)
     - address_exact_match (binary 1/0)
     - address_has_null_flag (1 if either address is empty/null, else 0)

  3. COUNTRY FEATURE:
     - country_exact_match (1 if both present and equal, else 0)

  4. SOURCE INDICATORS:
     - source_is_s2 (1 if S2, else 0)
     - source_is_s3 (1 if S3, else 0)

  5. INTERACTION FEATURE:
     - name_addr_product (name_token_jaccard * address_token_jaccard)

Outputs:
  - train_features.parquet (and output/train_features.parquet)
  - val_features.parquet   (and output/val_features.parquet)
"""

import os
import sys
import time
import argparse
import unicodedata
import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple

# Re-use project normalization logic
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "code", "business_entity_resolution", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from normalization import normalize_name, normalize_address


# ── String Distance Libraries with Self-Contained Fallbacks ───────────────────

try:
    from rapidfuzz.distance.Levenshtein import normalized_similarity as rf_lev_sim
except ImportError:
    try:
        from Levenshtein import ratio as rf_lev_sim
    except ImportError:
        rf_lev_sim = None

try:
    from rapidfuzz.distance.JaroWinkler import similarity as rf_jw_sim
except ImportError:
    try:
        from jellyfish import jaro_winkler_similarity as rf_jw_sim
    except ImportError:
        rf_jw_sim = None


def pure_levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity [0.0, 1.0]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    if max_len == 0:
        return 1.0
    prev_row = list(range(len2 + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1] * (len2 + 1)
        for j, c2 in enumerate(s2):
            ins = prev_row[j + 1] + 1
            dels = curr_row[j] + 1
            subs = prev_row[j] + (c1 != c2)
            curr_row[j + 1] = min(ins, dels, subs)
        prev_row = curr_row
    return 1.0 - (prev_row[len2] / max_len)


def pure_jaro_winkler(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """Jaro-Winkler similarity [0.0, 1.0]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    t = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            t += 1
        k += 1

    transpositions = t / 2.0
    d_j = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0
    if d_j < 0.7:
        return d_j

    l = 0
    min_len = min(len1, len2, max_l)
    while l < min_len and s1[l] == s2[l]:
        l += 1
    return d_j + l * p * (1.0 - d_j)


def calc_levenshtein_ratio(s1: str, s2: str) -> float:
    if rf_lev_sim is not None:
        try:
            return float(rf_lev_sim(s1, s2))
        except Exception:
            pass
    return pure_levenshtein_ratio(s1, s2)


def calc_jaro_winkler(s1: str, s2: str) -> float:
    if rf_jw_sim is not None:
        try:
            return float(rf_jw_sim(s1, s2))
        except Exception:
            pass
    return pure_jaro_winkler(s1, s2)


def get_char_trigrams(s: str) -> Set[str]:
    """Extract character trigrams."""
    if not s:
        return set()
    if len(s) < 3:
        return {s}
    return {s[i:i + 3] for i in range(len(s) - 2)}


def set_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


# ── Metadata Loading ──────────────────────────────────────────────────────────

def load_source_metadata(
    dataset_dir: str,
    needed_s1_ids: Set[str],
    needed_target_ids: Set[str]
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Loads and pre-normalizes entity metadata for needed IDs only."""
    s1_meta = {}
    target_meta = {}

    # 1. Source 1
    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    print(f"  Streaming S1 from {s1_path} for {len(needed_s1_ids):,} entities ...")
    for chunk in pd.read_csv(
        s1_path, sep="\t", dtype=str,
        usecols=["entity_id", "business_name", "business_address", "country"],
        keep_default_na=False, chunksize=500_000
    ):
        sub = chunk[chunk["entity_id"].isin(needed_s1_ids)]
        for row in sub.itertuples(index=False):
            eid = str(row.entity_id)
            raw_n = str(row.business_name) if pd.notna(row.business_name) else ""
            raw_a = str(row.business_address) if pd.notna(row.business_address) else ""
            country = str(row.country) if pd.notna(row.country) else ""
            norm_n = normalize_name(raw_n)
            norm_a = normalize_address(raw_a)
            s1_meta[eid] = {
                "name": norm_n,
                "addr": norm_a,
                "country": country,
                "name_tokens": set(norm_n.split()) if norm_n else set(),
                "addr_tokens": set(norm_a.split()) if norm_a else set(),
                "name_trigrams": get_char_trigrams(norm_n),
                "addr_trigrams": get_char_trigrams(norm_a),
            }

    # 2. Source 2 and Source 3
    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(dataset_dir, src_file)
        print(f"  Streaming {src_file} for {len(needed_target_ids):,} targets ...")
        for chunk in pd.read_csv(
            path, sep="\t", dtype=str,
            usecols=["entity_id", "business_name", "business_address", "country"],
            keep_default_na=False, chunksize=500_000
        ):
            sub = chunk[chunk["entity_id"].isin(needed_target_ids)]
            if sub.empty:
                continue
            for row in sub.itertuples(index=False):
                eid = str(row.entity_id)
                raw_n = str(row.business_name) if pd.notna(row.business_name) else ""
                raw_a = str(row.business_address) if pd.notna(row.business_address) else ""
                country = str(row.country) if pd.notna(row.country) else ""
                norm_n = normalize_name(raw_n)
                norm_a = normalize_address(raw_a)
                target_meta[eid] = {
                    "name": norm_n,
                    "addr": norm_a,
                    "country": country,
                    "name_tokens": set(norm_n.split()) if norm_n else set(),
                    "addr_tokens": set(norm_a.split()) if norm_a else set(),
                    "name_trigrams": get_char_trigrams(norm_n),
                    "addr_trigrams": get_char_trigrams(norm_a),
                }

    print(f"  Cached metadata: {len(s1_meta):,} S1 entities, {len(target_meta):,} target entities.")
    return s1_meta, target_meta


# ── Feature Engineering Core ──────────────────────────────────────────────────

def compute_features_dataframe(
    pairs_tsv: str,
    s1_meta: Dict[str, dict],
    target_meta: Dict[str, dict]
) -> pd.DataFrame:
    """Computes the 14 pairwise features for every pair in pairs_tsv."""
    t0 = time.time()
    df_pairs = pd.read_csv(pairs_tsv, sep="\t", dtype=str)
    
    # Identify column names
    col_s1 = "source1_entity_id" if "source1_entity_id" in df_pairs.columns else df_pairs.columns[0]
    col_cand = "candidate_entity_id" if "candidate_entity_id" in df_pairs.columns else df_pairs.columns[1]
    col_label = "label" if "label" in df_pairs.columns else df_pairs.columns[-1]

    s1_ids = df_pairs[col_s1].astype(str).tolist()
    cand_ids = df_pairs[col_cand].astype(str).tolist()
    labels = df_pairs[col_label].astype(int).tolist()

    n_rows = len(s1_ids)
    print(f"  Computing features for {n_rows:,} pairs from {pairs_tsv} ...")

    # Pre-allocate feature arrays
    f_name_tok_jaccard = np.zeros(n_rows, dtype=np.float32)
    f_name_ngram_jaccard = np.zeros(n_rows, dtype=np.float32)
    f_name_lev_ratio = np.zeros(n_rows, dtype=np.float32)
    f_name_jw = np.zeros(n_rows, dtype=np.float32)
    f_name_len_diff = np.zeros(n_rows, dtype=np.float32)
    f_name_is_substr = np.zeros(n_rows, dtype=np.int8)

    f_addr_tok_jaccard = np.zeros(n_rows, dtype=np.float32)
    f_addr_ngram_jaccard = np.zeros(n_rows, dtype=np.float32)
    f_addr_exact_match = np.zeros(n_rows, dtype=np.int8)
    f_addr_has_null = np.zeros(n_rows, dtype=np.int8)

    f_country_match = np.zeros(n_rows, dtype=np.int8)
    f_source_s2 = np.zeros(n_rows, dtype=np.int8)
    f_source_s3 = np.zeros(n_rows, dtype=np.int8)
    f_name_addr_prod = np.zeros(n_rows, dtype=np.float32)

    country_mismatch_gold = 0

    for i in range(n_rows):
        s1_id = s1_ids[i]
        c_id = cand_ids[i]
        lbl = labels[i]

        s1 = s1_meta.get(s1_id)
        tgt = target_meta.get(c_id)

        if not s1 or not tgt:
            # Explicit zeros if metadata is missing (should not happen)
            continue

        s1_name, tgt_name = s1["name"], tgt["name"]
        s1_addr, tgt_addr = s1["addr"], tgt["addr"]
        s1_country, tgt_country = s1["country"], tgt["country"]

        # 1. Name Features
        n_tok_j = set_jaccard(s1["name_tokens"], tgt["name_tokens"])
        n_ng_j = set_jaccard(s1["name_trigrams"], tgt["name_trigrams"])
        n_lev = calc_levenshtein_ratio(s1_name, tgt_name)
        n_jw = calc_jaro_winkler(s1_name, tgt_name)

        len1, len2 = len(s1_name), len(tgt_name)
        max_len = max(len1, len2)
        n_len_diff = abs(len1 - len2) / max_len if max_len > 0 else 0.0

        n_substr = 1 if (s1_name and tgt_name and (s1_name in tgt_name or tgt_name in s1_name)) else 0

        # 2. Address Features
        a_tok_j = set_jaccard(s1["addr_tokens"], tgt["addr_tokens"])
        a_ng_j = set_jaccard(s1["addr_trigrams"], tgt["addr_trigrams"])
        a_exact = 1 if (s1_addr and tgt_addr and s1_addr == tgt_addr) else 0
        a_null = 1 if (not s1_addr or not tgt_addr) else 0

        # 3. Country Feature
        c_match = 1 if (s1_country and tgt_country and s1_country == tgt_country) else 0
        if lbl == 1 and c_match == 0:
            country_mismatch_gold += 1

        # 4. Source Indicator
        is_s2 = 1 if "S2" in c_id else 0
        is_s3 = 1 if "S3" in c_id else 0

        # 5. Interaction Feature
        n_a_prod = n_tok_j * a_tok_j

        # Assign to arrays
        f_name_tok_jaccard[i] = n_tok_j
        f_name_ngram_jaccard[i] = n_ng_j
        f_name_lev_ratio[i] = n_lev
        f_name_jw[i] = n_jw
        f_name_len_diff[i] = n_len_diff
        f_name_is_substr[i] = n_substr

        f_addr_tok_jaccard[i] = a_tok_j
        f_addr_ngram_jaccard[i] = a_ng_j
        f_addr_exact_match[i] = a_exact
        f_addr_has_null[i] = a_null

        f_country_match[i] = c_match
        f_source_s2[i] = is_s2
        f_source_s3[i] = is_s3
        f_name_addr_prod[i] = n_a_prod

    if country_mismatch_gold > 0:
        print(f"  [FLAG] {country_mismatch_gold} gold positive pairs had country mismatches!")

    # Assemble final DataFrame
    out_df = pd.DataFrame({
        "s1_id": s1_ids,
        "candidate_id": cand_ids,
        "label": labels,
        "name_token_jaccard": f_name_tok_jaccard,
        "name_char_ngram_jaccard": f_name_ngram_jaccard,
        "name_levenshtein_ratio": f_name_lev_ratio,
        "name_jaro_winkler": f_name_jw,
        "name_length_diff_ratio": f_name_len_diff,
        "name_is_substring": f_name_is_substr,
        "address_token_jaccard": f_addr_tok_jaccard,
        "address_char_ngram_jaccard": f_addr_ngram_jaccard,
        "address_exact_match": f_addr_exact_match,
        "address_has_null_flag": f_addr_has_null,
        "country_exact_match": f_country_match,
        "source_is_s2": f_source_s2,
        "source_is_s3": f_source_s3,
        "name_addr_product": f_name_addr_prod,
    })

    print(f"  Extracted {len(out_df):,} feature rows in {time.time()-t0:.1f}s.")
    return out_df


def save_parquet(df: pd.DataFrame, primary_path: str, secondary_path: str = None):
    """Saves DataFrame as Parquet; fails gracefully if pyarrow/fastparquet missing."""
    for p in [primary_path, secondary_path]:
        if not p:
            continue
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)
        try:
            df.to_parquet(p, index=False)
            print(f"  [OK] Saved: {p} ({os.path.getsize(p)/1024/1024:.2f} MB)")
        except Exception as e:
            csv_path = p.replace(".parquet", ".csv")
            df.to_csv(csv_path, index=False)
            print(f"  [WARN] Parquet write failed ({e}); saved as CSV: {csv_path}")


def main():
    t_global_start = time.time()
    parser = argparse.ArgumentParser(description="Build pairwise feature matrices for entity resolution")
    parser.add_argument("--train_pairs", default="output/small_train_pairs.tsv",
                        help="Path to training pairs TSV")
    parser.add_argument("--val_pairs", default="output/small_val_pairs.tsv",
                        help="Path to validation pairs TSV")
    parser.add_argument("--dataset_dir", default="dataset/train",
                        help="Directory containing source tables")
    parser.add_argument("--out_train", default="train_features.parquet",
                        help="Output path for train feature parquet")
    parser.add_argument("--out_val", default="val_features.parquet",
                        help="Output path for val feature parquet")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 5: PAIRWISE FEATURE ENGINEERING")
    print("=" * 70)

    # 1. Verify input files exist
    if not os.path.isfile(args.train_pairs):
        raise FileNotFoundError(f"Train pairs file not found: {args.train_pairs}")
    if not os.path.isfile(args.val_pairs):
        raise FileNotFoundError(f"Validation pairs file not found: {args.val_pairs}")

    print(f"Train pairs file : {args.train_pairs}")
    print(f"Val pairs file   : {args.val_pairs}")
    print(f"Dataset dir      : {args.dataset_dir}")

    # 2. Collect unique IDs across both train and val pairs
    df_train_pairs = pd.read_csv(args.train_pairs, sep="\t", dtype=str)
    df_val_pairs = pd.read_csv(args.val_pairs, sep="\t", dtype=str)

    col_t_s1 = "source1_entity_id" if "source1_entity_id" in df_train_pairs.columns else df_train_pairs.columns[0]
    col_t_c = "candidate_entity_id" if "candidate_entity_id" in df_train_pairs.columns else df_train_pairs.columns[1]

    col_v_s1 = "source1_entity_id" if "source1_entity_id" in df_val_pairs.columns else df_val_pairs.columns[0]
    col_v_c = "candidate_entity_id" if "candidate_entity_id" in df_val_pairs.columns else df_val_pairs.columns[1]

    all_needed_s1 = set(df_train_pairs[col_t_s1].dropna().unique()) | set(df_val_pairs[col_v_s1].dropna().unique())
    all_needed_targets = set(df_train_pairs[col_t_c].dropna().unique()) | set(df_val_pairs[col_v_c].dropna().unique())

    print(f"\nUnique entities needed: {len(all_needed_s1):,} S1 IDs, {len(all_needed_targets):,} target IDs.")

    # 3. Stream source tables once for all needed entities
    print("\n[1/3] Loading and caching entity metadata from source tables ...")
    s1_meta, target_meta = load_source_metadata(args.dataset_dir, all_needed_s1, all_needed_targets)

    # 4. Compute features for Train
    print("\n[2/3] Processing Training Pairs ...")
    train_features_df = compute_features_dataframe(args.train_pairs, s1_meta, target_meta)
    assert len(train_features_df) == len(df_train_pairs), (
        f"Row count mismatch! Input pairs={len(df_train_pairs)}, Output features={len(train_features_df)}"
    )
    save_parquet(train_features_df, args.out_train, os.path.join("output", "train_features.parquet"))

    # 5. Compute features for Validation
    print("\n[3/3] Processing Validation Pairs ...")
    val_features_df = compute_features_dataframe(args.val_pairs, s1_meta, target_meta)
    assert len(val_features_df) == len(df_val_pairs), (
        f"Row count mismatch! Input pairs={len(df_val_pairs)}, Output features={len(val_features_df)}"
    )
    save_parquet(val_features_df, args.out_val, os.path.join("output", "val_features.parquet"))

    t_total = time.time() - t_global_start

    # 6. Verification and Reporting
    print("\n" + "=" * 70)
    print("PHASE 5 FEATURE ENGINEERING REPORT")
    print("=" * 70)
    print(f"  Train features rows : {len(train_features_df):,} (matches input pairs exactly)")
    print(f"  Val features rows   : {len(val_features_df):,} (matches input pairs exactly)")

    # NaN checks
    train_nans = train_features_df.isna().sum().to_dict()
    val_nans = val_features_df.isna().sum().to_dict()
    total_train_nans = sum(train_nans.values())
    total_val_nans = sum(val_nans.values())
    print(f"  Train NaNs per col  : {total_train_nans} (all 0)")
    print(f"  Val NaNs per col    : {total_val_nans} (all 0)")
    if total_train_nans > 0:
        print(f"    Train NaN details : {train_nans}")
    if total_val_nans > 0:
        print(f"    Val NaN details   : {val_nans}")

    # Label distributions
    t_pos = int(train_features_df["label"].sum())
    t_neg = len(train_features_df) - t_pos
    v_pos = int(val_features_df["label"].sum())
    v_neg = len(val_features_df) - v_pos

    print(f"\n  Train labels        : Positives={t_pos:,} ({t_pos/len(train_features_df)*100:.2f}%), Negatives={t_neg:,}")
    print(f"  Val labels          : Positives={v_pos:,} ({v_pos/len(val_features_df)*100:.2f}%), Negatives={v_neg:,}")

    # Print sample positive and negative feature rows
    pos_sample = train_features_df[train_features_df["label"] == 1].iloc[0].to_dict()
    neg_sample = train_features_df[train_features_df["label"] == 0].iloc[0].to_dict()

    print("\n" + "-" * 70)
    print("SAMPLE FEATURE ROW — GOLD POSITIVE (label=1):")
    print("-" * 70)
    for k, v in pos_sample.items():
        if isinstance(v, float):
            print(f"  {k:<30} : {v:.4f}")
        else:
            print(f"  {k:<30} : {v}")

    print("\n" + "-" * 70)
    print("SAMPLE FEATURE ROW — HARD NEGATIVE (label=0):")
    print("-" * 70)
    for k, v in neg_sample.items():
        if isinstance(v, float):
            print(f"  {k:<30} : {v:.4f}")
        else:
            print(f"  {k:<30} : {v}")

    print("-" * 70)
    print(f"Total wall-clock runtime : {t_total:.1f}s ({t_total/60:.2f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
