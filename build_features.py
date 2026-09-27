"""
build_features.py
=================
Phase 5: Feature Engineering for Business Entity Resolution.

Extracts 7 specific features for candidate pairs:
  1. exact_name_match
  2. exact_address_match
  3. name_token_jaccard
  4. address_token_jaccard
  5. name_jaro_winkler
  6. source_is_s2
  7. country_match
  + label (1 if true match in ground truth, 0 otherwise)

Outputs:
  - features_train.parquet
  - features_val.parquet
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple

# Re-use normalization logic
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "code", "business_entity_resolution", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from normalization import normalize_name, normalize_address


# ── Jaro-Winkler implementation (zero external dependencies) ──────────────────

def jaro_similarity(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

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
    return (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0


def jaro_winkler(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    d_j = jaro_similarity(s1, s2)
    if d_j < 0.7:
        return d_j

    # Common prefix length up to max_l
    l = 0
    min_len = min(len(s1), len(s2), max_l)
    while l < min_len and s1[l] == s2[l]:
        l += 1

    return d_j + l * p * (1.0 - d_j)


def token_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    if not tokens1 and not tokens2:
        return 0.0
    inter = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return inter / union if union else 0.0


# ── Ground truth loader ───────────────────────────────────────────────────────

def load_gt_map(gt_path: str) -> Dict[str, Set[str]]:
    """Load train_ground_truth.tsv into {s1_id: set_of_matched_ids}."""
    df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    col_s1 = next((c for c in df.columns if "source1" in c.lower() or "s1" in c.lower()), None)
    col_m = next((c for c in df.columns if "match" in c.lower()), None)
    if not col_s1 or not col_m:
        raise ValueError(f"Cannot find columns in {gt_path}: {df.columns}")

    gt_map = {}
    for s1_id, match_str in zip(df[col_s1].astype(str), df[col_m].fillna("").astype(str)):
        if match_str:
            ids = {x.strip() for x in match_str.split(",") if x.strip()}
            gt_map[s1_id] = ids
    return gt_map


# ── Feature extraction pipeline ───────────────────────────────────────────────

def build_features_for_candidates(
    candidates_tsv: str,
    dataset_dir: str,
    gt_map: Dict[str, Set[str]],
    out_path: str
):
    t0 = time.time()
    print(f"\nProcessing candidate pairs from: {candidates_tsv}")
    
    # 1. Parse candidate pairs
    pairs = []
    needed_s1_ids = set()
    needed_target_ids = set()

    with open(candidates_tsv, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            s1_id = parts[0].strip()
            cand_str = parts[1].strip()
            if not cand_str:
                continue
            cands = [c.strip() for c in cand_str.split(",") if c.strip()]
            needed_s1_ids.add(s1_id)
            for cid in cands:
                needed_target_ids.add(cid)
                pairs.append((s1_id, cid))

    print(f"  Parsed {len(pairs):,} candidate pairs across {len(needed_s1_ids):,} S1 IDs and {len(needed_target_ids):,} unique targets.")

    # 2. Load S1 metadata
    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    print(f"  Loading S1 metadata from {s1_path} ...")
    s1_meta = {}
    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str,
                             usecols=["entity_id", "business_name", "business_address", "country"],
                             keep_default_na=False, chunksize=500_000):
        sub = chunk[chunk["entity_id"].isin(needed_s1_ids)]
        for row in sub.itertuples(index=False):
            eid = str(row.entity_id)
            name = str(row.business_name) if pd.notna(row.business_name) else ""
            addr = str(row.business_address) if pd.notna(row.business_address) else ""
            country = str(row.country) if pd.notna(row.country) else ""
            norm_n = normalize_name(name)
            norm_a = normalize_address(addr)
            s1_meta[eid] = {
                "name": norm_n,
                "addr": norm_a,
                "country": country,
                "name_toks": set(norm_n.split()) if norm_n else set(),
                "addr_toks": set(norm_a.split()) if norm_a else set(),
            }

    # 3. Load Target (S2 & S3) metadata
    target_meta = {}
    for src_name in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(dataset_dir, src_name)
        print(f"  Streaming {src_name} for {len(needed_target_ids):,} targets ...")
        for chunk in pd.read_csv(path, sep="\t", dtype=str,
                                 usecols=["entity_id", "business_name", "business_address", "country"],
                                 keep_default_na=False, chunksize=500_000):
            sub = chunk[chunk["entity_id"].isin(needed_target_ids)]
            if sub.empty:
                continue
            for row in sub.itertuples(index=False):
                eid = str(row.entity_id)
                name = str(row.business_name) if pd.notna(row.business_name) else ""
                addr = str(row.business_address) if pd.notna(row.business_address) else ""
                country = str(row.country) if pd.notna(row.country) else ""
                norm_n = normalize_name(name)
                norm_a = normalize_address(addr)
                target_meta[eid] = {
                    "name": norm_n,
                    "addr": norm_a,
                    "country": country,
                    "name_toks": set(norm_n.split()) if norm_n else set(),
                    "addr_toks": set(norm_a.split()) if norm_a else set(),
                }

    print(f"  Loaded metadata for {len(s1_meta):,} S1 entities and {len(target_meta):,} target entities.")

    # 4. Compute 7 features
    print("  Computing features for candidate pairs ...")
    f_exact_name = []
    f_exact_addr = []
    f_name_jaccard = []
    f_addr_jaccard = []
    f_name_jw = []
    f_source_s2 = []
    f_country_match = []
    labels = []
    s1_col = []
    target_col = []

    for s1_id, target_id in pairs:
        s1 = s1_meta.get(s1_id)
        tgt = target_meta.get(target_id)
        if not s1 or not tgt:
            continue

        s1_name, tgt_name = s1["name"], tgt["name"]
        s1_addr, tgt_addr = s1["addr"], tgt["addr"]

        # 1. exact_name_match
        ex_n = 1 if s1_name and s1_name == tgt_name else 0
        # 2. exact_address_match
        ex_a = 1 if s1_addr and s1_addr == tgt_addr else 0
        # 3. name_token_jaccard
        nj = token_jaccard(s1["name_toks"], tgt["name_toks"])
        # 4. address_token_jaccard
        aj = token_jaccard(s1["addr_toks"], tgt["addr_toks"])
        # 5. name_jaro_winkler
        jw = jaro_winkler(s1_name, tgt_name) if s1_name and tgt_name else 0.0
        # 6. source_is_s2
        is_s2 = 1 if "S2" in target_id else 0
        # 7. country_match
        c_match = 1 if s1["country"] and s1["country"] == tgt["country"] else 0

        # Label
        matched_set = gt_map.get(s1_id, set())
        lbl = 1 if target_id in matched_set else 0

        s1_col.append(s1_id)
        target_col.append(target_id)
        f_exact_name.append(ex_n)
        f_exact_addr.append(ex_a)
        f_name_jaccard.append(nj)
        f_addr_jaccard.append(aj)
        f_name_jw.append(jw)
        f_source_s2.append(is_s2)
        f_country_match.append(c_match)
        labels.append(lbl)

    df_feat = pd.DataFrame({
        "source1_entity_id": s1_col,
        "target_entity_id": target_col,
        "exact_name_match": np.array(f_exact_name, dtype=np.int8),
        "exact_address_match": np.array(f_exact_addr, dtype=np.int8),
        "name_token_jaccard": np.array(f_name_jaccard, dtype=np.float32),
        "address_token_jaccard": np.array(f_addr_jaccard, dtype=np.float32),
        "name_jaro_winkler": np.array(f_name_jw, dtype=np.float32),
        "source_is_s2": np.array(f_source_s2, dtype=np.int8),
        "country_match": np.array(f_country_match, dtype=np.int8),
        "label": np.array(labels, dtype=np.int8)
    })

    # Save to parquet (with fallback to csv if pyarrow missing)
    try:
        df_feat.to_parquet(out_path, index=False)
        print(f"  [OK] Saved {len(df_feat):,} feature rows to {out_path} (parquet format)")
    except Exception as e:
        csv_fallback = out_path.replace(".parquet", ".csv")
        df_feat.to_csv(csv_fallback, index=False)
        print(f"  [WARN] Parquet write failed ({e}). Saved as CSV to {csv_fallback}")

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pos_rate = (n_pos / len(labels) * 100) if labels else 0.0
    print(f"  Summary: Total={len(df_feat):,} | Positives={n_pos:,} ({pos_rate:.2f}%) | Negatives={n_neg:,} | Time={time.time()-t0:.1f}s")
    return df_feat


def main():
    parser = argparse.ArgumentParser(description="Build 7 pairwise features for candidates")
    parser.add_argument("--train_cands", default="output/candidates_train2k_k200/candidate_pairs.tsv")
    parser.add_argument("--val_cands", default="output/candidates_val500_k200/candidate_pairs.tsv")
    parser.add_argument("--dataset_dir", default="dataset/train")
    parser.add_argument("--gt", default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--out_train", default="features_train.parquet")
    parser.add_argument("--out_val", default="features_val.parquet")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 5: FEATURE ENGINEERING (7 PAIRWISE FEATURES)")
    print("=" * 70)

    # Load ground truth once
    print(f"Loading ground truth from {args.gt} ...")
    gt_map = load_gt_map(args.gt)
    print(f"  Loaded ground truth for {len(gt_map):,} S1 entities.")

    # Train features
    if os.path.exists(args.train_cands):
        build_features_for_candidates(args.train_cands, args.dataset_dir, gt_map, args.out_train)
    else:
        print(f"Train candidate file {args.train_cands} does not exist yet.")

    # Val features
    if os.path.exists(args.val_cands):
        build_features_for_candidates(args.val_cands, args.dataset_dir, gt_map, args.out_val)
    else:
        print(f"Val candidate file {args.val_cands} does not exist yet.")

    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()
