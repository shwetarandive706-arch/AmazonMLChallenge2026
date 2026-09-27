"""
build_small_training_pairs.py
=============================
Creates a fast, high-quality, balanced pairwise training dataset for the 2,000 S1 train subset.

Process:
  1. Loads the 2,000 S1 IDs from splits/train_subset_2k_ids.txt.
  2. Extracts S1 records (name, address, country) from dataset/train/train_source1.tsv.
  3. Loads ground truth from dataset/train/train_ground_truth.tsv for these 2,000 S1 entities.
     - Adds every ground-truth match as a positive pair (label = 1).
  4. Collects set of unique meaningful name tokens across the 2,000 S1 entities.
  5. Single-pass streams S2 and S3, selecting records that share name tokens within the same country.
  6. Samples hard negative pairs (label = 0) with lexical overlap:
     - Must be same country
     - Must NOT be in ground truth for that S1
     - Capped at max 10 negatives per positive pair (min 5 for singletons)
     - Deterministic sampling with seed=42
  7. Outputs: output/small_train_pairs.tsv

Columns:
  source1_entity_id
  candidate_entity_id
  source
  label
"""

import os
import sys
import time
import random
import unicodedata
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Set, List, Tuple

STOPWORDS = {
    "and", "the", "of", "in", "for", "at", "to", "a", "an", "on", "by",
    "inc", "corp", "llc", "ltd", "pvt", "limited", "co", "company"
}


def clean_tokens(s: str) -> Set[str]:
    """Lightweight fast tokenization for candidate matching."""
    if not s or pd.isna(s):
        return set()
    # Normalize accents, lowercase
    s_norm = unicodedata.normalize("NFKD", str(s)).lower()
    # Extract alphanumeric tokens of length >= 3
    tokens = set()
    current = []
    for ch in s_norm:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                w = "".join(current)
                if len(w) >= 3 and w not in STOPWORDS:
                    tokens.add(w)
                current = []
    if current:
        w = "".join(current)
        if len(w) >= 3 and w not in STOPWORDS:
            tokens.add(w)
    return tokens


def main():
    t_start = time.time()
    random.seed(42)
    np.random.seed(42)

    train_ids_path = "splits/train_subset_2k_ids.txt"
    s1_path = "dataset/train/train_source1.tsv"
    s2_path = "dataset/train/train_source2.tsv"
    s3_path = "dataset/train/train_source3.tsv"
    gt_path = "dataset/train/train_ground_truth.tsv"
    out_dir = "output"
    out_path = os.path.join(out_dir, "small_train_pairs.tsv")

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 2: BUILD SMALL TRAINING PAIR DATASET (2,000 S1 SUBSET)")
    print("=" * 70)

    # 1. Load target S1 IDs ----------------------------------------------------
    print(f"[1/6] Loading target S1 IDs from {train_ids_path} ...")
    with open(train_ids_path, "r", encoding="utf-8") as f:
        target_s1_ids = {line.strip() for line in f if line.strip()}
    print(f"      Target S1 count: {len(target_s1_ids):,}")
    assert len(target_s1_ids) == 2000, f"Expected 2,000 IDs, found {len(target_s1_ids)}"

    # 2. Extract S1 records ----------------------------------------------------
    print(f"[2/6] Extracting S1 records from {s1_path} ...")
    s1_data = {}  # eid -> (name, country, tokens)
    s1_tokens_by_country = {"India": set(), "US": set()}

    for chunk in pd.read_csv(
        s1_path, sep="\t", dtype=str,
        usecols=["entity_id", "business_name", "country"],
        keep_default_na=False, chunksize=500_000
    ):
        sub = chunk[chunk["entity_id"].isin(target_s1_ids)]
        for row in sub.itertuples(index=False):
            eid = str(row.entity_id)
            name = str(row.business_name) if pd.notna(row.business_name) else ""
            country = str(row.country) if pd.notna(row.country) else ""
            toks = clean_tokens(name)
            s1_data[eid] = (name, country, toks)
            if country in s1_tokens_by_country:
                s1_tokens_by_country[country].update(toks)

    print(f"      Loaded {len(s1_data):,} S1 entities.")
    print(f"      Unique S1 query tokens: India={len(s1_tokens_by_country['India']):,}, US={len(s1_tokens_by_country['US']):,}")

    # 3. Load Ground Truth for target S1 entities ------------------------------
    print(f"[3/6] Loading Ground Truth from {gt_path} ...")
    gt_map = defaultdict(set)  # s1_id -> set of matched_entity_ids
    for chunk in pd.read_csv(
        gt_path, sep="\t", dtype=str,
        keep_default_na=False, chunksize=500_000
    ):
        col_s1 = next((c for c in chunk.columns if "source1" in c.lower() or "s1" in c.lower()), None)
        col_m = next((c for c in chunk.columns if "match" in c.lower()), None)
        sub = chunk[chunk[col_s1].isin(target_s1_ids)]
        for row in sub.itertuples(index=False):
            s1_id = str(getattr(row, col_s1))
            match_str = str(getattr(row, col_m)) if pd.notna(getattr(row, col_m)) else ""
            if match_str:
                matches = {m.strip() for m in match_str.split(",") if m.strip()}
                gt_map[s1_id].update(matches)

    total_gt_matches = sum(len(m) for m in gt_map.values())
    print(f"      Found {total_gt_matches:,} true positive links across {len(gt_map):,} non-singleton S1 entities.")

    # 4. Stream S2 and S3 for candidates sharing S1 tokens --------------------
    print(f"[4/6] Streaming S2 & S3 to collect candidate pool ...")
    # country -> token -> list of candidate eids
    token_index = {
        "India": defaultdict(list),
        "US": defaultdict(list)
    }
    all_candidate_ids = set()

    for src_name, path in [("S2", s2_path), ("S3", s3_path)]:
        t_src0 = time.time()
        print(f"      Streaming {src_name} ({path}) ...")
        cnt = 0
        for chunk in pd.read_csv(
            path, sep="\t", dtype=str,
            usecols=["entity_id", "business_name", "country"],
            keep_default_na=False, chunksize=500_000
        ):
            for row in chunk.itertuples(index=False):
                country = str(row.country)
                if country not in token_index:
                    continue
                name = str(row.business_name) if pd.notna(row.business_name) else ""
                eid = str(row.entity_id)
                toks = clean_tokens(name)
                # Check intersection with S1 tokens for this country
                shared = toks & s1_tokens_by_country[country]
                if shared:
                    for t in shared:
                        token_index[country][t].append(eid)
                    all_candidate_ids.add(eid)
                    cnt += 1
        print(f"        Matched {cnt:,} {src_name} records sharing tokens in {time.time()-t_src0:.1f}s.")

    print(f"      Total unique candidate records gathered: {len(all_candidate_ids):,}")

    # 5. Build Positive and Negative Pairs ------------------------------------
    print(f"[5/6] Assembling balanced training pairs ...")
    output_rows = []
    seen_pairs = set()

    india_s1_count = 0
    us_s1_count = 0
    s2_cand_count = 0
    s3_cand_count = 0
    pos_count = 0
    neg_count = 0

    rng = random.Random(42)

    for s1_id in sorted(target_s1_ids):
        name, country, s1_toks = s1_data.get(s1_id, ("", "", set()))
        if country == "India":
            india_s1_count += 1
        elif country == "US":
            us_s1_count += 1

        true_matches = gt_map.get(s1_id, set())

        # A. Add all true positives
        for match_id in sorted(true_matches):
            pair_key = (s1_id, match_id)
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                src = "S2" if "S2" in match_id else "S3"
                output_rows.append((s1_id, match_id, src, 1))
                pos_count += 1
                if src == "S2":
                    s2_cand_count += 1
                else:
                    s3_cand_count += 1

        # B. Find hard negatives: same country, share tokens, not true match
        cand_score = defaultdict(int)
        c_index = token_index.get(country, {})
        for t in s1_toks:
            for cid in c_index.get(t, []):
                if cid not in true_matches:
                    cand_score[cid] += 1

        # Max negatives: 10 per positive pair, or 5 if singleton
        max_negs = max(5, min(10 * len(true_matches), 30)) if true_matches else 5

        if cand_score:
            # Sort by shared token count descending, then deterministic tie-break
            scored_candidates = sorted(cand_score.items(), key=lambda x: (-x[1], x[0]))
            selected_negs = [cid for cid, _ in scored_candidates[:max_negs]]
        else:
            selected_negs = []

        for neg_id in selected_negs:
            pair_key = (s1_id, neg_id)
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                src = "S2" if "S2" in neg_id else "S3"
                output_rows.append((s1_id, neg_id, src, 0))
                neg_count += 1
                if src == "S2":
                    s2_cand_count += 1
                else:
                    s3_cand_count += 1

    # 6. Write to output TSV ---------------------------------------------------
    print(f"[6/6] Writing output to {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_id\tsource\tlabel\n")
        for s1_id, cid, src, lbl in output_rows:
            f.write(f"{s1_id}\t{cid}\t{src}\t{lbl}\n")

    t_elapsed = time.time() - t_start

    # 7. Print verification summary --------------------------------------------
    total_pairs = len(output_rows)
    pos_pct = (pos_count / total_pairs * 100) if total_pairs else 0.0

    print("\n" + "=" * 70)
    print("STEP 2 TRAINING PAIRS SUMMARY")
    print("=" * 70)
    print(f"  Total S1 entities processed : {len(target_s1_ids):,}")
    print(f"  Positive pairs (label=1)    : {pos_count:>8,}")
    print(f"  Negative pairs (label=0)    : {neg_count:>8,}")
    print(f"  Total pairs                 : {total_pairs:>8,}")
    print(f"  Positive percentage         : {pos_pct:>7.2f}%")
    print(f"\n  India / US S1 Distribution:")
    print(f"    India S1 entities         : {india_s1_count:>6,} ({india_s1_count/len(target_s1_ids)*100:.2f}%)")
    print(f"    US S1 entities            : {us_s1_count:>6,} ({us_s1_count/len(target_s1_ids)*100:.2f}%)")
    print(f"\n  Candidate Source Distribution:")
    print(f"    Source 2 (S2) candidates  : {s2_cand_count:>8,} ({s2_cand_count/total_pairs*100:.2f}%)")
    print(f"    Source 3 (S3) candidates  : {s3_cand_count:>8,} ({s3_cand_count/total_pairs*100:.2f}%)")
    print(f"\n  Output file created         : {out_path} ({os.path.getsize(out_path)/1024/1024:.2f} MB)")
    print(f"  Wall-clock time             : {t_elapsed:.1f}s ({t_elapsed/60:.2f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
