"""
build_small_training_pairs.py
=============================
Creates a fast, high-quality, balanced pairwise training dataset for the 2,000 S1 train subset.

Fix applied:
  1. Strict AND condition: Candidates must share AT LEAST 2 TOKENS with an S1 entity,
     OR share a rare distinctive token (frequency threshold: appears in <= 5 S1s, len >= 4, not stopword).
  2. Per-S1 candidate capping (max 30 candidates per S1 entity during streaming).
  3. Hard assertion: Candidate pool MUST be <= 100,000 (expected ~10,000 to 50,000).
  4. Top 20 token frequency analysis printed before streaming.
  5. Outputs: output/small_train_pairs.tsv
"""

import os
import sys
import time
import random
import unicodedata
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict, Counter
from typing import Dict, Set, List, Tuple

# Re-use LEGAL_TOKENS from normalization.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "code", "business_entity_resolution", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from normalization import LEGAL_TOKENS
except ImportError:
    LEGAL_TOKENS = {
        "llc", "inc", "corp", "corporation", "ltd", "limited", "pvt", "private",
        "plc", "sarl", "sas", "co", "company", "llp", "pc", "lc", "gmbh", "sa",
        "sci", "incorporated", "enterprises", "associates", "holding", "holdings"
    }

LINGUISTIC_STOPWORDS = {
    "and", "the", "of", "in", "for", "at", "to", "a", "an", "on", "by",
    "with", "from", "as", "into", "through", "during", "including", "until",
    "against", "among", "throughout", "despite", "towards", "upon"
}

GENERIC_BUSINESS_WORDS = {
    "services", "service", "solutions", "solution", "group", "technologies",
    "technology", "tech", "international", "global", "industries", "industry",
    "systems", "system", "consulting", "consultancy", "management", "enterprise",
    "enterprises", "associates", "associate", "partners", "partner", "national",
    "products", "product", "trading", "logistics", "commercial", "development",
    "financial", "finance", "ventures", "venture", "capital", "properties",
    "property", "realty", "agency", "center", "centre", "network", "marketing",
    "investments", "investment", "holdings", "holding", "india", "american"
}

GENERIC_STOPWORDS = set(LEGAL_TOKENS) | LINGUISTIC_STOPWORDS | GENERIC_BUSINESS_WORDS


def clean_tokens(s: str) -> List[str]:
    """Extract list of clean normalized tokens of length >= 2."""
    if not s or pd.isna(s):
        return []
    s_norm = unicodedata.normalize("NFKD", str(s)).lower()
    tokens = []
    current = []
    for ch in s_norm:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                w = "".join(current)
                if len(w) >= 2 and w not in LINGUISTIC_STOPWORDS:
                    tokens.append(w)
                current = []
    if current:
        w = "".join(current)
        if len(w) >= 2 and w not in LINGUISTIC_STOPWORDS:
            tokens.append(w)
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
    print(f"[1/7] Loading target S1 IDs from {train_ids_path} ...")
    with open(train_ids_path, "r", encoding="utf-8") as f:
        target_s1_ids = {line.strip() for line in f if line.strip()}
    print(f"      Target S1 count: {len(target_s1_ids):,}")
    assert len(target_s1_ids) == 2000, f"Expected 2,000 IDs, found {len(target_s1_ids)}"

    # 2. Extract S1 records ----------------------------------------------------
    print(f"[2/7] Extracting S1 records from {s1_path} ...")
    s1_data = {}  # eid -> (name, country, tokens_set)

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
            toks = set(clean_tokens(name))
            s1_data[eid] = (name, country, toks)

    print(f"      Loaded {len(s1_data):,} train S1 entities.")

    # 3. Frequency Analysis and Indexing ---------------------------------------
    print(f"[3/7] Analyzing S1 token frequencies & building AND / rare-token queries ...")
    all_s1_tokens_flat = [tok for eid in target_s1_ids for tok in s1_data[eid][2]]
    token_counts = Counter(all_s1_tokens_flat)

    print("\n      Top 20 most frequent S1 tokens:")
    print(f"      {'Token':<20} {'Count':>6}  {'Category':<15}")
    print("      " + "-" * 45)
    for tok, cnt in token_counts.most_common(20):
        cat = "GENERIC_STOP" if tok in GENERIC_STOPWORDS else "DISTINCTIVE"
        print(f"      {tok:<20} {cnt:>6}  {cat:<15}")
    print("      " + "-" * 45 + "\n")

    # Build token-pair and rare-token indices per country
    s1_by_token_pair = {"India": defaultdict(list), "US": defaultdict(list)}
    s1_by_rare_token = {"India": defaultdict(list), "US": defaultdict(list)}

    for eid, (name, country, toks) in s1_data.items():
        if country not in s1_by_token_pair:
            continue
        sorted_toks = sorted(toks)
        for t1, t2 in combinations(sorted_toks, 2):
            s1_by_token_pair[country][(t1, t2)].append(eid)

        for t in toks:
            if t not in GENERIC_STOPWORDS and len(t) >= 4 and token_counts[t] <= 5:
                s1_by_rare_token[country][t].append(eid)

    # 4. Load Ground Truth for target S1 entities ------------------------------
    print(f"[4/7] Loading Ground Truth from {gt_path} ...")
    gt_map = defaultdict(set)
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

    # 5. Stream S2 & S3 with AND condition and per-S1 cap (max 30) ------------
    print(f"[5/7] Streaming S2 & S3 with strict AND condition (cap 30 per S1) ...")
    
    s1_candidate_pool = defaultdict(lambda: defaultdict(int))
    all_matched_cands_by_source = {"S2": set(), "S3": set()}

    for src_name, path in [("S2", s2_path), ("S3", s3_path)]:
        t_src0 = time.time()
        print(f"      Streaming {src_name} ({path}) ...")
        for chunk in pd.read_csv(
            path, sep="\t", dtype=str,
            usecols=["entity_id", "business_name", "country"],
            keep_default_na=False, chunksize=500_000
        ):
            for row in chunk.itertuples(index=False):
                country = str(row.country)
                if country not in s1_by_token_pair:
                    continue
                name = str(row.business_name) if pd.notna(row.business_name) else ""
                eid = str(row.entity_id)
                cand_toks = set(clean_tokens(name))
                if len(cand_toks) == 0:
                    continue

                matched_s1_for_row = defaultdict(int)

                # A. Check token pairs (THE AND CONDITION: >= 2 shared tokens)
                if len(cand_toks) >= 2:
                    sorted_cand_toks = sorted(cand_toks)
                    pair_index = s1_by_token_pair[country]
                    for t1, t2 in combinations(sorted_cand_toks, 2):
                        hit_s1s = pair_index.get((t1, t2))
                        if hit_s1s:
                            for s1_id in hit_s1s:
                                matched_s1_for_row[s1_id] += 3

                # B. Check rare distinctive tokens (single-token match only if rare)
                rare_index = s1_by_rare_token[country]
                for t in cand_toks:
                    hit_s1s = rare_index.get(t)
                    if hit_s1s:
                        for s1_id in hit_s1s:
                            matched_s1_for_row[s1_id] += 1

                # Update candidate pools with per-S1 cap
                for s1_id, score in matched_s1_for_row.items():
                    pool = s1_candidate_pool[s1_id]
                    if len(pool) < 30 or score > min(pool.values()):
                        pool[eid] = max(pool[eid], score)
                        all_matched_cands_by_source[src_name].add(eid)
                        if len(pool) > 35:
                            worst_key = min(pool.keys(), key=lambda k: pool[k])
                            del pool[worst_key]

        print(f"        Matched {len(all_matched_cands_by_source[src_name]):,} unique {src_name} records in {time.time()-t_src0:.1f}s.")

    total_candidates = len(all_matched_cands_by_source["S2"] | all_matched_cands_by_source["S3"])
    print(f"\n      Total unique candidate pool size: {total_candidates:,}")

    if total_candidates > 100_000:
        print(f"\n[ALERT] Candidate pool size {total_candidates:,} exceeds 100,000 limit!")
        print("Stopping as instructed.")
        sys.exit(1)
    else:
        print(f"      [OK] Candidate pool size {total_candidates:,} is well under 100,000 threshold.")

    # 6. Build Positive and Negative Pairs ------------------------------------
    print(f"[6/7] Assembling balanced training pairs ...")
    output_rows = []
    seen_pairs = set()

    india_s1_count = 0
    us_s1_count = 0
    s2_cand_count = 0
    s3_cand_count = 0
    pos_count = 0
    neg_count = 0

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

        # B. Add hard negatives: from pool, excluding true matches
        cand_dict = s1_candidate_pool.get(s1_id, {})
        valid_negs = [
            (cid, score) for cid, score in cand_dict.items() if cid not in true_matches
        ]
        valid_negs.sort(key=lambda x: (-x[1], x[0]))

        max_negs = max(5, min(10 * len(true_matches), 30)) if true_matches else 5
        selected_negs = [cid for cid, _ in valid_negs[:max_negs]]

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

    # 7. Write to output TSV ---------------------------------------------------
    print(f"[7/7] Writing output to {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_id\tsource\tlabel\n")
        for s1_id, cid, src, lbl in output_rows:
            f.write(f"{s1_id}\t{cid}\t{src}\t{lbl}\n")

    t_elapsed = time.time() - t_start

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
