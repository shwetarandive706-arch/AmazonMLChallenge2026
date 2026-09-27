"""
run_candidate_generation.py
============================
Production + sanity-test runner for the capped candidate generation pipeline.

Wires together:
  CountryPartitionBlocking  (blocking.py)            -- inverted index per country
  build_raw_name_lookup     (candidate_generation.py) -- cheap entity_id->raw_name dict
  generate_candidates_for_s1 (candidate_generation.py) -- lazy Jaccard cap applied

Lookup strategy (lazy, fast):
  S2/S3 records are streamed once: index construction AND a cheap raw-name dict
  (entity_id -> raw business_name string, no normalize_name calls) are built in
  the same pass.  normalize_name() is called lazily inside
  generate_candidates_for_s1() only for entity_ids that appear in a candidate
  union AND whose union exceeds max_candidates.  For a 1000-S1 sanity test this
  normalizes ~tens of thousands of unique candidates instead of 4.1M records.

Usage
-----
# India 1000-entity sanity test, K=100:
python run_candidate_generation.py \\
    --countries India \\
    --sample_s1_limit 1000 \\
    --max_candidates 100 \\
    --gt dataset/train/train_ground_truth.tsv \\
    --out_dir output/candidates_k100/

# India 1000-entity sanity test, K=200:
python run_candidate_generation.py \\
    --countries India \\
    --sample_s1_limit 1000 \\
    --max_candidates 200 \\
    --gt dataset/train/train_ground_truth.tsv \\
    --out_dir output/candidates_k200/

# Full generation (do NOT run yet):
python run_candidate_generation.py \\
    --max_candidates 200 \\
    --out_dir output/candidates_k200_full/

Output files (in --out_dir)
---------------------------
  candidate_pairs.tsv   -- source1_entity_id <TAB> candidate_ids (comma-sep, sorted)
  run_stats.txt         -- timing, distribution, recall summary
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Set, Optional, List

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "code", "business_entity_resolution", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from blocking import CountryPartitionBlocking
from normalization import normalize_name, is_latin_majority
from candidate_generation import (
    build_raw_name_lookup,
    generate_candidates_for_s1,
    format_candidate_list,
)

CHUNK = 500_000


# ── Logging to file + stdout ──────────────────────────────────────────────────

class Tee:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.fh = open(path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.fh.write(msg)
        self.fh.flush()

    def flush(self):
        self.terminal.flush()
        self.fh.flush()

    def close(self):
        self.fh.close()


# ── GT loading ────────────────────────────────────────────────────────────────

def _detect_gt_columns(df: pd.DataFrame):
    """Return (col_s1, col_match) column names from a GT DataFrame."""
    col_s1 = next(
        (c for c in df.columns
         if "source1" in c.lower() or
         (c.lower().startswith("s1") and "entity" in c.lower())),
        None
    )
    col_match = next(
        (c for c in df.columns if "match" in c.lower()), None
    )
    if col_s1 is None or col_match is None:
        raise ValueError(
            f"Cannot find required columns in GT file.\n"
            f"Found: {df.columns.tolist()}"
        )
    return col_s1, col_match


def _df_to_gt_map(df: pd.DataFrame, col_s1: str, col_match: str) -> Dict[str, List[str]]:
    """
    Convert a GT DataFrame to {s1_id: [matched_ids...]} using fast vectorised ops.

    Replaces the previous iterrows() loop which was O(N) Python-level iterations
    over up to 2.2M rows and took 3-8 minutes.  This version uses pandas apply
    on a single column, which is ~20-50x faster.
    """
    # Strip whitespace from s1 IDs vectorially
    s1_ids = df[col_s1].str.strip()
    match_strs = df[col_match].fillna("")

    gt_map: Dict[str, List[str]] = {}
    for s1_id, match_str in zip(s1_ids, match_strs):
        if not match_str:
            continue
        matches = [m.strip() for m in match_str.split(",") if m.strip()]
        if matches:
            gt_map[s1_id] = matches
    return gt_map


def load_ground_truth(
    gt_path: str,
    allowed_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, List[str]]:
    """
    Load train_ground_truth.tsv -> {source1_entity_id: [matched_ids...]}.

    Parameters
    ----------
    gt_path : str
        Path to train_ground_truth.tsv.
    allowed_s1_ids : set, optional
        When provided (sanity-test mode), only GT rows whose source1_entity_id
        is in this set are loaded.  For a 1000-entity sanity test this reduces
        the working set from 2.2M rows to ~1000, making loading near-instant.
        When None (full-dataset mode), all rows are loaded (unchanged behaviour).

    Implementation note
    -------------------
    Uses zip() over two pandas Series (vectorised str ops) instead of iterrows(),
    which was the original bottleneck: iterrows() converts each of 2.2M rows to
    a Python Series object, taking 3-8 minutes.  The new path takes ~5-10s for
    the full file and <1s when filtered.
    """
    # When we have an allowed set we can skip rows cheaply using pandas filtering
    # before the Python-level loop, keeping the hot loop small.
    if allowed_s1_ids is not None:
        # Read only the two needed columns, filter by allowed set immediately
        df = pd.read_csv(gt_path, sep="\t", dtype=str,
                         keep_default_na=False, na_values=[""])
        col_s1, col_match = _detect_gt_columns(df)
        df = df[df[col_s1].str.strip().isin(allowed_s1_ids)]
        print(f"    GT filter: {len(df):,} rows kept for {len(allowed_s1_ids):,} "
              f"sampled S1 IDs (out of {col_s1} column).")
        return _df_to_gt_map(df, col_s1, col_match)
    else:
        df = pd.read_csv(gt_path, sep="\t", dtype=str,
                         keep_default_na=False, na_values=[""])
        col_s1, col_match = _detect_gt_columns(df)
        return _df_to_gt_map(df, col_s1, col_match)


# ── Per-country pipeline ──────────────────────────────────────────────────────

def process_country(
    country: str,
    dataset_dir: str,
    max_candidates: Optional[int],
    sample_s1_limit: Optional[int],
    gt_map: Optional[Dict[str, List[str]]],
    name_cap: int,
    digit_cap: int,
    word_cap: int,
    out_fh,
    allowed_s1_ids: Optional[Set[str]] = None,
) -> Dict:
    """
    Full pipeline for one country partition:
      1. Stream S2+S3 -> build blocking index + raw_name_lookup (single pass each)
      2. Finalize indices; print posting-list stats
      3. Stream S1 -> generate_candidates_for_s1 with lazy Jaccard cap
      4. Write candidate_pairs.tsv rows via out_fh
      5. Return stats dict
    """
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"COUNTRY PARTITION: {country}")
    print(f"{'='*70}")

    s1_path = os.path.join(dataset_dir, "train_source1.tsv")
    s2_path = os.path.join(dataset_dir, "train_source2.tsv")
    s3_path = os.path.join(dataset_dir, "train_source3.tsv")

    engine = CountryPartitionBlocking(
        country=country,
        name_cap=name_cap,
        digit_cap=digit_cap,
        word_cap=word_cap,
    )

    # raw_name_lookup: entity_id -> raw business_name string (no normalization)
    # Built in the same streaming pass as index construction.
    raw_name_lookup: Dict[str, str] = {}

    for src_label, src_path in [("S2", s2_path), ("S3", s3_path)]:
        print(f"  Streaming {src_label} ({country}) -> index + raw name lookup ...")
        rec_count = 0
        for chunk in pd.read_csv(
            src_path, sep="\t", dtype=str,
            usecols=["entity_id", "business_name", "business_address", "country"],
            keep_default_na=False, na_values=[""], chunksize=CHUNK
        ):
            c = chunk[chunk["country"] == country]
            if c.empty:
                continue

            # Index records for blocking (unchanged)
            engine.index_records(c)

            # Cheap raw-name dict: plain string assignment, zero normalize calls
            for row in c.itertuples(index=False):
                eid = str(row.entity_id)
                raw_name_lookup[eid] = (
                    str(row.business_name) if pd.notna(row.business_name) else ""
                )
            rec_count += len(c)

        print(f"    {rec_count:,} {src_label} records indexed; "
              f"{len(raw_name_lookup):,} entries in raw name lookup.")

    # Finalize indices --------------------------------------------------------
    print(f"  Finalizing indices ...")
    idx_stats = engine.finalize()

    print(f"\n  Index Posting-List Distribution ({country}):")
    hdr = (f"  {'Index':<10} {'Keys':>8} {'Postings':>12} "
           f"{'p50':>6} {'p90':>6} {'p99':>6} {'Max':>8} "
           f"{'Cap':>6} {'Capped%':>8}")
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")
    for idx_name, s in idx_stats.items():
        print(
            f"  {idx_name:<10} {s['total_keys']:>8,} {s['total_postings']:>12,} "
            f"{s['p50']:>6.0f} {s['p90']:>6.0f} {s['p99']:>6.0f} "
            f"{s['max']:>8,} {s['cap']:>6,} {s['capped_keys_pct']:>7.2f}%"
        )

    # Stream S1 and generate candidates ---------------------------------------
    print(f"\n  Streaming S1 ({country}), max_candidates={max_candidates} ...")

    s1_count = 0
    cand_counts: List[int] = []
    true_pairs_total = 0
    true_pairs_recalled = 0
    true_pairs_missed = 0

    WRITE_CHUNK = 50_000
    buffer_rows: List[str] = []
    country_tok_cache: Dict[str, frozenset] = {}

    for chunk in pd.read_csv(
        s1_path, sep="\t", dtype=str,
        usecols=["entity_id", "business_name", "business_address", "country"],
        keep_default_na=False, na_values=[""], chunksize=CHUNK
    ):
        c = chunk[chunk["country"] == country]
        if c.empty:
            continue

        if allowed_s1_ids is not None:
            c = c[c["entity_id"].isin(allowed_s1_ids)]
            if c.empty:
                continue
        elif sample_s1_limit is not None:
            remaining = sample_s1_limit - s1_count
            if remaining <= 0:
                break
            c = c.head(remaining)

        # generate_candidates_for_s1 applies cap + lazy Jaccard internally
        results = generate_candidates_for_s1(
            s1_df=c,
            blocking_engine=engine,
            candidate_cap=max_candidates,
            raw_name_lookup=raw_name_lookup,   # cheap raw-name dict, not token dict
            tok_cache=country_tok_cache,
        )

        for s1_id, cands in results.items():
            cand_counts.append(len(cands))

            buffer_rows.append(
                f"{s1_id}\t{format_candidate_list(cands)}\n"
            )
            if len(buffer_rows) >= WRITE_CHUNK:
                out_fh.writelines(buffer_rows)
                buffer_rows = []

            if gt_map is not None:
                for tm in gt_map.get(s1_id, []):
                    true_pairs_total += 1
                    if tm in cands:
                        true_pairs_recalled += 1
                    else:
                        true_pairs_missed += 1

        s1_count += len(results)
        if sample_s1_limit is not None and s1_count >= sample_s1_limit:
            break

    if buffer_rows:
        out_fh.writelines(buffer_rows)

    t_elapsed = time.time() - t0

    # Stats -------------------------------------------------------------------
    arr = np.array(cand_counts, dtype=np.int32) if cand_counts else np.array([0])
    total_pairs = int(arr.sum())
    recall = (true_pairs_recalled / true_pairs_total) if true_pairs_total > 0 else None

    print(f"\n  {country} Results:")
    print(f"    S1 entities processed  : {s1_count:,}")
    print(f"    Total candidate pairs  : {total_pairs:,}")
    print(f"    Mean   candidates/S1   : {arr.mean():.1f}")
    print(f"    Median candidates/S1   : {float(np.median(arr)):.1f}")
    print(f"    p90  candidates/S1     : {np.percentile(arr, 90):.1f}")
    print(f"    p99  candidates/S1     : {np.percentile(arr, 99):.1f}")
    print(f"    Max  candidates/S1     : {int(arr.max()):,}")
    if recall is not None:
        print(f"    True match pairs       : {true_pairs_total:,}")
        print(f"    True matches retained  : {true_pairs_recalled:,}")
        print(f"    True matches missed    : {true_pairs_missed:,}")
        print(f"    Recall                 : {recall*100:.4f}%")
    print(f"    Wall-clock time        : {t_elapsed:.1f}s")

    return {
        "country":             country,
        "s1_count":            s1_count,
        "total_pairs":         total_pairs,
        "cand_counts":         cand_counts,
        "true_pairs_total":    true_pairs_total,
        "true_pairs_recalled": true_pairs_recalled,
        "true_pairs_missed":   true_pairs_missed,
        "recall":              recall,
        "idx_stats":           idx_stats,
        "elapsed":             t_elapsed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Capped candidate generation for Business Entity Resolution"
    )
    parser.add_argument(
        "--countries", default=None,
        help="Comma-separated list of countries to process. "
             "Omit to process all countries in train_source1.tsv."
    )
    parser.add_argument(
        "--dataset_dir", default="dataset/train",
        help="Directory containing train_source*.tsv files (default: dataset/train)"
    )
    parser.add_argument(
        "--out_dir", default="output/candidates",
        help="Output directory (default: output/candidates)"
    )
    parser.add_argument(
        "--max_candidates", type=int, default=None,
        help="Per-S1 candidate cap.  When set, raw unions exceeding this are "
             "ranked by name-token Jaccard (lazily computed) and trimmed. "
             "None = no cap (original behaviour)."
    )
    parser.add_argument(
        "--gt", default=None,
        help="Path to train_ground_truth.tsv for recall evaluation (optional)."
    )
    parser.add_argument(
        "--sample_s1_limit", type=int, default=None,
        help="Limit total S1 entities processed (sanity tests only)."
    )
    parser.add_argument(
        "--s1_ids_file", default=None,
        help="Path to file containing allowed S1 entity IDs (one per line). "
             "When provided, candidate generation is restricted to these IDs only."
    )
    parser.add_argument("--name_cap",  type=int, default=2000)
    parser.add_argument("--digit_cap", type=int, default=5000)
    parser.add_argument("--word_cap",  type=int, default=2000)
    args = parser.parse_args()

    if args.s1_ids_file and args.sample_s1_limit is not None:
        parser.error("Cannot use both --s1_ids_file and --sample_s1_limit together.")

    t_total_start = time.time()

    # Setup output ------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    log_path   = os.path.join(args.out_dir, "run_stats.txt")
    pairs_path = os.path.join(args.out_dir, "candidate_pairs.tsv")

    tee = Tee(log_path)
    sys.stdout = tee

    print("=" * 70)
    print("CANDIDATE GENERATION PIPELINE")
    print("=" * 70)
    print(f"  dataset_dir     : {args.dataset_dir}")
    print(f"  out_dir         : {args.out_dir}")
    print(f"  max_candidates  : {args.max_candidates}")
    print(f"  sample_s1_limit : {args.sample_s1_limit}")
    print(f"  s1_ids_file     : {args.s1_ids_file}")
    print(f"  countries       : {args.countries or '(all)'}")
    print(f"  name_cap        : {args.name_cap}")
    print(f"  digit_cap       : {args.digit_cap}")
    print(f"  word_cap        : {args.word_cap}")
    print(f"  gt              : {args.gt or '(none -- recall not computed)'}")
    print(f"  lookup strategy : lazy (normalize_name only on candidate hits)")

    # Discover countries first (needed before GT filter pre-scan) -------------
    s1_path = os.path.join(args.dataset_dir, "train_source1.tsv")
    if args.countries:
        countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    else:
        print("\nScanning countries from train_source1.tsv ...")
        s1_countries = pd.read_csv(
            s1_path, sep="\t", usecols=["country"], dtype=str,
            keep_default_na=False
        )["country"].dropna().unique()
        countries = sorted(c for c in s1_countries if c)
        print(f"  Countries: {countries}")

    # Load allowed S1 IDs from file if provided --------------------------------
    allowed_s1_ids: Optional[Set[str]] = None
    if args.s1_ids_file:
        print(f"\nLoading allowed S1 IDs from {args.s1_ids_file} ...")
        with open(args.s1_ids_file, "r", encoding="utf-8") as f:
            allowed_s1_ids = {line.strip() for line in f if line.strip()}
        print(f"  Loaded {len(allowed_s1_ids):,} allowed S1 IDs.")

    # Load GT -----------------------------------------------------------------
    # 1. If --s1_ids_file is supplied, filter GT rows by that allowed set.
    # 2. If --sample_s1_limit is supplied, pre-scan S1 to filter GT rows.
    # 3. Otherwise, load all GT rows without filtering.
    gt_map = None
    if args.gt:
        gt_filter_ids = allowed_s1_ids

        if gt_filter_ids is None and args.sample_s1_limit is not None:
            print(f"\nPre-scanning S1 IDs for sanity-test GT filter "
                  f"(sample_s1_limit={args.sample_s1_limit}) ...")
            # Read only entity_id + country -- cheap, two columns only
            s1_scan = pd.read_csv(
                s1_path, sep="\t",
                usecols=["entity_id", "country"],
                dtype=str, keep_default_na=False
            )
            gt_filter_ids = set()
            for country in countries:
                country_ids = (
                    s1_scan[s1_scan["country"] == country]["entity_id"]
                    .head(args.sample_s1_limit)
                    .tolist()
                )
                gt_filter_ids.update(str(i) for i in country_ids)
            print(f"  Pre-scan complete: {len(gt_filter_ids):,} S1 IDs to evaluate.")

        print(f"\nLoading ground truth from {args.gt} ...")
        gt_map = load_ground_truth(args.gt, allowed_s1_ids=gt_filter_ids)
        print(f"  {len(gt_map):,} S1 entries with GT matches.")

    # Process each country and write output -----------------------------------
    all_stats = []
    with open(pairs_path, "w", encoding="utf-8") as out_fh:
        out_fh.write("source1_entity_id\tcandidate_ids\n")

        for country in countries:
            stats = process_country(
                country=country,
                dataset_dir=args.dataset_dir,
                max_candidates=args.max_candidates,
                sample_s1_limit=args.sample_s1_limit,
                gt_map=gt_map,
                name_cap=args.name_cap,
                digit_cap=args.digit_cap,
                word_cap=args.word_cap,
                out_fh=out_fh,
                allowed_s1_ids=allowed_s1_ids,
            )
            all_stats.append(stats)

    # Final summary -----------------------------------------------------------
    total_wall  = time.time() - t_total_start
    total_s1    = sum(s["s1_count"]            for s in all_stats)
    total_pairs = sum(s["total_pairs"]          for s in all_stats)
    total_tp    = sum(s["true_pairs_total"]     for s in all_stats)
    total_rec   = sum(s["true_pairs_recalled"]  for s in all_stats)
    total_miss  = sum(s["true_pairs_missed"]    for s in all_stats)
    all_counts  = [c for s in all_stats for c in s["cand_counts"]]

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Countries          : {', '.join(countries)}")
    print(f"  Total S1 entities  : {total_s1:,}")
    print(f"  Total cand pairs   : {total_pairs:,}")
    if all_counts:
        arr = np.array(all_counts, dtype=np.float32)
        print(f"  Mean  cands/S1     : {arr.mean():.1f}")
        print(f"  Median cands/S1    : {float(np.median(arr)):.1f}")
        print(f"  p90  cands/S1      : {np.percentile(arr, 90):.1f}")
        print(f"  p99  cands/S1      : {np.percentile(arr, 99):.1f}")
        print(f"  Max  cands/S1      : {int(arr.max()):,}")
    if total_tp > 0:
        print(f"  True match pairs   : {total_tp:,}")
        print(f"  Retained           : {total_rec:,}")
        print(f"  Missed             : {total_miss:,}")
        print(f"  Overall Recall     : {total_rec/total_tp*100:.4f}%")
    print(f"  Wall-clock time    : {total_wall:.1f}s ({total_wall/60:.1f} min)")
    print(f"\n  Output: {pairs_path}")
    print(f"  Log:    {log_path}")

    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
