"""
Candidate generation module for Business Entity Resolution pipeline.
Generates candidate pairs per S1 entity using partitioned inverted indices.
Produces candidate pair mapping or candidate_pairs.tsv format.

Changes vs. original (2026-09-27):
  - Added build_raw_name_lookup()    -- entity_id -> raw business_name string (no
                                        normalization; O(1) per record, cheap)
  - Added _jaccard()                 -- token-set Jaccard, no external deps
  - Added cap_by_jaccard()           -- rank + truncate candidate set to top-K
  - Modified generate_candidates_for_s1():
      * new optional params: candidate_cap, raw_name_lookup
      * maintains _tok_cache internally -- normalize_name() called LAZILY, only
        for entity_ids that appear in a candidate union, and only once per unique
        entity_id across all S1 rows in the batch
      * when candidate_cap is None: identical to original (zero behaviour change)
"""

import os
import sys
import time
import unicodedata
from typing import Dict, Set, List, Iterator, Optional
import pandas as pd

try:
    from .blocking import CountryPartitionBlocking
    from .data_loader import load_tsv
    from .normalization import normalize_name
except ImportError:
    from blocking import CountryPartitionBlocking
    from data_loader import load_tsv
    from normalization import normalize_name


# ── Original helper (unchanged) ─────────────────────────────────────────────

def build_country_indices(
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    country: str,
    name_cap: int = 2000,
    digit_cap: int = 5000,
    word_cap: int = 2000
) -> CountryPartitionBlocking:
    """Builds and finalizes CountryPartitionBlocking index for a specific country."""
    engine = CountryPartitionBlocking(
        country=country,
        name_cap=name_cap,
        digit_cap=digit_cap,
        word_cap=word_cap
    )
    if not s2_df.empty:
        engine.index_records(s2_df)
    if not s3_df.empty:
        engine.index_records(s3_df)
    engine.finalize()
    return engine


# ── NEW: raw name lookup (cheap upfront) + lazy Jaccard ranking ──────────────

def build_raw_name_lookup(
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> Dict[str, str]:
    """
    Build entity_id -> raw business_name string for all S2/S3 records.

    Deliberately stores the raw (un-normalized) string so this function is
    cheap: one dict-insert per record, zero normalize_name() calls.

    normalize_name() is deferred until a specific entity_id actually appears
    in a candidate union inside generate_candidates_for_s1(), via the lazy
    _tok_cache mechanism there.  For a 1000-S1 sanity test this avoids
    normalizing the full ~4.1M S2+S3 corpus and instead normalizes only the
    ~few-thousand distinct entity_ids that appear as candidates.
    """
    lookup: Dict[str, str] = {}
    for df in (s2_df, s3_df):
        if df is None or df.empty:
            continue
        for row in df.itertuples(index=False):
            eid = str(row.entity_id)
            lookup[eid] = (
                str(row.business_name) if pd.notna(row.business_name) else ""
            )
    return lookup


def _jaccard(a: frozenset, b: frozenset) -> float:
    """
    Token-set Jaccard similarity: |A ∩ B| / |A ∪ B|.
    Returns 0.0 when both sets are empty.
    """
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union_size = len(a | b)
    return inter / union_size if union_size else 0.0


def cap_by_jaccard(
    candidates: Set[str],
    s1_name_tokens: frozenset,
    raw_lookup: Dict[str, str],
    tok_cache: Dict[str, frozenset],
    cap: int,
) -> Set[str]:
    """
    Rank `candidates` by descending token-Jaccard against S1 name tokens,
    return the top `cap` as a set.

    Determinism: primary key = descending Jaccard; tie-break = ascending eid.
    Uses fast substring pre-filtering to skip expensive normalize_name() calls
    for the vast majority of candidates that have zero token overlap.
    """
    if not s1_name_tokens:
        # If S1 has no name tokens, all Jaccard scores are 0.0; sort purely by eid
        return set(sorted(candidates)[:cap])

    scored: List[tuple] = []
    for eid in candidates:
        if eid in tok_cache:
            cand_toks = tok_cache[eid]
            score = _jaccard(s1_name_tokens, cand_toks)
        else:
            raw = raw_lookup.get(eid, "")
            if not raw:
                score = 0.0
            else:
                raw_lower = raw.lower()
                # Fast check: could raw contain any token from s1_name_tokens?
                has_potential = any(tok in raw_lower for tok in s1_name_tokens)
                if not has_potential:
                    # Check accent-folded representation in case of Latin diacritics
                    raw_folded = unicodedata.normalize("NFKD", raw_lower)
                    has_potential = any(tok in raw_folded for tok in s1_name_tokens)

                if not has_potential:
                    # Impossible to have positive Jaccard; score is strictly 0.0
                    score = 0.0
                else:
                    norm = normalize_name(raw)
                    cand_toks = frozenset(norm.split()) if norm else frozenset()
                    tok_cache[eid] = cand_toks
                    score = _jaccard(s1_name_tokens, cand_toks)

        scored.append((score, eid))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return {eid for _, eid in scored[:cap]}


# ── MODIFIED: generate_candidates_for_s1 ────────────────────────────────────

def generate_candidates_for_s1(
    s1_df: pd.DataFrame,
    blocking_engine: CountryPartitionBlocking,
    candidate_cap: Optional[int] = None,
    raw_name_lookup: Optional[Dict[str, str]] = None,
    tok_cache: Optional[Dict[str, frozenset]] = None,
) -> Dict[str, Set[str]]:
    """
    Generate candidate sets for all S1 records in a single country partition.

    Parameters
    ----------
    s1_df : DataFrame
        S1 entities for this country partition.
    blocking_engine : CountryPartitionBlocking
        Finalized inverted index for this partition.
    candidate_cap : int, optional
        Per-S1 cap. When a raw candidate union exceeds this size, candidates
        are ranked by name-token Jaccard and trimmed to the top `candidate_cap`.
        When None (default), behaviour is identical to the original: full union
        is returned for every S1 entity, no extra work done.
    raw_name_lookup : dict, optional
        entity_id -> raw (un-normalized) business_name string.
        Built by build_raw_name_lookup() from the S2/S3 DataFrames.
        Used for lazy Jaccard ranking.
    tok_cache : dict, optional
        Optional external cache mapping entity_id -> frozenset{normalized tokens}.
        If provided, persists across chunks within the country partition.

    Returns
    -------
    dict entity_id -> Set[str] of candidate entity IDs
    """
    candidates: Dict[str, Set[str]] = {}
    _raw: Dict[str, str] = raw_name_lookup or {}
    _tok_cache: Dict[str, frozenset] = tok_cache if tok_cache is not None else {}

    for row in s1_df.itertuples(index=False):
        s1_name = str(row.business_name) if pd.notna(row.business_name) else ""
        s1_addr = str(row.business_address) if pd.notna(row.business_address) else ""

        # Raw union from all three blocking passes (unchanged logic)
        cands: Set[str] = blocking_engine.get_candidates(s1_name, s1_addr)

        # ── Per-S1 candidate cap (new; zero cost when candidate_cap is None) ──
        if candidate_cap is not None and len(cands) > candidate_cap:
            norm_s1 = normalize_name(s1_name)
            s1_toks = frozenset(norm_s1.split()) if norm_s1 else frozenset()
            cands = cap_by_jaccard(cands, s1_toks, _raw, _tok_cache, candidate_cap)
        # ─────────────────────────────────────────────────────────────────────

        candidates[str(row.entity_id)] = cands

    return candidates


# ── Original helper (unchanged) ─────────────────────────────────────────────

def format_candidate_list(cands: Set[str]) -> str:
    """Formats set of candidate IDs as comma-separated string, sorted for determinism."""
    if not cands:
        return ""
    return ",".join(sorted(cands))
