#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build MMQA diagnostic splits: No-Rules, M1 (short), M2 (long)
=============================================================

We assume you already have two JSON files:

  Synthesized_three_table.json  -> 3-table MMQA samples
  Synthesized_two_table.json    -> 2-table MMQA samples

Each sample has at least:

  {
    "id_": ...,
    "Question": ...,
    "answer": ...,
    "table_names": [...],
    "tables": [
      {
        "table_columns": [...],
        "table_content": [... rows ...]
      },
      ...
    ]
  }

This script will:

  1) Optionally build "No-Rules" subsets (random 100 samples each):
        filtered_three_table_No_Rules.json
        filtered_two_table_No_Rules.json

  2) Build M1 (short, multi-table):
        - at least `min_valid_table_count` tables satisfy:
              min_rows_short <= row_count <= max_rows_short
        - samples from both two-table & three-table files
        - saved as: MMQA_M1_short_multi.json

  3) Build M2 (long, multi-table):
        - at least `min_valid_table_count` tables satisfy:
              row_count >= min_rows_long
        - samples from both two-table & three-table files
        - saved as: MMQA_M2_long_multi.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple


# -------------------- basic I/O helpers --------------------

def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# -------------------- core filtering functions --------------------

def filter_samples_by_table_row_range(
    data: List[Dict[str, Any]],
    min_row_count: int = None,
    max_row_count: int = None,
    sample_table_num: int = None,
    min_valid_table_count: int = 2,
    max_sample_num: int = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Filter samples by the row-count range of their tables.

    For each sample:
      1) Randomly pick `sample_table_num` tables (or all tables if None).
      2) Count how many of these tables have row_count in [min_row_count, max_row_count].
      3) If count >= min_valid_table_count, keep this sample.
      4) Stop when we have collected max_sample_num samples (if not None).

    Args:
      data: list of MMQA samples.
      min_row_count: minimum number of rows (inclusive), or None to ignore lower bound.
      max_row_count: maximum number of rows (inclusive), or None to ignore upper bound.
      sample_table_num: number of tables to randomly sample per sample; if None, use all.
      min_valid_table_count: minimum number of tables satisfying the row constraint.
      max_sample_num: maximum number of samples to keep; None means no limit.
      seed: random seed for reproducibility.

    Returns:
      A list of filtered samples.
    """
    random.seed(seed)
    selected = []

    for sample in data:
        tables = sample.get("tables", [])
        if not tables:
            continue

        # Decide how many tables to sample
        if sample_table_num is None:
            k = len(tables)
        else:
            k = min(sample_table_num, len(tables))

        chosen_tables = random.sample(tables, k=k)

        valid_count = 0
        for t in chosen_tables:
            row_count = len(t.get("table_content", []))
            if ((min_row_count is None or row_count >= min_row_count) and
                (max_row_count is None or row_count <= max_row_count)):
                valid_count += 1

        if valid_count >= min_valid_table_count:
            selected.append(sample)
            if max_sample_num is not None and len(selected) >= max_sample_num:
                break

    return selected


def compute_avg_table_size(
    data: List[Dict[str, Any]]
) -> Tuple[float, float]:
    """
    Compute average rows and columns per table across all samples.
    """
    total_rows, total_cols, total_tables = 0, 0, 0
    for sample in data:
        for t in sample.get("tables", []):
            rows = len(t.get("table_content", []))
            cols = len(t.get("table_columns", []))
            total_rows += rows
            total_cols += cols
            total_tables += 1
    if total_tables == 0:
        return 0.0, 0.0
    avg_rows = total_rows / total_tables
    avg_cols = total_cols / total_tables
    return avg_rows, avg_cols


# -------------------- main pipeline --------------------

def build_mmqa_splits(args):
    # Resolve paths
    three_path = Path(args.three_table_json).resolve()
    two_path = Path(args.two_table_json).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"[CFG] three_table_json = {three_path}")
    print(f"[CFG] two_table_json   = {two_path}")
    print(f"[CFG] out_dir          = {out_dir}")
    print(f"[CFG] random_seed      = {args.seed}")
    print(f"[CFG] short_threshold  = <= {args.short_max_rows} rows")
    print(f"[CFG] long_threshold   = >= {args.long_min_rows} rows")
    print(f"[CFG] min_valid_tables = {args.min_valid_table_count}")
    print(f"[CFG] max_samples      = {args.max_samples}")
    print(f"[CFG] build_no_rules   = {args.build_no_rules}")

    # Load source data
    data_three = load_json(three_path)
    data_two = load_json(two_path)

    print(f"[INFO] Loaded {len(data_three)} three-table samples")
    print(f"[INFO] Loaded {len(data_two)} two-table samples")

    # Optionally: build "No-Rules" random subsets (original version)
    if args.build_no_rules:
        print("\n[STEP] Building No-Rules random subsets...")

        filtered_three_no = filter_samples_by_table_row_range(
            data_three,
            min_row_count=None,
            max_row_count=None,
            sample_table_num=None,
            min_valid_table_count=args.min_valid_table_count,
            max_sample_num=args.max_samples,
            seed=args.seed,
        )
        filtered_two_no = filter_samples_by_table_row_range(
            data_two,
            min_row_count=None,
            max_row_count=None,
            sample_table_num=None,
            min_valid_table_count=args.min_valid_table_count,
            max_sample_num=args.max_samples,
            seed=args.seed,
        )

        out_three_no = out_dir / "filtered_three_table_No_Rules.json"
        out_two_no = out_dir / "filtered_two_table_No_Rules.json"
        save_json(filtered_three_no, out_three_no)
        save_json(filtered_two_no, out_two_no)

        print(f"[INFO] three-table No-Rules: {len(filtered_three_no)} → {out_three_no}")
        print(f"[INFO] two-table No-Rules  : {len(filtered_two_no)} → {out_two_no}")

    # --------- Build M1: short multi-table ---------
    print("\n[STEP] Building M1 (short, multi-table)...")

    M1_three = filter_samples_by_table_row_range(
        data_three,
        min_row_count=args.short_min_rows,
        max_row_count=args.short_max_rows,
        sample_table_num=None,
        min_valid_table_count=args.min_valid_table_count,
        max_sample_num=args.max_samples,
        seed=args.seed,
    )
    M1_two = filter_samples_by_table_row_range(
        data_two,
        min_row_count=args.short_min_rows,
        max_row_count=args.short_max_rows,
        sample_table_num=None,
        min_valid_table_count=args.min_valid_table_count,
        max_sample_num=args.max_samples,
        seed=args.seed,
    )
    M1_all = M1_three + M1_two
    out_M1 = out_dir / "MMQA_M1_short_multi.json"
    save_json(M1_all, out_M1)

    print(f"[INFO] M1 three-table: {len(M1_three)} samples")
    print(f"[INFO] M1 two-table  : {len(M1_two)} samples")
    print(f"[INFO] M1 total      : {len(M1_all)} → {out_M1}")

    # --------- Build M2: long multi-table ---------
    print("\n[STEP] Building M2 (long, multi-table)...")

    M2_three = filter_samples_by_table_row_range(
        data_three,
        min_row_count=args.long_min_rows,
        max_row_count=None,
        sample_table_num=None,
        min_valid_table_count=args.min_valid_table_count,
        max_sample_num=args.max_samples,
        seed=args.seed,
    )
    M2_two = filter_samples_by_table_row_range(
        data_two,
        min_row_count=args.long_min_rows,
        max_row_count=None,
        sample_table_num=None,
        min_valid_table_count=args.min_valid_table_count,
        max_sample_num=args.max_samples,
        seed=args.seed,
    )
    M2_all = M2_three + M2_two
    out_M2 = out_dir / "MMQA_M2_long_multi.json"
    save_json(M2_all, out_M2)

    print(f"[INFO] M2 three-table: {len(M2_three)} samples")
    print(f"[INFO] M2 two-table  : {len(M2_two)} samples")
    print(f"[INFO] M2 total      : {len(M2_all)} → {out_M2}")

    print("\n✅ Done. MMQA diagnostic splits have been generated.")


# -------------------- argparse --------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build MMQA diagnostic splits: No-Rules, M1 (short), M2 (long)."
    )

    parser.add_argument(
        "--three_table_json",
        type=str,
        required=True,
        help="Path to Synthesized_three_table.json (3-table MMQA samples).",
    )
    parser.add_argument(
        "--two_table_json",
        type=str,
        required=True,
        help="Path to Synthesized_two_table.json (2-table MMQA samples).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory for all generated JSON files.",
    )

    # thresholds for short / long
    parser.add_argument(
        "--short_min_rows",
        type=int,
        default=1,
        help="Minimum row count for a 'short' table (inclusive).",
    )
    parser.add_argument(
        "--short_max_rows",
        type=int,
        default=100,
        help="Maximum row count for a 'short' table (inclusive).",
    )
    parser.add_argument(
        "--long_min_rows",
        type=int,
        default=101,
        help="Minimum row count for a 'long' table (inclusive).",
    )

    # multi-table & sampling settings
    parser.add_argument(
        "--min_valid_table_count",
        type=int,
        default=2,
        help="Minimum number of tables in a sample that must satisfy the row-range condition.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples per (three-table / two-table) group for M1 / M2. "
             "Set a large number if you don't want downsampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )

    # optional: also build No-Rules subsets
    parser.add_argument(
        "--build_no_rules",
        action="store_true",
        help="If set, also build filtered_three_table_No_Rules.json and filtered_two_table_No_Rules.json.",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    build_mmqa_splits(args)


if __name__ == "__main__":
    main()