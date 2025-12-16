import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def load_data(input_path: str) -> List[Dict[str, Any]]:
    """Load matched table data from a JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_tables(
    samples: Iterable[Dict[str, Any]],
) -> Iterable[Tuple[str, str, List[str], List[List[Any]], str]]:
    """
    Iterate through each (db, table_name, columns, contents, sql_lower).

    This allows table_match / high_freq to reuse this logic.
    """
    for sample in samples:
        sql = str(sample.get("SQL", "")).lower()
        db = sample.get("db", "UNKNOWN")
        tables = sample.get("tables") or []
        table_names = sample.get("table_names") or []

        for table_name, table in zip(table_names, tables):
            columns = table.get("table_columns") or []
            contents = table.get("table_content") or []
            yield db, table_name, columns, contents, sql


def extract_examples(column_values: List[Any], limit: int = 3) -> List[str]:
    """
    Extract up to limit non-empty and non-repeated examples from a column to help understand the field meaning.
    """
    seen = set()
    examples: List[str] = []
    for v in column_values:
        if v in (None, ""):
            continue
        if v in seen:
            continue
        seen.add(v)
        examples.append(str(v))
        if len(examples) >= limit:
            break
    return examples


def get_column_values(contents: List[List[Any]], col_idx: int) -> List[Any]:
    """Safely extract all values from a column in table_content."""
    values: List[Any] = []
    for row in contents:
        if col_idx < len(row):
            values.append(row[col_idx])
    return values


def get_table_level_used_columns(matched_data: List[Dict[str, Any]], output_csv: str) -> None:
    """
    For each table, count:
    - All columns
    - Columns mentioned in SQL (matched_columns)
    - Columns not used (unmatched_columns)
    - Examples for each column
    """
    table_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "db": None,
            "all_columns": set(),
            "matched_columns": set(),
            "column_examples": {},
        }
    )

    for db, table_name, columns, contents, sql in iter_tables(matched_data):
        stats = table_stats[table_name]
        stats["db"] = db

        for col_idx, col in enumerate(columns):
            stats["all_columns"].add(col)

            # Only count once: if already exists, don't repeat
            if col not in stats["column_examples"]:
                values = get_column_values(contents, col_idx)
                stats["column_examples"][col] = extract_examples(values, limit=3)

            if col.lower() in sql:
                stats["matched_columns"].add(col)

    rows: List[Dict[str, Any]] = []
    for table, stats in table_stats.items():
        all_cols = stats["all_columns"]
        matched_cols = stats["matched_columns"]
        unmatched_cols = all_cols - matched_cols

        total = len(all_cols)
        matched = len(matched_cols)
        match_rate = matched / total if total > 0 else 0.0

        rows.append(
            {
                "table": table,
                "db": stats["db"],
                "matched_columns": ", ".join(sorted(matched_cols)),
                "unmatched_columns": ", ".join(sorted(unmatched_cols)),
                # Save as JSON string for later processing
                "column_examples": json.dumps(stats["column_examples"], ensure_ascii=False),
                "match_rate (matched/total)": f"{match_rate:.2%}",
            }
        )

    df = pd.DataFrame(rows).sort_values(by="table", ascending=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved detailed column stats per table to: {output_csv}")


def get_high_freq_columns(matched_data: List[Dict[str, Any]], output_csv: str) -> None:
    """
    For each column in each table, count:
    - Total number of occurrences in samples
    - Number of times used in SQL (matched)
    - Threshold for "high-frequency columns": matched / total >= 0.5
    """
    # column_usage_stats[table_name][column] -> { total, matched, db, examples }
    column_usage_stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "matched": 0, "db": None, "examples": []})
    )

    for db, table_name, columns, contents, sql in iter_tables(matched_data):
        for col_idx, col in enumerate(columns):
            stats = column_usage_stats[table_name][col]
            stats["db"] = db
            stats["total"] += 1

            if col.lower() in sql:
                stats["matched"] += 1

            # Only collect examples when seeing the column for the first time
            if not stats["examples"]:
                values = get_column_values(contents, col_idx)
                stats["examples"] = extract_examples(values, limit=3)

    rows: List[Dict[str, Any]] = []
    for table, cols in column_usage_stats.items():
        high_freq_cols: List[str] = []
        low_freq_cols: List[str] = []
        db = None

        for col_name, stats in cols.items():
            db = stats["db"]
            total = stats["total"]
            matched = stats["matched"]
            if total > 0 and (matched / total) >= 0.5:
                high_freq_cols.append(col_name)
            else:
                low_freq_cols.append(col_name)

        total_cols = len(high_freq_cols) + len(low_freq_cols)
        ratio = len(high_freq_cols) / total_cols if total_cols > 0 else 0.0

        # Collect all column examples into a dict
        column_examples = {c: cols[c]["examples"] for c in cols.keys()}

        rows.append(
            {
                "table": table,
                "db": db,
                "high_freq_columns": ", ".join(sorted(high_freq_cols)),
                "low_freq_columns": ", ".join(sorted(low_freq_cols)),
                "column_examples": json.dumps(column_examples, ensure_ascii=False),
                # Keep original column names to avoid downstream scripts breaking; semantic: high-frequency column ratio
                "match_rate (matched/total)": f"{ratio:.2%}",
            }
        )

    df = pd.DataFrame(rows).sort_values(by="table", ascending=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved high-frequency column summary to {output_csv}")


def count_unique_tables(
    matched_data: List[Dict[str, Any]], output_csv: str, save_csv: bool = True
) -> None:
    """
    Count the number of occurrences of each table in the samples and output CSV (optional).
    """
    table_counter: Counter = Counter()
    for sample in matched_data:
        table_names = sample.get("table_names") or []
        table_counter.update(table_names)

    total_unique = len(table_counter)
    print(f"📊 Total unique tables: {total_unique}")

    if save_csv:
        rows = [{"table": t, "count": c} for t, c in table_counter.items()]
        df = pd.DataFrame(rows).sort_values(by="count", ascending=False)
        df.to_csv(output_csv, index=False)
        print(f"Saved table count list to {output_csv}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze matched table/SQL data.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path.")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["table_match", "high_freq", "unique_table"],
        required=True,
        help="Analysis mode: table_match | high_freq | unique_table",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Whether to save CSV for unique_table mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    matched_data = load_data(args.input)

    if args.mode == "table_match":
        get_table_level_used_columns(matched_data, args.output)
    elif args.mode == "high_freq":
        get_high_freq_columns(matched_data, args.output)
    elif args.mode == "unique_table":
        count_unique_tables(matched_data, args.output, save_csv=args.save_csv)


if __name__ == "__main__":
    main()