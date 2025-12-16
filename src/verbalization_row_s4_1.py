#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verbalize tables using templates that (typically) cover all columns.

For each item:
  - For each table:
    * Load a template set (columns + templates) from a CSV.
    * Drop rows with NaN in any placeholder column.
    * Randomly select a subset of rows (controlled by `ratio`) to verbalize.
      These rows are converted into natural-language sentences and concatenated
      into a single `paragraph`.
    * The remaining rows are kept as a semi-structured table, with the
      placeholder columns removed.
  - Re-run SQL on the cleaned raw tables to recompute the answer.
  - Save successful items and failed items separately.

This script is designed for "all-columns" templates, e.g. produced by a pipeline
where column1..N are all table columns.
"""

import argparse
import copy
import json
import random
import re
from typing import Any, Dict, List, Tuple

import duckdb
import pandas as pd
from tqdm import tqdm

from utils import read_json, table_dict_to_df, df_to_dict_table


# ----------------------------------------------------------------------
# Template & verbalization helpers
# ----------------------------------------------------------------------


def pick_template_set(
    db: str,
    table_name: str,
    template_sets: Dict[Tuple[str, str], List[Dict[str, Any]]],
    df_columns: set,
) -> Dict[str, Any] | None:
    """
    From template_sets[(db, table)] pick one valid (columns + templates) entry.

    A template set has the structure:
        {
            "columns":   [col1, col2, ...],
            "templates": [tpl1, tpl2, ...],
        }

    Only sets where columns ⊆ df_columns are considered valid.
    Returns one random valid set, or None if no valid set exists.
    """
    candidates = template_sets.get((db, table_name), [])
    valid = [c for c in candidates if set(c["columns"]).issubset(df_columns)]
    return random.choice(valid) if valid else None


def verbalize_table_row(template: str, row: pd.Series, placeholders: List[str]) -> str:
    """
    Fill a single template for one row.

    If a value is NaN, it is replaced with a marker like "[col=NA]".
    """
    filled = template
    for key in placeholders:
        value = row[key] if pd.notna(row[key]) else f"[{key}=NA]"
        filled = filled.replace(f"{{{key}}}", str(value))
    return filled


def verbalize_table(
    df: pd.DataFrame,
    placeholders: List[str],
    templates: List[str],
    db: str | None = None,
    table_name: str | None = None,
    sample_id: str | None = None,
    ratio: float = 0.3,
) -> tuple[pd.DataFrame, List[str], str, pd.DataFrame]:
    """
    Verbalize a subset of rows in a table into a single paragraph.

    Args:
        df:           Original DataFrame.
        placeholders: Columns that are referenced in the templates.
        templates:    A list of template strings containing {placeholders}.
        db/table_name/sample_id: Only used for debug logging.
        ratio:        Fraction of rows to verbalize (0 < ratio < 1).

    Returns:
        semi_table:       Non-verbalized rows, with placeholder columns removed.
        used_keys:        The list of placeholder column names.
        paragraph:        Concatenated verbalized sentences.
        raw_table_wo_na:  df.dropna on placeholders (for SQL execution).
    """
    try:
        df_clean = df.dropna(subset=placeholders).copy()
    except KeyError as exc:
        missing = list(exc.args[0])
        print("\n=== COLUMN MISMATCH DETECTED ===")
        print(f"DB:           {db}")
        print(f"Table:        {table_name}")
        print(f"Sample ID:    {sample_id}")
        print(f"Placeholders: {placeholders}")
        print(f"DF columns:   {list(df.columns)}")
        print(f"Missing cols: {missing}")
        print("================================\n")
        raise

    if df_clean.empty:
        # All rows have NaN in placeholders; no verbalization possible.
        semi_table = df.copy().drop(columns=placeholders, errors="ignore")
        return semi_table, placeholders, "", df.copy()

    n_rows = len(df_clean)
    n_verbal = int(n_rows * ratio)

    # If ratio is too small or too large, skip verbalization and keep table only.
    if n_verbal < 1 or n_verbal >= n_rows:
        semi_table = df_clean.drop(columns=placeholders, errors="ignore")
        return semi_table, placeholders, "", df_clean

    verbal_indices = set(random.sample(range(n_rows), n_verbal))

    paragraph_texts: List[str] = []
    remaining_rows: List[pd.Series] = []

    for idx, (_, row) in enumerate(df_clean.iterrows()):
        if idx in verbal_indices:
            desc = verbalize_table_row(random.choice(templates), row, placeholders)
            paragraph_texts.append(desc)
        else:
            remaining_rows.append(row)

    semi_table = pd.DataFrame(remaining_rows)
    paragraph = " ".join(paragraph_texts)

    return semi_table, placeholders, paragraph, df_clean


# ----------------------------------------------------------------------
# SQL execution
# ----------------------------------------------------------------------


def execute_sql(query: str, raw_tables: Dict[str, Any]) -> List[List[Any]]:
    """
    Execute SQL on provided raw_tables (MMQA-style format).

    Strategy:
        1) Try DuckDB in-memory.
        2) If DuckDB fails, fall back to in-memory SQLite.
        3) On failure, log both errors and return [].

    raw_tables is expected to have:
        {
            "table_names": [...],
            "tables": [
                {"table_columns": [...], "table_content": [[...], ...]},
                ...
            ]
        }
    """
    table_names = raw_tables["table_names"]
    tables = raw_tables["tables"]

    def _register_duck(conn: duckdb.DuckDBPyConnection) -> None:
        for name, tbl in zip(table_names, tables):
            df = pd.DataFrame(tbl["table_content"], columns=tbl["table_columns"])
            conn.register(name, df)

    def _register_sqlite(conn) -> None:
        import sqlite3  # local import to avoid hard dependency at import time

        for name, tbl in zip(table_names, tables):
            cols_def = ", ".join([f'"{c}" TEXT' for c in tbl["table_columns"]])
            conn.execute(f'CREATE TABLE "{name}" ({cols_def});')
            placeholders = ",".join(["?"] * len(tbl["table_columns"]))
            conn.executemany(
                f'INSERT INTO "{name}" VALUES ({placeholders});',
                tbl["table_content"],
            )
            conn.commit()

    # 1) Try DuckDB
    duck_error_msg = ""
    try:
        with duckdb.connect(":memory:") as conn:
            _register_duck(conn)
            result_df = conn.query(query).df()
            return result_df.values.tolist()
    except Exception as exc:  # noqa: BLE001
        duck_error_msg = str(exc)

    # 2) Fallback to SQLite
    try:
        import sqlite3

        with sqlite3.connect(":memory:") as conn:
            _register_sqlite(conn)
            cur = conn.execute(query)
            rows = cur.fetchall()
            return rows
    except Exception as sqlite_exc:  # noqa: BLE001
        print("\n[DuckDB → SQLite BOTH FAILED]")
        print(f"DuckDB Error : {duck_error_msg}")
        print(f"SQLite Error : {sqlite_exc}")
        print(f"SQL          : {query}\n")
        return []


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------


def build_template_sets(template_df: pd.DataFrame) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """
    Build a mapping:
        (db, table) -> [ { "columns": [...], "templates": [...] }, ... ]
    from a template CSV that contains:
        - db
        - table
        - column1..columnN
        - template1..template5
    """
    template_sets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for _, row in template_df.iterrows():
        key = (row["db"], row["table"])

        cols = [
            row[col_name]
            for col_name in row.index
            if col_name.startswith("column")
            and pd.notna(row[col_name])
            and str(row[col_name]).strip()
        ]
        tpls = [
            row[f"template{i}"]
            for i in range(1, 6)
            if f"template{i}" in row
            and pd.notna(row[f"template{i}"])
            and str(row[f"template{i}"]).strip()
        ]

        if cols and tpls:
            template_sets.setdefault(key, []).append(
                {"columns": list(cols), "templates": list(tpls)}
            )

    return template_sets


def run(
    data_path: str,
    template_path: str,
    output_path: str,
    failed_path: str,
    ratio: float,
) -> None:
    """
    End-to-end verbalization using all-column templates.

    Args:
        data_path:     Path to input JSON data file.
        template_path: Path to CSV templates file.
        output_path:   Path to save successful verbalized items.
        failed_path:   Path to save failed items.
        ratio:         Fraction of rows per table to verbalize into text.
    """
    skipped_verbalize_tables = 0
    no_template_match = 0

    template_df = pd.read_csv(template_path)
    raw_data: List[Dict[str, Any]] = read_json(data_path)

    template_sets = build_template_sets(template_df)

    data_new_ret: List[Dict[str, Any]] = []
    failed_items: List[Dict[str, Any]] = []

    for item in tqdm(raw_data, total=len(raw_data), desc="Processing items"):
        db = item.get("db", "UNKNOWN")
        sample_id = item.get("id_", "")
        table_names = item.get("table_names", [])
        tables = item.get("tables", [])

        new_item = copy.deepcopy(item)      # final output object
        tables_new = copy.deepcopy(tables)  # new_item["tables"]
        raw_tables_new = copy.deepcopy(tables)  # for raw_tables
        used_keys_map: Dict[str, List[str]] = {}

        # Verbalize each table in this item
        for idx, table_name in enumerate(
            tqdm(table_names, total=len(table_names), desc="Verbalizing tables", leave=False)
        ):
            raw_df = table_dict_to_df(tables[idx])

            # Fix known typo in specific schema: "Appelation" → "Appellation"
            if "Appelation" in raw_df.columns:
                raw_df.rename(columns={"Appelation": "Appellation"}, inplace=True)
            tables[idx]["table_columns"] = [
                "Appellation" if col == "Appelation" else col
                for col in tables[idx]["table_columns"]
            ]

            df_columns = set(raw_df.columns)
            chosen = pick_template_set(db, table_name, template_sets, df_columns)
            if not chosen:
                no_template_match += 1
                continue

            placeholders = chosen["columns"]
            templates = chosen["templates"]

            if not templates:
                continue

            semi_table, used_keys, paragraph, raw_table_wo_na = verbalize_table(
                df=raw_df,
                placeholders=placeholders,
                templates=templates,
                db=db,
                table_name=table_name,
                sample_id=sample_id,
                ratio=ratio,
            )

            if not paragraph:
                skipped_verbalize_tables += 1
                continue

            # Save semi-structured table
            tables_new[idx] = df_to_dict_table(semi_table)
            tables_new[idx]["paragraph"] = paragraph

            # Save raw table without NaNs for SQL
            raw_tables_new[idx] = df_to_dict_table(raw_table_wo_na)

            # Track which columns were used
            used_keys_map[table_name] = sorted(used_keys)

        # Skip items that have no paragraph at all
        has_paragraph = any(t.get("paragraph") for t in tables_new)
        if not has_paragraph:
            continue

        new_item["tables"] = tables_new
        new_item["raw_tables"] = {
            "table_names": table_names,
            "tables": raw_tables_new,
        }
        new_item["used_keys"] = used_keys_map

        # Collect paragraphs for convenience
        verbalized_data = []
        for idx in range(len(table_names)):
            paragraph = tables_new[idx].get("paragraph", "").strip()
            if paragraph:
                verbalized_data.append({"text": paragraph})
        new_item["verbalized_data"] = verbalized_data

        # Re-run SQL to get the final answer
        sql = item.get("SQL", "")
        if any(tbl == "appellations" for tbl in table_names) and db == "wine_1":
            sql = re.sub(r"\bAppelation\b", "Appellation", sql)
        answer = execute_sql(sql, new_item["raw_tables"])
        new_item["answer"] = answer
        item["SQL"] = sql

        if (not answer) or any(pd.isna(v) for row in answer for v in row):
            failed_items.append(new_item)
            continue

        data_new_ret.append(new_item)

    # Save successful items
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_new_ret, f, indent=2, ensure_ascii=False)

    # Save failed items
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed_items, f, indent=2, ensure_ascii=False)

    print(f"Final verbalized samples: {len(data_new_ret)}")
    print(f"Skipped {skipped_verbalize_tables} tables (no row could be verbalized).")
    print(f"Failed items due to SQL result NA or empty: {len(failed_items)}")
    print(f"Skipped {no_template_match} tables due to no matching template.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verbalize tables into paragraphs using all-column templates.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input JSON data file.",
    )
    parser.add_argument(
        "--template_path",
        type=str,
        required=True,
        help="Path to CSV templates file (with columns column1..N and template1..5).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save verbalized JSON.",
    )
    parser.add_argument(
        "--failed_path",
        type=str,
        required=True,
        help="Path to save failed items JSON.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Fraction of rows per table to verbalize into text (default: 0.5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Make sampling reproducible
    random.seed(42)
    run(
        data_path=args.data_path,
        template_path=args.template_path,
        output_path=args.output_path,
        failed_path=args.failed_path,
        ratio=args.ratio,
    )