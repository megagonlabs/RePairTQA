#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verbalize table rows using pre-defined templates.

Pipeline:
1. Load raw table data (MMQA / BIRD / WikiSQL-style JSON).
2. Load per-(db, table) column + template combinations from a CSV.
3. For each table:
   - Pick one valid (columns, templates) combination whose columns all exist in the DataFrame.
   - Drop rows with NaN in any placeholder column.
   - Generate a natural-language description _description by filling a random template
     for each remaining row.
   - Save the semi-structured table and the "raw table without NaNs".
4. (Optional) Re-run the original SQL on the cleaned raw tables to recompute the answer.

Outputs:
- A JSON file with updated tables, raw_tables, used_keys, and possibly updated answer.
- A JSON file containing failed items (e.g., empty answers or answers containing NaN).
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

from utils import df_to_dict_table, read_json, table_dict_to_df


# ----------------------------------------------------------------------
# Template / verbalization helpers
# ----------------------------------------------------------------------


def extract_template_columns(template: str) -> List[str]:
    """Extract all {placeholders} from a template string."""
    return re.findall(r"{(.*?)}", template)


def verbalize_table_row(
    template: str,
    row: pd.Series,
    placeholders: List[str],
    db: str | None = None,
    table_name: str | None = None,
    sample_id: str | None = None,
) -> str:
    """
    Fill a single template with values from a row.

    Any missing column/value will be reported to stdout, but the function will
    still return a (partially filled) template string.
    """
    filled = template
    for key in placeholders:
        if key in row.index:
            value: Any = row[key]
            # Handle possible nested pandas objects defensively
            if isinstance(value, (pd.Series, pd.DataFrame)):
                value = value.iloc[0] if hasattr(value, "iloc") and not value.empty else None

            if pd.notna(value):
                filled = filled.replace(f"{{{key}}}", str(value))
            else:
                print(
                    f"[Missing value] DB={db} | Table={table_name} | Sample={sample_id} | "
                    f"Key={key} | Value={value}"
                )
        else:
            print(
                f"[Missing column] DB={db} | Table={table_name} | Sample={sample_id} | "
                f"Key={key} not found in columns"
            )
    return filled


def verbalize_table(
    df: pd.DataFrame,
    placeholders: List[str],
    templates: List[str],
    db: str | None = None,
    table_name: str | None = None,
    sample_id: str | None = None,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Verbalize a table by:
      - Dropping rows with NaN in any placeholder column.
      - Generating one description per row using a random template.
      - Returning:
          * semi_table: original df without placeholder columns + '_description' column
          * used_keys: the placeholder list (for bookkeeping)
          * df_clean: the filtered DataFrame without NaNs on placeholders

    Raises:
        KeyError: if one or more placeholder columns do not exist in df.
    """
    try:
        df_clean = df.dropna(subset=placeholders).copy()
    except KeyError as exc:
        missing_cols = list(exc.args[0])

        print("\n=== COLUMN MISMATCH DETECTED ===")
        print(f"DB:          {db}")
        print(f"Table:       {table_name}")
        print(f"Sample ID:   {sample_id}")
        print(f"Placeholders: {placeholders}")
        print(f"DF columns:   {list(df.columns)}")
        print(f"Missing cols: {missing_cols}")
        print("================================\n")

        # Either re-raise or choose to skip this table. For now, we re-raise to be explicit.
        raise

    if df_clean.empty:
        # No rows left after dropping NaNs; return an empty semi_table with description.
        semi_table = df.drop(columns=[c for c in placeholders if c in df.columns]).copy()
        semi_table["_description"] = []
        return semi_table, placeholders, df_clean

    verbalized_rows: List[str] = []
    for _, row in df_clean.iterrows():
        template = random.choice(templates)
        text = verbalize_table_row(
            template=template,
            row=row,
            placeholders=placeholders,
            db=db,
            table_name=table_name,
            sample_id=sample_id,
        )
        verbalized_rows.append(text)

    semi_table = df_clean.drop(columns=placeholders).copy()
    semi_table["_description"] = verbalized_rows
    return semi_table, placeholders, df_clean


def pick_template_set(
    db: str,
    table_name: str,
    template_sets: Dict[Tuple[str, str], List[Dict[str, Any]]],
    df_columns: set,
) -> Dict[str, Any] | None:
    """
    Pick a single template set for a given (db, table_name).

    A template set has the structure:
        {
            "columns":   [col1, col2, ...],
            "templates": [tpl1, tpl2, ...],
        }

    Only template sets with columns fully contained in df_columns are considered valid.
    If no valid template set exists, returns None.
    """
    candidates = template_sets.get((db, table_name), [])
    valid = [c for c in candidates if set(c["columns"]).issubset(df_columns)]
    return random.choice(valid) if valid else None



def execute_sql(query: str, raw_tables: Dict[str, Any]) -> List[List[Any]]:
    """
    Execute an SQL query against a collection of tables in memory.

    Strategy:
    1. Try DuckDB (fast, flexible).
    2. On failure, fall back to in-memory SQLite.
    3. If both fail, print errors and return [].

    Args:
        query:       SQL string to be executed.
        raw_tables:  A dict with keys "table_names" and "tables",
                     where each table is a dict with "table_columns" and "table_content".

    Returns:
        A list of result rows (each row is a list/tuple), or [] on error.
    """
    table_names = raw_tables["table_names"]
    tables = raw_tables["tables"]

    def _register_duck(conn: duckdb.DuckDBPyConnection) -> None:
        for name, tbl in zip(table_names, tables):
            df = pd.DataFrame(tbl["table_content"], columns=tbl["table_columns"])
            conn.register(name, df)

    def _register_sqlite(conn) -> None:
        import sqlite3  # local import to avoid hard dependency at module import time

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

        # Collect non-empty columnN fields
        columns = [
            row[col_name]
            for col_name in row.index
            if col_name.startswith("column")
            and pd.notna(row[col_name])
            and str(row[col_name]).strip()
        ]

        # Collect non-empty templateN fields
        templates = [
            row[f"template{i}"]
            for i in range(1, 6)
            if f"template{i}" in row
            and pd.notna(row[f"template{i}"])
            and str(row[f"template{i}"]).strip()
        ]

        if columns and templates:
            template_sets.setdefault(key, []).append(
                {"columns": list(columns), "templates": list(templates)}
            )

    return template_sets


def main(args: argparse.Namespace) -> None:
    """
    End-to-end verbalization pipeline.

    Steps:
        1) Load templates and raw data.
        2) Build per-(db, table) template sets.
        3) For each item:
           - For each table:
             * Pick a template set.
             * Verbalize rows into _description.
             * Save semi-structured table + raw table without NaNs.
           - Optionally re-run SQL on cleaned raw_tables to recompute answer.
        4) Save successful items and failed items separately.
    """
    data_path = args.data_path
    template_path = args.template_path
    output_path = args.output_path
    failed_path = args.failed_path

    # Load templates and raw data
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

        # Work on deep copies to avoid mutating the original item
        new_item = copy.deepcopy(item)
        tables_new = copy.deepcopy(tables)
        raw_tables_new = copy.deepcopy(tables)
        used_keys_map: Dict[str, List[str]] = {}

        # Verbalize each table in the sample
        for idx, table_name in enumerate(
            tqdm(table_names, desc="Verbalizing tables", leave=False)
        ):
            raw_df = table_dict_to_df(tables[idx])

            # Fix a known typo in a specific schema ("Appelation" → "Appellation")
            if "Appelation" in raw_df.columns:
                raw_df.rename(columns={"Appelation": "Appellation"}, inplace=True)
            tables[idx]["table_columns"] = [
                "Appellation" if col == "Appelation" else col
                for col in tables[idx]["table_columns"]
            ]

            df_columns = set(raw_df.columns)
            chosen = pick_template_set(db, table_name, template_sets, df_columns)
            if not chosen:
                # No valid template set for this table; skip verbalization
                continue

            placeholders = chosen["columns"]
            templates = chosen["templates"]
            if not templates:
                continue

            try:
                semi_table, used_keys, raw_table_wo_na = verbalize_table(
                    df=raw_df,
                    placeholders=placeholders,
                    templates=templates,
                    db=db,
                    table_name=table_name,
                    sample_id=sample_id,
                )
            except KeyError:
                # If columns are inconsistent, skip this table
                failed_items.append(copy.deepcopy(new_item))
                continue

            tables_new[idx] = df_to_dict_table(semi_table)
            raw_tables_new[idx] = df_to_dict_table(raw_table_wo_na)
            used_keys_map[table_name] = sorted(used_keys)

        # Update item with new tables and raw_tables
        new_item["tables"] = tables_new
        new_item["raw_tables"] = {
            "table_names": table_names,
            "tables": raw_tables_new,
        }
        new_item["used_keys"] = used_keys_map

        # Either keep original answer or recompute via SQL
        answer = new_item.get("answer", None)

        if not args.no_sql:
            sql = item.get("SQL", "")
            # Fix specific schema typo for queries involving the "appellations" table
            if any(tbl == "appellations" for tbl in table_names) and db == "wine_1":
                sql = re.sub(r"\bAppelation\b", "Appellation", sql)

            answer = execute_sql(sql, new_item["raw_tables"])
            new_item["answer"] = answer
            item["SQL"] = sql  # keep the fixed SQL in the original item as well

        # Filter out items with empty answers or NaNs in answers
        if (not answer) or any(
            pd.isna(v) for row in answer for v in (row if isinstance(row, (list, tuple)) else [row])
        ):
            failed_items.append(new_item)
            continue

        data_new_ret.append(new_item)

    # Save successful items
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_new_ret, f, indent=2, ensure_ascii=False)

    print(f"Failed items: {len(failed_items)}")

    # Save failed items separately
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed_items, f, indent=2, ensure_ascii=False)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verbalize tables from templates.")
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
        help="Path to CSV templates file.",
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
        "--no_sql",
        action="store_true",
        help="If set, skip SQL execution and keep the original answer field.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Make the sampling reproducible (template selection, etc.)
    random.seed(42)
    main(args)  