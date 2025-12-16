#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use an LLM to (1) classify table columns and (2) propose useful column combinations.

Input:
    - A CSV produced by the table/column matching script
      (must contain: db, table, matched_columns, unmatched_columns, column_examples).

Outputs:
    - csv_cols_out: one row per (db, table, column) with a label.
    - csv_tmpl_out: one row per (db, table) with column1 ~ columnN as a combination template.
    - failures.csv (optional): raw LLM outputs that could not be parsed.
"""

import argparse
import json
import re
import time
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-2024-11-20"
FAILURE_CSV = "failures.csv"


def parse_llm_output(output_str: str) -> Tuple[Dict[str, str], List[List[str]]]:
    """
    Parse LLM output into:
      - yes/no labels for each column (column_classification)
      - suggested column combinations (combinations).

    Expected JSON structure (example):

    ```json
    {
      "column_classification": {
        "col_a": "YES",
        "col_b": "NO"
      },
      "combinations": [
        ["col_a", "col_c"],
        ["col_a", "col_d", "col_e"]
      ]
    }
    ```

    The function first tries to extract a ```json ... ``` fenced block.
    If not found, it attempts to parse the whole string as JSON.
    """
    # Try to find a fenced ```json ... ``` block
    match = re.search(r"```json\s*(.*?)\s*```", output_str, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        print("No ```json``` block found — trying to parse the entire output.")
        cleaned = output_str.strip()

    try:
        parsed = json.loads(cleaned)
    except Exception as exc:  # noqa: BLE001
        print("JSON parsing failed:", exc)
        return {}, []

    # Be robust to different shapes: dict at top-level, or list of dicts, etc.
    if isinstance(parsed, list) and parsed:
        # If it's a list, try the first element
        parsed = parsed[0]

    if not isinstance(parsed, dict):
        print("Parsed JSON is not a dict; got type:", type(parsed))
        return {}, []

    yesno_dict = parsed.get("column_classification", {}) or {}
    combinations = parsed.get("combinations", []) or []

    # Ensure correct types
    if not isinstance(yesno_dict, dict):
        print("column_classification is not a dict in parsed JSON.")
        yesno_dict = {}

    if not isinstance(combinations, list):
        print("combinations is not a list in parsed JSON.")
        combinations = []

    # Normalize labels to strings
    yesno_dict = {str(k): str(v) for k, v in yesno_dict.items()}

    # Normalize combinations to list[list[str]]
    normalized_combinations: List[List[str]] = []
    for comb in combinations:
        if isinstance(comb, list):
            normalized_combinations.append([str(c) for c in comb if isinstance(c, (str, int))])

    return yesno_dict, normalized_combinations


def load_prompt_template(path: str) -> str:
    """Load a plain-text prompt template from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def create_client(api_key: str) -> OpenAI:
    """Create an OpenAI client with the given API key."""
    return OpenAI(api_key=api_key)


def call_gpt(
    client: OpenAI,
    prompt: str,
    model: str,
    retries: int = 3,
    max_tokens: int = 2048,
    sleep_seconds: float = 2.0,
) -> str:
    """
    Call the OpenAI Chat Completion API with basic retry logic.

    Returns the text content from the first choice, or an empty string if all retries fail.
    """
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            message = response.choices[0].message
            return (message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            print(f"GPT error on attempt {attempt}/{retries}: {exc}")
            time.sleep(sleep_seconds)

    return ""


def build_column_list_for_prompt(
    row: pd.Series,
    only_query_cols: bool,
    exclude_query_cols: bool,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Build the list of columns (with examples) that will be shown in the prompt.

    Returns:
        - all_cols: the selected column names in the order to be presented.
        - column_examples: mapping from column name to example strings.
    """
    matched_cols = [
        c.strip()
        for c in str(row.get("matched_columns", "")).split(",")
        if c and c.strip()
    ]

    unmatched_raw = row.get("unmatched_columns")
    if isinstance(unmatched_raw, float) and pd.isna(unmatched_raw):
        unmatched_cols: List[str] = []
    else:
        unmatched_cols = [
            c.strip()
            for c in str(unmatched_raw).split(",")
            if c and c.strip()
        ]

    if only_query_cols:
        all_cols = matched_cols
    elif exclude_query_cols:
        all_cols = unmatched_cols
    else:
        # Use all columns but keep them sorted and unique
        all_cols = sorted(set(matched_cols + unmatched_cols))

    # Parse column_examples JSON safely
    examples_raw = row.get("column_examples", "{}")
    if isinstance(examples_raw, str):
        try:
            column_examples = json.loads(examples_raw)
        except json.JSONDecodeError:
            print("Failed to parse column_examples JSON; falling back to empty dict.")
            column_examples = {}
    elif isinstance(examples_raw, dict):
        column_examples = examples_raw
    else:
        column_examples = {}

    # Ensure values are lists of strings
    column_examples = {
        col: [str(v) for v in (vals or [])]
        for col, vals in column_examples.items()
    }

    return all_cols, column_examples


def build_prompt(
    template: str,
    db: str,
    table: str,
    all_cols: List[str],
    column_examples: Dict[str, List[str]],
) -> str:
    """
    Fill the prompt template with database, table, and column information.

    Each column will be rendered as:
        col_name (example1, example2, example3)
    if examples exist, otherwise just col_name.
    """
    cols_with_examples: List[str] = []
    for col in all_cols:
        examples = column_examples.get(col, [])
        if examples:
            cols_with_examples.append(f"{col} ({', '.join(examples)})")
        else:
            cols_with_examples.append(col)

    return template.format(
        db=db,
        table=table,
        cols="\n".join(cols_with_examples),
    )


def expand_combinations_to_rows(raw_combinations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Expand combination records into a flat table:

        db, table, column1, column2, ..., columnN

    where N is the maximum combination length.
    """
    max_cols = max((len(r["columns"]) for r in raw_combinations), default=0)
    tmpl_rows: List[Dict[str, Any]] = []

    for record in raw_combinations:
        row: Dict[str, Any] = {"db": record["db"], "table": record["table"]}
        for i in range(max_cols):
            key = f"column{i + 1}"
            row[key] = record["columns"][i] if i < len(record["columns"]) else ""
        tmpl_rows.append(row)

    return pd.DataFrame(tmpl_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify table columns and generate column combinations using an LLM.",
    )
    parser.add_argument("--csv_in", required=True, help="Input CSV file.")
    parser.add_argument(
        "--csv_cols_out",
        required=True,
        help="Output CSV path for per-column labels.",
    )
    parser.add_argument(
        "--csv_tmpl_out",
        required=True,
        help="Output CSV path for column combinations.",
    )
    parser.add_argument(
        "--prompt_template_path",
        required=True,
        help="Path to the prompt template (e.g. ./prompts/column_selection.txt).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--api_key",
        required=True,
        help="OpenAI API key.",
    )
    parser.add_argument(
        "--query_column",
        action="store_true",
        help="Use only matched_columns (columns that appear in SQL).",
    )
    parser.add_argument(
        "--exclude_query_column",
        action="store_true",
        help="Use only unmatched_columns (columns that never appear in SQL).",
    )

    args = parser.parse_args()

    # Basic argument sanity check
    if args.query_column and args.exclude_query_column:
        parser.error(
            "Cannot use --query_column and --exclude_query_column at the same time.",
        )

    return args


def main() -> None:
    args = parse_args()

    client = create_client(args.api_key)
    prompt_template = load_prompt_template(args.prompt_template_path)
    df = pd.read_csv(args.csv_in)

    col_rows: List[Dict[str, Any]] = []
    raw_combinations: List[Dict[str, Any]] = []
    fail_rows: List[Dict[str, Any]] = []

    total_rows = len(df)

    for idx, row in df.iterrows():
        db = str(row["db"])
        table = str(row["table"])
        key = f"{db}|{table}"

        all_cols, column_examples = build_column_list_for_prompt(
            row=row,
            only_query_cols=args.query_column,
            exclude_query_cols=args.exclude_query_column,
        )

        if not all_cols:
            print(f"[{idx + 1}/{total_rows}] No columns to classify for {key}, skipping.")
            continue

        prompt = build_prompt(
            template=prompt_template,
            db=db,
            table=table,
            all_cols=all_cols,
            column_examples=column_examples,
        )

        print(f"[{idx + 1}/{total_rows}] Classifying columns for {key}")
        llm_output = call_gpt(client, prompt, args.model)

        yesno_dict, combinations = parse_llm_output(llm_output)

        if not yesno_dict and not combinations:
            print(f"Failed to parse GPT output for {key}")
            fail_rows.append({"db": db, "table": table, "output": llm_output})
            continue

        # Save per-column labels
        for col, label in yesno_dict.items():
            col_rows.append(
                {
                    "db": db,
                    "table": table,
                    "column": col,
                    "label": label,
                },
            )

        # Keep only combinations whose columns are labeled YES
        yes_cols = {c for c, l in yesno_dict.items() if l.upper() == "YES"}
        filtered_combinations = [
            comb
            for comb in combinations
            if comb and all(col in yes_cols for col in comb)
        ]

        for comb in filtered_combinations:
            raw_combinations.append(
                {
                    "db": db,
                    "table": table,
                    "columns": comb,
                },
            )

    # Write column labels
    pd.DataFrame(col_rows).to_csv(args.csv_cols_out, index=False)

    # Expand combinations to column1 ~ columnN
    tmpl_df = expand_combinations_to_rows(raw_combinations)
    tmpl_df.to_csv(args.csv_tmpl_out, index=False)

    # Save failures, if any
    if fail_rows:
        pd.DataFrame(fail_rows).to_csv(FAILURE_CSV, index=False)

    print(
        f"Finished.\n"
        f"Column labels saved to: {args.csv_cols_out}\n"
        f"Templates saved to:     {args.csv_tmpl_out}",
    )
    if fail_rows:
        print(f"{len(fail_rows)} failed cases saved to: {FAILURE_CSV}")


if __name__ == "__main__":
    main()