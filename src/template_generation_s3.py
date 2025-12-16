#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill empty template1–template5 fields in a table template file using an LLM.

Two modes:
- Default: read an existing CSV of column combinations (column1, column2, ...)
  and fill template1–template5 columns in place.
- --all_columns: ignore the combination CSV and instead use all columns from the
  matched-table CSV (matched + unmatched) to create a new CSV with
  column1–columnN + template1–template5.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

DEFAULT_MODEL = "gpt-4o-2024-11-20"


def call_gpt(
    client: OpenAI,
    prompt: str,
    model: str,
    max_tokens: int = 512,
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> str:
    """
    Call the OpenAI Chat Completion API with basic retry logic.

    Returns the text content of the first choice, or "ERROR" if all retries fail.
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

    return "ERROR"


def clean_json_markdown(raw: str) -> str:
    """
    Remove fenced markdown code block markers (``` or ```json) around JSON text.

    This function keeps only the inner JSON-like content if code fences are present.
    """
    text = raw.strip()
    # Remove leading ```json or ``` (case-insensitive)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Remove trailing ```
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def build_example_block_with_context(
    selected_columns: List[str],
    all_examples: Dict[str, List[str]],
    max_examples: int = 3,
) -> str:
    """
    Build a text block with column examples for the prompt.

    Each line has the format:
        [SELECTED] col_name: example1, example2, example3
    for selected columns, or:
        col_name: example1, example2, example3
    for non-selected columns.
    """
    lines: List[str] = []

    for col_name, example_list in all_examples.items():
        examples = example_list[:max_examples]
        if not examples:
            continue
        prefix = "[SELECTED] " if col_name in selected_columns else ""
        lines.append(f"{prefix}{col_name}: {', '.join(examples)}")

    return "\n".join(lines)


def parse_columns_field(field: Any) -> List[str]:
    """
    Parse a column-list field that may be:
    - a JSON-encoded list (e.g. '["col_a", "col_b"]'), or
    - a comma-separated string (e.g. 'col_a, col_b'), or
    - already a list.

    Returns a clean list of non-empty strings.
    """
    if field is None:
        return []

    # Handle NaN from pandas
    if isinstance(field, float) and pd.isna(field):
        return []

    # Already a list
    if isinstance(field, list):
        return [str(x).strip() for x in field if str(x).strip()]

    if isinstance(field, str):
        text = field.strip()
        if not text:
            return []

        # Try JSON first if it looks like JSON
        if (text.startswith("[") and text.endswith("]")) or (
            text.startswith("{") and text.endswith("}")
        ):
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()]
            except json.JSONDecodeError:
                # Fallback to comma-splitting
                pass

        # Fallback: treat as comma-separated
        return [part.strip() for part in text.split(",") if part.strip()]

    # Fallback for any other unexpected type
    return [str(field).strip()] if str(field).strip() else []



def load_examples_map(match_df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, List[str]]]:
    """
    Build a mapping:
        (db, table) -> { column_name: [example1, example2, ...] }
    from the 'column_examples' field in the matched-table CSV.
    """
    examples_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}

    for _, row in match_df.iterrows():
        db = str(row["db"])
        table = str(row["table"])
        key = (db, table)

        examples_raw = row.get("column_examples", "{}")

        # column_examples is usually a JSON string; be robust to dict as well.
        if isinstance(examples_raw, str):
            try:
                parsed = json.loads(examples_raw or "{}")
            except json.JSONDecodeError:
                parsed = {}
        elif isinstance(examples_raw, dict):
            parsed = examples_raw
        else:
            parsed = {}

        # Ensure values are lists of strings
        parsed_clean: Dict[str, List[str]] = {}
        for col_name, vals in parsed.items():
            if isinstance(vals, list):
                parsed_clean[col_name] = [str(v) for v in vals]
            else:
                parsed_clean[col_name] = [str(vals)]

        examples_map[key] = parsed_clean

    return examples_map


def ensure_template_columns(df: pd.DataFrame, num_templates: int = 5) -> None:
    """
    Ensure the DataFrame has template1..templateN columns, creating them if missing.
    """
    for i in range(1, num_templates + 1):
        col_name = f"template{i}"
        if col_name not in df.columns:
            df[col_name] = ""


def generate_templates(
    csv_templates: str,
    csv_matched: str,
    prompt_template_path: str,
    api_key: str,
    model: str,
    max_tokens: int,
    all_columns: bool,
) -> None:
    """
    Main entry for generating templates.

    Args:
        csv_templates: CSV with column combinations (or will be overwritten in-place).
        csv_matched:   CSV with column_examples information for each (db, table).
        prompt_template_path: path to the prompt template file.
        api_key:       OpenAI API key.
        model:         model name for LLM.
        max_tokens:    max tokens per call.
        all_columns:   whether to ignore combinations and use all columns from csv_matched.
    """
    tmpl_df = pd.read_csv(csv_templates)
    match_df = pd.read_csv(csv_matched)

    examples_map = load_examples_map(match_df)
    prompt_template = Path(prompt_template_path).read_text(encoding="utf-8")
    client = OpenAI(api_key=api_key)

    # Decide which dataframe to iterate over
    if all_columns:
        rows: Iterable[Tuple[int, pd.Series]] = match_df.iterrows()
        total_rows = len(match_df)
    else:
        rows = tmpl_df.iterrows()
        total_rows = len(tmpl_df)

    ensure_template_columns(tmpl_df, num_templates=5)

    output_rows: List[Dict[str, Any]] = []

    for idx, row in tqdm(rows, total=total_rows, desc="Generating templates"):
        db = str(row["db"])
        table = str(row["table"])
        key = (db, table)

        all_examples = examples_map.get(key, {})

        # Decide which columns should be verbalized for this row
        if all_columns:
            matched = parse_columns_field(row.get("matched_columns", ""))
            unmatched = parse_columns_field(row.get("unmatched_columns", ""))
            selected_columns = matched + unmatched
            if not selected_columns:
                # Fall back to any column that has examples
                selected_columns = list(all_examples.keys())
        else:
            selected_columns = [
                str(row[col_name])
                for col_name in row.index
                if col_name.startswith("column")
                and pd.notna(row[col_name])
                and str(row[col_name]).strip()
            ]

        if not selected_columns:
            print(f"No columns found for {db}.{table}, skipping.")
            continue

        example_block = build_example_block_with_context(
            selected_columns,
            all_examples,
            max_examples=3,
        )
        prompt = prompt_template.format(
            db=db,
            table=table,
            sel_cols=", ".join(selected_columns),
            example_block=example_block,
        )

        print(f"[{idx + 1}/{total_rows}] {db}.{table} → GPT")
        raw_output = call_gpt(
            client=client,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
        )

        clean_output = clean_json_markdown(raw_output)

        try:
            data = json.loads(clean_output)
            if not isinstance(data, dict):
                raise ValueError("Parsed JSON is not a dict")
        except Exception as exc:  # noqa: BLE001
            print(f"JSON parse failed for {db}.{table}: {exc}")
            data = {f"template{i}": "ERROR" for i in range(1, 6)}

        if all_columns:
            # Build a fresh row with db, table, column1..columnN, template1..template5
            out_row: Dict[str, Any] = {"db": db, "table": table}
            for i, col_name in enumerate(selected_columns, start=1):
                out_row[f"column{i}"] = col_name
            out_row.update(data)
            output_rows.append(out_row)
        else:
            # In-place filling of template1..template5 for the existing CSV
            for i in range(1, 6):
                key_name = f"template{i}"
                tmpl_df.at[idx, key_name] = data.get(key_name, "ERROR")

    # Persist results
    if all_columns:
        if not output_rows:
            print("No rows generated in --all_columns mode; no CSV written.")
            return

        out_df = pd.DataFrame(output_rows)

        # Determine the maximum number of columnN fields across all rows
        max_col_index = 0
        for record in output_rows:
            for key in record.keys():
                if key.startswith("column"):
                    try:
                        index = int(key.replace("column", ""))
                    except ValueError:
                        continue
                    max_col_index = max(max_col_index, index)

        ordered_columns: List[str] = (
            ["db", "table"]
            + [f"column{i}" for i in range(1, max_col_index + 1)]
            + [f"template{i}" for i in range(1, 6)]
        )

        # Ensure all ordered columns exist
        for col_name in ordered_columns:
            if col_name not in out_df.columns:
                out_df[col_name] = ""

        out_df = out_df[ordered_columns]

        output_path = csv_templates.replace(".csv", "_all_columns.csv")
        out_df.to_csv(output_path, index=False)
        print(f"Templates saved to {output_path}")
    else:
        tmpl_df.to_csv(csv_templates, index=False)
        print(f"Templates saved to {csv_templates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate table templates using an LLM.",
    )
    parser.add_argument(
        "--csv_templates",
        type=str,
        required=True,
        help="CSV file with table columns (to be filled).",
    )
    parser.add_argument(
        "--csv_matched",
        type=str,
        required=True,
        help="CSV file with column_examples.",
    )
    parser.add_argument(
        "--prompt_template_path",
        type=str,
        default="./prompts/template_generation.txt",
        help="Prompt template file path.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="OpenAI API key.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Max tokens for each LLM call.",
    )
    parser.add_argument(
        "--all_columns",
        action="store_true",
        help="If set, ignore existing column1..columnN in csv_templates and "
        "build combinations from all columns (matched + unmatched) in csv_matched.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_templates(
        csv_templates=args.csv_templates,
        csv_matched=args.csv_matched,
        prompt_template_path=args.prompt_template_path,
        api_key=args.api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        all_columns=args.all_columns,
    )


if __name__ == "__main__":
    main()