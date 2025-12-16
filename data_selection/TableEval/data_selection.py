#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build S2 split from TableEval (simple lookup, short, flat tables)
=================================================================

This script constructs the S2 diagnostic split used in the paper:

  - Only keep samples with:
      * sub_task_name == "简单查询"  (simple lookup queries)
      * table_structure_type NOT in {"嵌套表格", "层次表格"}  (flat tables)
      * number of data rows within [min_rows, max_rows]
      * a valid golden answer

  - Convert markdown tables into:
      tables = [
        {
          "table_columns": [...],
          "table_content": [... rows ...]
        }
      ]

  - Output format: a JSON list of samples:
      {
        "id_": ...,
        "Question": ...,
        "answer": ...,
        "table_names": [table_name],
        "tables": [...],
        "sub_task_name": ...,
        "table_structure_type": ...,
        "num_rows": <number of data rows>
      }

Usage example:

  python tableeval_build_S2.py \
      --meta_file TableEval-meta.jsonl \
      --test_file TableEval-test.jsonl \
      --out TableEval_S2_simple_short_flat.json \
      --min_rows 0 \
      --max_rows 100
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_table_markdown(markdown: str):
    """
    Parse a markdown table into (header, rows).

    The expected format is the usual pipe-separated markdown:

      col1 | col2 | col3
      ---- | ---- | ----
      v11  | v12  | v13
      v21  | v22  | v23
      ...

    Returns:
      header: list[str]
      rows:   list[list[str]]
    """
    lines = markdown.strip().split("\n")
    if len(lines) < 2:
        return [], []

    # First line: header
    header = [h.strip() for h in lines[0].strip("|").split("|")]

    rows = []
    # Skip the separator line (index 1)
    for line in lines[2:]:
        row = [cell.strip() for cell in line.strip("|").split("|")]
        if len(row) == len(header):
            rows.append(row)

    return header, rows


def build_S2(args):
    meta_path = Path(args.meta_file).resolve()
    test_path = Path(args.test_file).resolve()
    out_path = Path(args.out).resolve()

    print(f"[CFG] meta_file = {meta_path}")
    print(f"[CFG] test_file = {test_path}")
    print(f"[CFG] out       = {out_path}")
    print(f"[CFG] min_rows  = {args.min_rows}")
    print(f"[CFG] max_rows  = {args.max_rows}")
    print(f"[CFG] simple_subtask_name   = {args.simple_subtask_name}")
    print(f"[CFG] excluded_struct_types = {args.excluded_struct_types}")

    # Load meta and test
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = [json.loads(line) for line in f]

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    meta_dict = {m["table_id"]: m for m in meta_data}

    # For statistics
    struct_type_counter = Counter()
    kept_counter = 0

    samples = []

    for item in test_data:
        # 1) Only simple lookup sub-task
        if item.get("sub_task_name") != args.simple_subtask_name:
            continue

        table_id = item.get("table_id")
        meta = meta_dict.get(table_id)
        if not meta:
            continue

        struct_type = meta.get("table_structure_type", "")
        struct_type_counter[struct_type] += 1

        # 2) Exclude nested / hierarchical tables
        if struct_type in args.excluded_struct_types:
            continue

        # 3) Extract golden answer (we only keep the first one for now)
        try:
            golden = item["golden_answer_list"]
            answer = golden[0]["问题列表"][0]["最终答案"][0]
        except (KeyError, IndexError, TypeError):
            continue

        # 4) Parse markdown table into columns and rows
        columns, content = parse_table_markdown(meta.get("table_markdown", ""))
        if not columns or not content:
            continue

        num_rows = len(content)
        if num_rows < args.min_rows:
            continue
        if args.max_rows is not None and num_rows > args.max_rows:
            continue

        # Table name: context before and after
        pre_list = meta.get("pre_context_list", [])
        post_list = meta.get("post_context_list", [])
        table_name = "".join(pre_list + post_list).strip()

        question = ""
        if item.get("question_list"):
            question = item["question_list"][0]

        sample = {
            "id_": item["id"],
            "Question": question,
            "answer": answer,
            "table_names": [table_name],
            "tables": [
                {
                    "table_columns": columns,
                    "table_content": content,
                }
            ],
            "sub_task_name": item.get("sub_task_name", ""),
            "table_structure_type": struct_type,
            "num_rows": num_rows,
        }
        samples.append(sample)
        kept_counter += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Written {kept_counter} samples → {out_path}")
    print("\n[Table structure type distribution in test_data]")
    total = sum(struct_type_counter.values())
    for k, v in struct_type_counter.most_common():
        pct = 100.0 * v / total if total else 0.0
        print(f"  {k:<10}: {v:>5} ({pct:.2f}%)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build S2 split from TableEval (simple lookup, short, flat tables)."
    )
    parser.add_argument("--meta_file", type=str, required=True,
                        help="Path to TableEval-meta.jsonl")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to TableEval-test.jsonl")
    parser.add_argument("--out", type=str, required=True,
                        help="Output JSON file path.")
    parser.add_argument("--min_rows", type=int, default=0,
                        help="Minimum number of data rows to keep (inclusive).")
    parser.add_argument("--max_rows", type=int, default=100,
                        help="Maximum number of data rows to keep (inclusive).")
    parser.add_argument("--simple_subtask_name", type=str, default="简单查询",
                        help="Sub-task name used to indicate simple lookup queries.")
    parser.add_argument(
        "--excluded_struct_types",
        nargs="*",
        default=["嵌套表格", "层次表格"],
        help="Table structure types to exclude (e.g., nested / hierarchical tables).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_S2(args)