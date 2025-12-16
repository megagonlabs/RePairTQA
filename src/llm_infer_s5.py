#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run LLM-based QA over table / semi-structured data.

Features
--------
- Support three data views:
  * semi-structured: tables + verbalized text
  * structured: tables only
  * unstructured: verbalized text only
- Support two providers:
  * OpenAI / compatible via `openai` (provider="gpt")
  * Gemini via LiteLLM (provider="gemini")
- Evaluate Exact Match (EM) and partial match (PM) against ground truth.

Expected input JSON schema (per sample)
---------------------------------------
{
  "id_": "optional-id",
  "Question": "...",
  "answer": <any>,  # string/list/dict etc.
  "tables": [
    {
      "table_columns": [...],
      "table_content": [[...], ...]
    },
    ...
  ],
  "table_names": [...],
  "verbalized_data": [{"text": "..."}],
  "raw_tables": {
    "table_names": [...],
    "tables": [
      {"table_columns": [...], "table_content": [[...], ...]},
      ...
    ]
  }
}
"""

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import tiktoken
from openai import OpenAI
from tqdm import tqdm
from litellm import completion  # Gemini via LiteLLM

MAX_INPUT_TOKENS = 128000


def markdown_table_text(
    multi_row_cols: List[Any],
    rows: List[List[Any]],
    table_name: Optional[str] = None,
) -> str:
    """
    Render a table as GitHub-flavored Markdown.

    multi_row_cols:
        Either a flat list of column names
        or a list-of-lists for multi-row headers.
    rows:
        List of rows, each row is a list of cell values.
    table_name:
        Optional title shown above the table.
    """
    title = f"### Table: {table_name}\n" if table_name else ""

    # Multi-row header: [[..., ...], [..., ...], ...]
    if isinstance(multi_row_cols[0], list):
        num_cols = len(multi_row_cols[0])
        header_lines = []
        for header_row in multi_row_cols:
            line = "| " + " | ".join(str(c) if c != "" else " " for c in header_row) + " |"
            header_lines.append(line)
    else:
        num_cols = len(multi_row_cols)
        header_lines = ["| " + " | ".join(str(c) for c in multi_row_cols) + " |"]

    # Separator
    sep = "| " + " | ".join(["---"] * num_cols) + " |"

    # Body
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]

    return "\n".join([title] + header_lines + [sep] + body)


def format_tables_markdown(
    cols_list: List[Any],
    rows_list: List[List[List[Any]]],
    table_names: Optional[List[str]] = None,
) -> str:
    """Format multiple tables as Markdown blocks separated by blank lines."""
    names = table_names or [None] * len(cols_list)
    return "\n\n".join(
        markdown_table_text(c, r, name)
        for c, r, name in zip(cols_list, rows_list, names)
    )


def build_prompt_md(
    cols_list: List[Any],
    rows_list: List[List[List[Any]]],
    verbal_texts: List[str],
    question: str,
    table_names: Optional[List[str]] = None,
) -> str:
    """
    (Unused in main pipeline but kept as a utility.)
    Build a prompt that includes tables + question.
    """
    tables_md = format_tables_markdown(cols_list, rows_list, table_names)
    _ = "\n".join(verbal_texts)  # currently ignored in this helper
    return (
        "You are an expert at table question answering. You need to extract answers "
        "based on the following information:\n\n"
        "[Tables]\n"
        f"{tables_md}\n\n"
        "[Question]\n"
        f"{question}\n\n"
        "Please return your answer in JSON format only, with no explanation, "
        'following this structure:\n{"Answer": "[your answer]"}'
    )


def estimate_prompt_tokens(
    cols_list: List[Any],
    rows_list: List[List[List[Any]]],
    verbal_texts: List[str],
    question: str,
    data_type: str = "semi-structured",
    model: str = "gpt-4o",
    table_names: Optional[List[str]] = None,
) -> Tuple[int, str]:
    """
    Build a prompt for a given data view and estimate its token length.

    data_type:
        - "semi-structured": tables + verbal text
        - "structured":      tables only
        - "unstructured":    verbal text only
    model:
        Name used by tiktoken to choose the encoding. For non-OpenAI models
        we default to "gpt-4o" as an approximation.
    """
    # Normalize model name for tiktoken
    encoding_model = model
    if encoding_model == "o4-mini-2025-04-16":
        encoding_model = "o4-mini"
    # For safety, treat unknown models as gpt-4o
    try:
        enc = tiktoken.encoding_for_model(encoding_model)
    except KeyError:
        enc = tiktoken.encoding_for_model("gpt-4o")

    tables_md = format_tables_markdown(cols_list, rows_list, table_names)
    verbal = " ".join(verbal_texts)

    semi_structured_prompt = (
        "You are an expert at table question answering. You need to extract answers "
        "based on the following information:\n\n"
        "[Tables]\n"
        f"{tables_md}\n\n"
        "[Additional Information]\n"
        f"{verbal}\n\n"
        "[Question]\n"
        f"{question}\n\n"
        "Please return your answer in JSON format only, with no explanation, "
        'following this structure:\n{"Answer": "[your answer]"}'
    )

    structured_prompt = (
        "You are an expert at table question answering. You need to extract answers "
        "based on the following information:\n\n"
        "[Tables]\n"
        f"{tables_md}\n\n"
        "[Question]\n"
        f"{question}\n\n"
        "Please return your answer in JSON format only, with no explanation, "
        'following this structure:\n{"Answer": "[your answer]"}'
    )

    unstructured_prompt = (
        "You are an expert at question answering. You need to extract answers "
        "based on the following information:\n\n"
        "[Textual Information]\n"
        f"{verbal}\n\n"
        "[Question]\n"
        f"{question}\n\n"
        "Please return your answer in JSON format only, with no explanation, "
        'following this structure:\n{"Answer": "[your answer]"}'
    )

    if data_type == "semi-structured":
        prompt = semi_structured_prompt
    elif data_type == "structured":
        prompt = structured_prompt
    elif data_type == "unstructured":
        prompt = unstructured_prompt
    else:
        raise ValueError(f"Invalid data type: {data_type}")

    return len(enc.encode(prompt)), prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM QA inference with different data types and formats.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSON file with question and table data.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save main output (for chosen data_type).",
    )
    parser.add_argument(
        "--structured_output_file",
        type=str,
        default=None,
        help=(
            "Optional path to save a structured-only baseline output. "
            "If provided, the script will run an additional pass using raw_tables."
        ),
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="structured",
        choices=["semi-structured", "structured", "unstructured"],
        help="Data view for the main run: tables + text, tables only, or text only.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model name passed to the provider (OpenAI or Gemini via LiteLLM).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick debugging).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gpt",
        choices=["gpt", "gemini", "custom"],
        help=(
            "Backend provider:\n"
            "  gpt    : use OpenAI client (requires OPENAI_API_KEY)\n"
            "  gemini : use LiteLLM with a Gemini model (requires GEMINI_API_KEY)\n"
            "  custom : currently treated the same as 'gpt' (for compatibility)"
        ),
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for the selected provider (overrides env var if set).",
    )
    return parser.parse_args()


def extract_answer(resp_text: str) -> Tuple[str, Optional[str]]:
    """
    Extract the answer from a JSON-only response.

    The model is instructed to return:
        {"Answer": "..."}
    but we are robust to minor deviations and fenced ```json blocks.
    """
    try:
        raw = resp_text.strip()
        if raw.startswith("```json"):
            raw = raw.removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(raw)
        for key in ("Answer", "answer"):
            if key in parsed:
                return str(parsed[key]).strip().lower(), None
        return raw, "extract_failed"
    except Exception:
        return resp_text.strip(), "extract_failed"


def flatten_list(lst: Iterable[Any]) -> Iterable[Any]:
    for item in lst:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item


def normalize(x: Any) -> str:
    """
    Normalize a value to a lowercase string for EM/PM comparison.
    """
    if isinstance(x, list):
        flat = list(flatten_list(x))
        return " ".join(map(str, flat)).lower()
    if isinstance(x, dict):
        return json.dumps(x, sort_keys=True).lower()
    if isinstance(x, str):
        return x.lower()
    return str(x).lower()


def eval_em_pm(pred: Any, gt: Any) -> Tuple[int, int]:
    """
    Exact-match (EM) and partial-match (PM) metrics.

    EM = 1 if normalized(pred) == normalized(gt), else 0.
    PM = 1 if one normalized string is a substring of the other, else 0.
    """
    p = normalize(pred)
    g = normalize(gt)
    em = int(p == g)
    pm = int(p in g or g in p)
    return em, pm


_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    """Create or reuse a global OpenAI client (for provider='gpt')."""
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when --provider gpt/custom")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def call_gpt_and_eval(
    prompt: str,
    gt: Any,
    model: str = "gpt-4o-2024-11-20",
) -> Tuple[Optional[str], Optional[str], int, int]:
    """
    Call OpenAI-compatible /chat/completions and compute EM/PM.

    Uses tiktoken to skip overly long prompts.
    """
    # Use gpt-4o encoding as an approximation for token counting
    enc = tiktoken.encoding_for_model("gpt-4o")
    tok_len = len(enc.encode(prompt))
    if tok_len > MAX_INPUT_TOKENS:
        msg = f"SKIPPED: prompt too long ({tok_len} tokens)"
        return msg, "prompt_too_long", 0, 0

    max_retries, delay = 5, 2
    pred: Optional[str] = None
    error: Optional[str] = None

    for attempt in range(max_retries):
        try:
            client = _get_openai_client()
            if model == "o4-mini-2025-04-16":
                # New-style API uses `max_completion_tokens`
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant for table question answering.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=4096,
                )
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant for table question answering.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=4096,
                )

            raw_text = resp.choices[0].message.content or ""
            pred, error = extract_answer(raw_text)
            break
        except Exception as exc:  # noqa: BLE001
            if "rate limit" in str(exc).lower() and attempt < max_retries - 1:
                print(f"Rate limit hit, retrying in {delay}s")
                time.sleep(delay)
                delay *= 2
            else:
                pred, error = None, str(exc)
                break

    em, pm = (0, 0) if error else eval_em_pm(pred, gt)
    return pred, error, em, pm


def call_litellm_and_eval(
    prompt: str,
    gt: Any,
    model: str = "gemini/gemini-2.5-flash",
) -> Tuple[Optional[str], Optional[str], int, int]:
    """
    Call a Gemini model via LiteLLM's /chat/completions-compatible interface.

    Requires GEMINI_API_KEY to be set in the environment.
    """
    response_format: Dict[str, Any] = {
        "type": "json_object",
        "response_schema": {
            "type": "object",
            "properties": {"Answer": {"type": "string"}},
            "required": ["Answer"],
        },
    }

    max_retries, delay = 5, 2
    pred: Optional[str] = None
    error: Optional[str] = None

    for attempt in range(max_retries):
        try:
            resp = completion(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for table question answering.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=4096,
                response_format=response_format,
            )
            raw_text = resp.choices[0].message.content or ""
            pred, error = extract_answer(raw_text)
            break
        except Exception as exc:  # noqa: BLE001
            if "rate limit" in str(exc).lower() and attempt < max_retries - 1:
                print(f"Rate limit hit (Gemini), retrying in {delay}s")
                time.sleep(delay)
                delay *= 2
            else:
                pred, error = None, str(exc)
                break

    em, pm = (0, 0) if error else eval_em_pm(pred, gt)
    return pred, error, em, pm


def main() -> None:
    args = parse_args()

    # --- NEW: handle explicit API key override ---
    if args.api_key:
        if args.provider in ("gpt", "custom"):
            os.environ["OPENAI_API_KEY"] = args.api_key
        elif args.provider == "gemini":
            os.environ["GEMINI_API_KEY"] = args.api_key

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Choose backend based on provider
    if args.provider == "gemini":
        infer = lambda p, g: call_litellm_and_eval(p, g, model=args.model)
    else:
        # "gpt" and "custom" are both treated as OpenAI-style providers
        infer = lambda p, g: call_gpt_and_eval(p, g, model=args.model)

    samples = data if args.max_samples is None else data[: args.max_samples]

    em_total = 0
    pm_total = 0
    processed = 0

    # Clean up previous outputs if they exist
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    if args.structured_output_file and os.path.exists(args.structured_output_file):
        os.remove(args.structured_output_file)

    # If structured_output_file is not given, we write the baseline to /dev/null
    dummy_path = os.devnull
    with open(args.output_file, "w", encoding="utf-8") as fout_main, \
        (open(args.structured_output_file, "w", encoding="utf-8")
        if args.structured_output_file
        else open(dummy_path, "w", encoding="utf-8")) as fout_struct:

        for sample in tqdm(samples, desc="Evaluating"):
            question = sample["Question"]
            gt = sample["answer"]
            sid = sample.get("id_", None)
            if gt is None:
                continue

            verbal = [v["text"] for v in sample.get("verbalized_data", [])]
            cols_list = [t["table_columns"] for t in sample.get("tables", [])]
            rows_list = [t["table_content"] for t in sample.get("tables", [])]
            table_names = sample.get("table_names", None)

            # ---- Main run (configured data_type) ----
            _, prompt = estimate_prompt_tokens(
                cols_list,
                rows_list,
                verbal,
                question,
                data_type=args.data_type,
                model="gpt-4o",  # encoding model only; provider uses args.model
                table_names=table_names,
            )

            pred, error, em, pm = infer(prompt, gt)
            fout_main.write(
                json.dumps(
                    {
                        "id": sid,
                        "question": question,
                        "ground_truth": gt,
                        "prediction": pred,
                        "error": error,
                        "EM": em,
                        "PM": pm,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fout_main.flush()

            em_total += em
            pm_total += pm
            processed += 1
            print(f"{processed} | EM={em} PM={pm}")

            # ---- Structured baseline (raw tables only) ----
            if args.structured_output_file:
                raw_tables = sample.get("raw_tables", {})
                table_names_raw = raw_tables.get("table_names", [])
                table_entries = raw_tables.get("tables", [])

                if len(table_names_raw) != len(table_entries):
                    print(f"Raw table count mismatch for sample id={sid}")
                    continue

                raw_cols_list = [t["table_columns"] for t in table_entries]
                raw_rows_list = [t["table_content"] for t in table_entries]

                _, prompt_s = estimate_prompt_tokens(
                    raw_cols_list,
                    raw_rows_list,
                    [],
                    question,
                    data_type="structured",
                    model="gpt-4o",
                    table_names=table_names_raw,
                )
                pred_s, error_s, em_s, pm_s = infer(prompt_s, gt)
                fout_struct.write(
                    json.dumps(
                        {
                            "id": sid,
                            "question": question,
                            "ground_truth": gt,
                            "prediction": pred_s,
                            "error": error_s,
                            "EM": em_s,
                            "PM": pm_s,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fout_struct.flush()


if __name__ == "__main__":
    main()