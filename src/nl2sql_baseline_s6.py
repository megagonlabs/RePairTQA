#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
--------------------------------
Purpose:
- Build an in-memory SQLite database from JSON that contains table schemas and data
- Use an LLM to generate SQL (SELECT-only)
- Execute locally and output results (no automatic LIMIT injection)
"""

import os, re, json, argparse, sqlite3
from typing import List, Dict, Any, Tuple, Optional

# ============ Utility functions ============

def detect_type(value: Any) -> str:
    """
    Best-effort detection of a SQLite column type for a given Python value.
    """
    if value is None:
        return "TEXT"
    if isinstance(value, bool):
        return "INTEGER"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    if isinstance(value, str):
        try:
            int(value)
            return "INTEGER"
        except:
            try:
                float(value)
                return "REAL"
            except:
                return "TEXT"
    return "TEXT"

def choose_sql_type(values: List[Any]) -> str:
    """
    Choose a column type (INTEGER/REAL/TEXT) based on a small sample of values.
    """
    samples = [v for v in values if v is not None][:10]
    if not samples:
        return "TEXT"
    has_real = any(isinstance(v, float) for v in samples)
    has_int  = any(isinstance(v, int) for v in samples)
    if has_real: return "REAL"
    if has_int:  return "INTEGER"
    types = {detect_type(v) for v in samples}
    if "REAL" in types: return "REAL"
    if "INTEGER" in types: return "INTEGER"
    return "TEXT"

def sanitize_sql(sql: str) -> str:
    """
    - Extract the content inside ```sql ... ``` if present
    - Enforce SELECT-only
    - Do not append LIMIT automatically
    - Normalize by appending a semicolon
    """
    fence = re.findall(r"```sql(.*?)```", sql, flags=re.S|re.I)
    if fence:
        sql = fence[0].strip()
    sql = sql.strip().strip(";")
    head = sql.lower().lstrip("(").strip()
    if not head.startswith("select"):
        raise ValueError("Only SELECT is allowed.")
    return sql + ";"

def extract_note(sql: str) -> Tuple[str, Optional[str]]:
    """
    Split out an optional HTML comment note appended as <!--note-->... at the end.
    Returns (pure_sql, note_or_none).
    """
    m = re.search(r"<!--note-->(.*)$", sql, flags=re.S|re.I)
    if m:
        note = m.group(1).strip()
        pure = re.sub(r"<!--note-->.*$", "", sql, flags=re.S|re.I).strip()
        return pure, note
    return sql, None

def schema_md_from_sqlite(conn: sqlite3.Connection, preview_rows: int = 3) -> str:
    """
    Render a compact Markdown-like schema with a short preview for each table.
    """
    def qident(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    cur = conn.cursor()
    try:
        tables = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        ).fetchall()
    except Exception as e:
        return f"_(Failed to list tables: {e})_"

    parts = []
    for (tname,) in tables:
        try:
            cols = cur.execute(f"PRAGMA table_info({qident(tname)})").fetchall()
        except Exception as e:
            parts.append(f"Table {tname}\n_(skipped: {e})_")
            continue

        if not cols:
            parts.append(f"Table {tname}\n_(no columns)_")
            continue

        header = [f"{c[1]} ({c[2]})" if c[2] else str(c[1]) for c in cols]

        try:
            lim = int(preview_rows)
            rows = cur.execute(f"SELECT * FROM {qident(tname)} LIMIT {lim};").fetchall()
        except Exception as e:
            parts.append(
                f"Table {tname}\n| " + " | ".join(header) + " |\n| " + " | ".join(['---'] * len(header)) + " |\n"
                f"_(preview failed: {e})_"
            )
            continue

        col_line = " | ".join(header)
        sep_line = " | ".join(["---"] * len(header))
        body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
        parts.append(f"Table {tname}\n| {col_line} |\n| {sep_line} |\n{body}")

    return "\n".join(parts) if parts else "_(No tables)_"


# ============ Build SQLite from JSON ============

def build_sqlite_from_sample(sample: Dict[str, Any], use_raw_table: bool=True) -> sqlite3.Connection:
    """
    Build an in-memory SQLite database from a JSON sample payload.
    When use_raw_table=True, expect 'raw_table'/'raw_tables' with table_names and tables.
    Otherwise, use top-level 'tables' and 'table_names'.
    """
    conn = sqlite3.connect(":memory:")
    cur  = conn.cursor()

    table_map: Dict[str, Tuple[List[str], List[List[Any]]]] = {}

    if use_raw_table:
        rt = sample.get("raw_table") or sample.get("raw_tables")
        if not rt:
            raise RuntimeError("Requested raw tables, but neither 'raw_table' nor 'raw_tables' exists.")
        names = rt["table_names"]
        for tname, tinfo in zip(names, rt["tables"]):
            cols = list(tinfo["table_columns"])
            rows = list(tinfo["table_content"])
            table_map[tname] = (cols, rows)
    else:
        names = sample["table_names"]
        for tname, tinfo in zip(names, sample["tables"]):
            cols = list(tinfo["table_columns"])
            rows = list(tinfo["table_content"])
            table_map[tname] = (cols, rows)

    for tname, (cols, rows) in table_map.items():
        col_types = []
        for j, cname in enumerate(cols):
            col_values = [r[j] if j < len(r) else None for r in rows]
            col_types.append(choose_sql_type(col_values))
        col_defs = ", ".join(f"\"{c}\" {t}" for c, t in zip(cols, col_types))

        try:
            cur.execute(f'CREATE TABLE "{tname}" ({col_defs});')
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower() or "no such column" in str(e).lower() or "syntax error" in str(e).lower():
                print(f"[WARN] Skipping table {tname}: {e}")
                continue
            else:
                raise

        if rows:
            ph = ", ".join(["?"] * len(cols))
            cur.executemany(f'INSERT INTO "{tname}" VALUES ({ph});', rows)

    conn.commit()
    return conn

# ============ Prompt construction ============

BASE_SYS_PROMPT = """You are a careful NL2SQL expert for SQLite.
Rules:
- Return a single SQL query that answers the question.
- Use ONLY provided tables/columns.
- Prefer explicit JOINs with ON.
- If ambiguous, make the least risky assumption and add a short <!--note--> comment.
- Use SQLite syntax.
- Output ONLY SQL inside one fenced block:
```sql
SELECT ...
```
"""

USER_PROMPT = """SQLite schema & preview:
{schema_md}

Question:
{question}

Constraints:
- Only SELECT is allowed.
- If dates like "last month" appear, assume today is {today}.
"""

def build_messages(schema_md: str, question: str) -> List[Dict[str, str]]:
    from datetime import date
    sys_prompt = BASE_SYS_PROMPT
    usr_prompt = USER_PROMPT.format(schema_md=schema_md, question=question, today=date.today().isoformat())
    return [{"role":"system","content":sys_prompt},{"role":"user","content":usr_prompt}]

# ============ LLM invocation (OpenAI example, replaceable) ============

def call_llm(messages: List[Dict[str,str]], model: str, api_key: Optional[str]=None) -> str:
    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_BETA")
    if not api_key:
        raise RuntimeError("No API key found. Pass --openai_api_key or set OPENAI_API_KEY env var.")
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Missing 'openai' package. pip install openai") from e

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
    return resp.choices[0].message.content or ""

# ============ Single self-refinement (optional) ============

def refine_with_error(messages: List[Dict[str,str]], sql: str, error_msg: str, model: str, api_key: Optional[str]) -> str:
    assist_sql = f"```sql\n{sql.strip()}\n```"
    messages2 = messages + [
        {"role":"assistant","content":assist_sql},
        {"role":"user","content":f"Your SQL failed with this SQLite error:\n{error_msg}\n\nPlease return a corrected SQL (one fenced block). Remember: only SELECT."}
    ]
    return call_llm(messages2, model=model, api_key=api_key)

# ============ Main flow ============

def derive_prediction(rows, columns):
    """
    Derive a simplified prediction value from query rows if it is a single scalar cell.
    Otherwise, return the rows as-is.
    """
    try:
        if rows is None:
            return None
        if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], (list, tuple)) and len(rows[0]) == 1:
            return rows[0][0]
        return rows
    except Exception:
        return rows


try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

def sample_id(sample):
    """
    Try a set of common keys to extract a sample identifier.
    """
    return sample.get("id_") or sample.get("id") or sample.get("sid") or sample.get("ID") or sample.get("qid")

def sample_gt(sample):
    """
    Try a set of common keys to extract ground truth.
    """
    if "ground_truth" in sample:
        return sample["ground_truth"]
    for k in ["gt", "answer", "answers", "gold", "label", "labels", "GroundTruth", "gold_answer"]:
        if k in sample:
            return sample[k]
    return None


def run_once(sample: Dict[str,Any], model: str, api_key: Optional[str], allow_refine: bool=True, use_raw_table: bool=True) -> Dict[str,Any]:
    conn = build_sqlite_from_sample(sample, use_raw_table=use_raw_table)
    schema_md = schema_md_from_sqlite(conn, preview_rows=3)
    question  = sample.get("Question") or sample.get("question") or ""
    messages  = build_messages(schema_md, question)

    llm_text = call_llm(messages, model=model, api_key=api_key)
    pure_sql, note = extract_note(llm_text)

    try:
        sql_exec = sanitize_sql(pure_sql)
    except Exception as e:
        return {
            "id": sample_id(sample),
            "question": question,
            "ground_truth": sample_gt(sample),
            "sql": pure_sql.strip(),
            "note": note,
            "rows": [],
            "columns": [],
            "prediction": pure_sql.strip(),
            "error": str(e)
        }

    cur = conn.cursor()
    try:
        cur.execute(sql_exec)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return {
            "id": sample_id(sample),
            "question": question,
            "ground_truth": sample_gt(sample),
            "sql": pure_sql.strip(),
            "note": note,
            "rows": rows,
            "columns": cols,
            "prediction": rows
        }
    except Exception as e:
        if allow_refine:
            try:
                llm_text2 = refine_with_error(messages, pure_sql, str(e), model=model, api_key=api_key)
                pure_sql2, note2 = extract_note(llm_text2)
                sql_exec2 = sanitize_sql(pure_sql2)
                cur.execute(sql_exec2)
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                return {
                    "id": sample_id(sample),
                    "question": question,
                    "ground_truth": sample_gt(sample),
                    "sql": pure_sql2.strip(),
                    "note": note2,
                    "rows": rows,
                    "columns": cols,
                    "prediction": rows
                }
            except Exception as e2:
                return {
                    "id": sample_id(sample),
                    "question": question,
                    "ground_truth": sample_gt(sample),
                    "sql": pure_sql.strip(),
                    "note": note,
                    "rows": [],
                    "columns": [],
                    "prediction": pure_sql.strip(),
                    "error": f"Refine failed: {e2} (original error: {e})"
                }
        else:
            return {
                "id": sample_id(sample),
                "question": question,
                "ground_truth": sample_gt(sample),
                "sql": pure_sql.strip(),
                "note": note,
                "rows": [],
                "columns": [],
                "prediction": pure_sql.strip(),
                "error": str(e)
            }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", type=str, required=True, help="Path to a JSON file containing a list of samples")
    p.add_argument("--index", type=int, default=None, help="(Optional) Only run the specified sample index; if omitted, run all")
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    p.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API Key (or use env var OPENAI_API_KEY)")
    p.add_argument("--no_refine", action="store_true", help="Disable self-refinement on failure")
    p.add_argument("--no_raw", action="store_true", help="Use top-level 'tables' instead of 'raw_table'")
    p.add_argument("--output_file", type=str, default=None, help="Output JSONL path when processing all samples (default <input>_results.jsonl)")
    args = p.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError("Input JSON must be a non-empty list.")

    if args.index is not None:
        sample = data[args.index]
        result = run_once(
            sample,
            model=args.model,
            api_key=args.openai_api_key,
            allow_refine=(not args.no_refine),
            use_raw_table=(not args.no_raw),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    out_path = args.output_file or (os.path.splitext(args.input_json)[0] + "_results.jsonl")
    n_ok, n_fail = 0, 0
    with open(out_path, "w", encoding="utf-8") as w:
        for idx, sample in enumerate(tqdm(data, desc="NL2SQL", unit="sample")):
            result = run_once(
                sample,
                model=args.model,
                api_key=args.openai_api_key,
                allow_refine=(not args.no_refine),
                use_raw_table=(not args.no_raw),
            )
            result["_index"] = idx
            w.write(json.dumps(result, ensure_ascii=False) + "\n")
            if "error" in result and result["error"]:
                n_fail += 1
            else:
                n_ok += 1

    print(json.dumps({
        "Input file": args.input_json,
        "Output file": out_path,
        "Processed samples": len(data),
        "Succeeded": n_ok,
        "Failed": n_fail
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
