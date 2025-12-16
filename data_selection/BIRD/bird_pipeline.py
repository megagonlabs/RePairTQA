#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIRD Pipeline: Build splits + Execute SQL answers
=================================================

This script provides two sub-commands:

1) build
   - Construct BIRD subsets based on:
       * table length (short / long via row thresholds)
       * SQL type (lookup = SimpleProjection, compositional = non-SimpleProjection)
       * single-table only (default)
   - Output format is a JSON list, where each sample includes:
       * Question, SQL, db id, table_names, tables[ {table_columns, table_content} ],
         primary_keys, foreign_keys, sql_tags, sql_type.

2) answers
   - Take a JSON file produced by `build`.
   - Execute SQL over the in-memory tables using DuckDB (with fallback to SQLite).
   - Filter out non-meaningful answers.
   - Group by `sql_type` and sample at most N items per type.
   - Write a new JSON file with an added "answer" field.

Example usage:

  # Build a long-table + lookup (SimpleProjection) split (similar to S4)
  python bird_pipeline.py build \
      --bird_root /path/to/BIRD \
      --splits train,dev_20240627 \
      --min_rows 101 --sql_mode lookup --single_table_only \
      --out bird_S4_long_lookup.json

  # Add answers and sample 100 per sql_type
  python bird_pipeline.py answers \
      --in bird_S4_long_lookup.json \
      --out bird_S4_long_lookup_with_answer.json \
      --samples_per_type 100
"""

import argparse
import json
import math
import pickle
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pandas as pd
from sql_metadata import Parser
from tqdm import tqdm

# =========================
# Shared helpers
# =========================

def safe_json(obj):
    """Make objects JSON-safe (e.g., replace NaN/Inf with None)."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    return obj


# =========================
# Part 1: SQL classification & BIRD split construction
# =========================

_SQL_FROM = re.compile(r"\bfrom\s+([`\"\[]?\w+)", re.I)
_SQL_JOIN = re.compile(r"\bjoin\s+([`\"\[]?\w+)", re.I)

def extract_tables(sql: str):
    """
    Extract table names from SQL (FROM + JOIN clauses), deduplicated.
    """
    return list(
        {m.group(1).strip("`\"[]") for m in _SQL_FROM.finditer(sql)} |
        {m.group(1).strip("`\"[]") for m in _SQL_JOIN.finditer(sql)}
    )

def _is_simple_projection(query: str) -> bool:
    """
    Determine whether a query is a "simple lookup"/SimpleProjection.

    Rules:
      - Single table
      - No UNION / INTERSECT / EXCEPT / HAVING / GROUP BY / ORDER BY
      - Exactly one SELECT (no subqueries)
      - No JOIN
      - No aggregation (SUM/AVG/COUNT/MIN/MAX)
      - SELECT only contains column names or '*' (optionally DISTINCT)
      - WHERE (if present) is a conjunction (AND) of simple predicates:
          col = literal
          col LIKE literal
          col BETWEEN lit1 AND lit2
          col IN (literals)
    """
    q = " ".join(query.strip().split())
    q_up = q.upper()

    # 1) Disallow certain structures
    if any(k in q_up for k in [" UNION ", " INTERSECT ", " EXCEPT ", " HAVING ", " GROUP BY ", " ORDER BY "]):
        return False
    if q_up.count("SELECT") != 1:
        return False
    if re.search(r"\bJOIN\b", q_up):
        return False
    if re.search(r"\b(SUM|AVG|COUNT|MIN\s*\(|MAX\s*\()\b", q_up):
        return False

    # 2) Single table
    tbls = extract_tables(q_up)
    if len(tbls) != 1:
        return False

    # 3) SELECT list: SELECT ... FROM
    m = re.search(r"\bSELECT\b(.*?)\bFROM\b", q_up, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return False
    sel = m.group(1).strip()

    # Allow DISTINCT
    sel = re.sub(r"^\s*DISTINCT\s+", "", sel, flags=re.IGNORECASE)

    # Only allow column names (possibly with table prefix) and '*'
    col_token = r'(?:\*|[A-Z_][A-Z0-9_]*\.[A-Z_][A-Z0-9_]*|[A-Z_][A-Z0-9_]*|`[^`]+`|"[^"]+"|\[[^\]]+\])'
    if not re.fullmatch(rf'\s*{col_token}(?:\s*,\s*{col_token})*\s*', sel, flags=re.IGNORECASE):
        return False

    # 4) WHERE clause (optional)
    m_where = re.search(r"\bWHERE\b(.*?)(?:\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|$)",
                        q, flags=re.IGNORECASE | re.DOTALL)
    if not m_where:
        return True

    where = m_where.group(1).strip()
    if not where:
        return True

    # Only AND-connected predicates, no OR
    if re.search(r"\bOR\b", where, flags=re.IGNORECASE):
        return False

    lit = r"(?:'[^']*'|\"[^\"]*\"|[+-]?\d+(?:\.\d+)?)"
    col = r"(?:[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*|`[^`]+`|\"[^\"]+\"|\[[^\]]+\])"

    simple_atom = rf"(?:{col}\s*=\s*{lit}|{col}\s+LIKE\s+{lit}|{col}\s+BETWEEN\s+{lit}\s+AND\s+{lit}|{col}\s+IN\s*\(\s*{lit}(?:\s*,\s*{lit})*\s*\))"
    simple_where = rf"^\s*{simple_atom}(?:\s+AND\s+{simple_atom})*\s*$"

    return re.fullmatch(simple_where, where, flags=re.IGNORECASE) is not None


def classify_sql_query(query: str):
    """
    Return a list of tags describing the SQL query.

    Possible tags:
      - 'Aggregation'
      - 'Max/Min'
      - 'Join'
      - 'Ranking'
      - 'Comparison'
      - 'SimpleProjection'   (our lookup/simple-query class)
    """
    parser = Parser(query)
    tokens = [str(tok).upper() for tok in parser.tokens]
    tags = []

    # Aggregation
    if any(tok in {"SUM", "AVG", "COUNT", "MIN(", "MAX("} for tok in tokens):
        tags.append("Aggregation")
    # Max/Min (including LIMIT 1)
    if "MAX" in tokens or "MIN" in tokens or ("LIMIT" in tokens and "1" in tokens[tokens.index("LIMIT") + 1:]):
        tags.append("Max/Min")
    # Joins
    if any(tok in {"JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN", "CROSS JOIN"} for tok in tokens):
        tags.append("Join")
    # Ranking
    if "ORDER BY" in " ".join(tokens):
        tags.append("Ranking")

    # Comparison (outside JOIN conditions)
    comp_ops = {">", "<", ">=", "<=", "=", "!=", "<>", "BETWEEN", "IN"}
    numeric_re = re.compile(r"^[+-]?\d+(\.\d+)?$")
    inside_join = False
    join_starters = {"ON", "USING"}
    clause_breakers = {"WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"}

    for i, tok in enumerate(tokens):
        if tok in join_starters:
            inside_join = True
        elif tok in clause_breakers:
            inside_join = False
        elif tok in comp_ops and not inside_join:
            before_num = (i and numeric_re.match(tokens[i - 1]))
            after_num = (i + 1 < len(tokens) and numeric_re.match(tokens[i + 1]))
            if before_num or after_num:
                tags.append("Comparison")
                break

    # If no tag has been assigned yet, check if it's a SimpleProjection
    if not tags and _is_simple_projection(query):
        tags.append("SimpleProjection")

    return tags


def build_db_map(bird_root: Path):
    """
    Index all .sqlite files under the BIRD root.

    Returns:
      dict: {db_id: sqlite_path}
    """
    db_map = {}
    for p in bird_root.rglob("*.sqlite"):
        if p.name.startswith("._"):
            continue
        db_map[p.stem] = p
    print(f"[DB] indexed {len(db_map)} sqlite files")
    return db_map


def build_row_cache(db_map, cache_path: Path):
    """
    Count rows for each (db.table) and cache the result on disk.

    Cache format:
      { "db_id.table_name": row_count }
    """
    if cache_path.exists():
        try:
            data = pickle.load(open(cache_path, "rb"))
            if isinstance(data, dict) and data:
                print(f"[ROWS] using cached table rows from {cache_path}")
                return data
        except Exception:
            print("[ROWS] cache load error, rebuilding…")

    table_rows = {}
    print("[ROWS] counting rows (this may be slow on first run)…")
    for db_id, db_path in db_map.items():
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            for (tbl,) in cur.fetchall():
                key = f"{db_id}.{tbl}"
                try:
                    cur.execute(f'SELECT COUNT(*) FROM "{tbl}"')
                    table_rows[key] = cur.fetchone()[0]
                except sqlite3.Error:
                    # If something goes wrong, set a huge row count to avoid filtering by accident
                    table_rows[key] = 10**9
            conn.close()
        except sqlite3.DatabaseError:
            continue

    pickle.dump(table_rows, open(cache_path, "wb"))
    print(f"[ROWS] cached {len(table_rows)} tables to {cache_path}")
    return table_rows


def load_tables_meta(split_dir: Path):
    """
    Load table meta information from {split_prefix}_tables.json.

    Returns:
      dict: {
        db_id: {
          "table_names": [...],
          "column_names": [...],
          "primary_keys": [...],
          "foreign_keys": [...]
        }
      }
    """
    meta = {}
    tbl_file = split_dir / f"{split_dir.name.split('_')[0]}_tables.json"
    objs = json.load(open(tbl_file))
    for obj in objs:
        db_id = obj["db_id"]
        meta[db_id] = {
            "table_names": obj["table_names_original"],
            "column_names": obj["column_names_original"],
            "primary_keys": obj["primary_keys"],
            "foreign_keys": obj["foreign_keys"],
        }
    return meta


def get_columns_for_table(meta_db, table):
    """Return a list of column names for the given table in a db."""
    if not meta_db:
        return []
    t_names, col_names = meta_db["table_names"], meta_db["column_names"]
    if table not in t_names:
        return []
    idx = t_names.index(table)
    return [col for t_idx, col in col_names if t_idx == idx]


def fetch_table_content(db_path: Path, table: str, row_count: int, row_sample_cap: int):
    """
    Fetch table content from a sqlite db.

    If the table is larger than row_sample_cap, we randomly sample at most
    row_sample_cap rows; otherwise we return the full table.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        if row_count > row_sample_cap:
            cur.execute(f'SELECT * FROM "{table}" ORDER BY RANDOM() LIMIT {row_sample_cap}')
        else:
            cur.execute(f'SELECT * FROM "{table}"')
        rows = cur.fetchall()
    except sqlite3.Error:
        rows = []
    conn.close()
    return rows


def cmd_build(args):
    """
    Sub-command: build

    Construct splits from BIRD, filtered by:
      - min_rows / max_rows
      - sql_mode (any / lookup / compositional)
      - single_table_only
    """
    bird_root = Path(args.bird_root).resolve()
    splits = [sp.strip() for sp in args.splits.split(",") if sp.strip()]

    out_path = Path(args.out).resolve()
    row_cache_path = Path(args.row_cache).resolve()

    print(f"[CFG] bird_root          = {bird_root}")
    print(f"[CFG] splits             = {splits}")
    print(f"[CFG] min_rows           = {args.min_rows}")
    print(f"[CFG] max_rows           = {args.max_rows}")
    print(f"[CFG] sql_mode           = {args.sql_mode}")
    print(f"[CFG] single_table_only  = {args.single_table_only}")
    print(f"[CFG] row_sample_cap     = {args.row_sample_cap}")
    print(f"[CFG] row_cache          = {row_cache_path}")
    print(f"[CFG] out                = {out_path}")

    db_map = build_db_map(bird_root)
    rows_map = build_row_cache(db_map, row_cache_path)

    all_samples = []
    global_label_cnt = Counter()
    global_total = 0

    for sp in splits:
        split_dir = bird_root / sp
        tables_meta = load_tables_meta(split_dir)
        data_file = split_dir / f"{sp.split('_')[0]}.json"
        data = json.load(open(data_file))

        label_cnt = Counter()
        reasons = Counter()

        print(f"\n[SPLIT] {sp}, {len(data)} items in {data_file}")
        for idx, itm in enumerate(tqdm(data, desc=f"[{sp}]")):
            sql = itm["SQL"]
            db_id = itm["db_id"]

            # 1) Single-table / multi-table filter
            tbls = extract_tables(sql)
            if args.single_table_only and len(tbls) != 1:
                reasons["multi-table SQL"] += 1
                continue
            if not tbls:
                reasons["no-table SQL"] += 1
                continue
            tbl = tbls[0]

            # 2) Row count filter
            key = f"{db_id}.{tbl}"
            row_count = rows_map.get(key, 0)
            if row_count < args.min_rows:
                reasons["rows < min_rows"] += 1
                continue
            if args.max_rows is not None and row_count > args.max_rows:
                reasons["rows > max_rows"] += 1
                continue

            # 3) SQL tags
            tags = classify_sql_query(sql)
            if not tags:
                reasons["no-tag"] += 1
                continue

            # 4) Filter by sql_mode
            keep = False
            if args.sql_mode == "any":
                keep = True
            elif args.sql_mode == "lookup":
                # Only pure SimpleProjection
                keep = (len(tags) == 1 and tags[0] == "SimpleProjection")
            elif args.sql_mode == "compositional":
                # At least one tag, and not pure SimpleProjection
                if len(tags) == 1 and tags[0] == "SimpleProjection":
                    keep = False
                else:
                    keep = True
            else:
                raise ValueError(f"Unknown sql_mode: {args.sql_mode}")

            if not keep:
                reasons["filtered_by_sql_mode"] += 1
                continue

            # 5) Build output structure
            meta_db = tables_meta.get(db_id)
            columns = get_columns_for_table(meta_db, tbl)

            db_folder = "train_databases" if "train" in sp else "dev_databases"
            db_file = split_dir / db_folder / db_id / f"{db_id}.sqlite"
            table_rows = fetch_table_content(db_file, tbl, row_count, args.row_sample_cap)

            qid = itm.get("question_id", idx)
            if len(tags) == 1:
                sql_type = tags[0]
            else:
                sql_type = ",".join(tags)

            sample = {
                "id_": f"{sp}_{qid}",
                "Question": itm["question"],
                "SQL": sql,
                "sql_tags": tags,
                "sql_type": sql_type,  # for compatibility with downstream scripts
                "db": db_id,
                "difficulty": itm.get("difficulty", "unknown"),
                "table_names": [tbl],
                "tables": [{
                    "table_columns": columns,
                    "table_content": table_rows,
                }],
                "primary_keys": meta_db["primary_keys"] if meta_db else [],
                "foreign_keys": meta_db["foreign_keys"] if meta_db else [],
            }

            all_samples.append(sample)
            label_cnt[sql_type] += 1
            reasons["kept"] += 1

        print(f"\n[STATS] {sp}")
        print(f"  kept {sum(label_cnt.values())} / {len(data)}")
        for k, v in label_cnt.most_common():
            print(f"  sql_type={k:<25} : {v}")
        print("  filtered reasons:")
        for k, v in reasons.most_common():
            print(f"    {k:<24}: {v}")

        global_label_cnt.update(label_cnt)
        global_total += len(data)

    # Write out as JSON list
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[SAVE] {len(all_samples)} samples → {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, sample in enumerate(all_samples):
            line = json.dumps(safe_json(sample), ensure_ascii=False, indent=2)
            f.write(line)
            if i < len(all_samples) - 1:
                f.write(",\n")
        f.write("\n]\n")
    print("[DONE] build")

    print("\n[GLOBAL STATS]")
    print(f"  kept {sum(global_label_cnt.values())} / {global_total}")
    for k, v in global_label_cnt.most_common():
        print(f"  sql_type={k:<25} : {v}")


# =========================
# Part 2: Execute SQL, attach answers, sample per sql_type
# =========================

def execute_sql(query: str, raw_tables: dict):
    """
    Execute SQL over in-memory tables using DuckDB, then fallback to SQLite.

    raw_tables format:
      {
        "table_names": [...],
        "tables": [
          {
            "table_columns": [...],
            "table_content": [...]
          },
          ...
        ]
      }

    Returns:
      list[list]  - list of rows, each row is a list of cell values
      or []       - if execution failed or no result
    """
    table_names = raw_tables["table_names"]
    tables = raw_tables["tables"]

    def try_duckdb():
        """Primary backend: DuckDB."""
        def run(conn):
            try:
                result_df = conn.query(query).df()
                return result_df.values.tolist()
            except Exception:
                return None

        # First attempt: use original table/column names
        try:
            with duckdb.connect(":memory:") as conn:
                for name, tbl in zip(table_names, tables):
                    df = pd.DataFrame(tbl["table_content"], columns=tbl["table_columns"])
                    conn.register(name, df)
                result = run(conn)
                if isinstance(result, list):
                    return result
        except Exception:
            pass

        # Second attempt: lowercase table/column names and patch SQL
        try:
            with duckdb.connect(":memory:") as conn:
                for name, tbl in zip(table_names, tables):
                    df = pd.DataFrame(tbl["table_content"], columns=tbl["table_columns"])
                    df = df.dropna(axis=1, how="all")
                    df.columns = [str(c).lower() for c in df.columns]
                    conn.register(name.lower(), df)

                lowered_sql = re.sub(
                    r'(?i)\bFROM\s+([A-Za-z_][\w]*)',
                    lambda m: f'FROM {m.group(1).lower()}',
                    query
                )
                lowered_sql = re.sub(
                    r'(?i)\bJOIN\s+([A-Za-z_][\w]*)',
                    lambda m: f'JOIN {m.group(1).lower()}',
                    lowered_sql
                )
                result = run(conn)
                if isinstance(result, list):
                    return result
        except Exception:
            pass

        return None

    def try_sqlite():
        """Fallback backend: SQLite (less flexible, but sometimes works)."""
        try:
            conn = sqlite3.connect(":memory:")
            cur = conn.cursor()
            for name, tbl in zip(table_names, tables):
                cols = tbl["table_columns"]
                col_defs = ", ".join(f'"{col}" TEXT' for col in cols)
                cur.execute(f'CREATE TABLE "{name}" ({col_defs})')
                for row in tbl["table_content"]:
                    cur.execute(
                        f'INSERT INTO "{name}" VALUES ({",".join(["?"]*len(cols))})',
                        row
                    )
            result = cur.execute(query).fetchall()
            conn.close()
            return result
        except Exception as e:
            print(f"[SQLite Fallback Error] {e}\nSQL: {query}")
            return []

    result = try_duckdb()
    if isinstance(result, list):
        return result
    return try_sqlite()


def is_meaningful(ans):
    """
    Decide whether an answer is meaningful enough to keep.

    Heuristics:
      - Non-empty result set.
      - At least one non-empty / non-NaN value.
      - Ignore trivial single value [0].
    """
    if not ans:
        return False
    flat = [x for row in ans for x in row]
    if all(x in (None, "") or (isinstance(x, float) and math.isnan(x)) for x in flat):
        return False
    if len(flat) == 1 and flat[0] == 0:
        return False
    return True


def cmd_answers(args):
    """
    Sub-command: answers

    Take a JSON file produced by `build`, execute SQL to obtain answers,
    filter out non-meaningful results, then sample at most N items per sql_type.
    """
    in_path = Path(args.in_file).resolve()
    out_path = Path(args.out).resolve()
    samples_per_type = args.samples_per_type

    print(f"[ANS] input  = {in_path}")
    print(f"[ANS] output = {out_path}")
    print(f"[ANS] samples_per_type = {samples_per_type}")

    data = json.load(open(in_path, encoding="utf-8"))
    grouped = defaultdict(list)

    # Execute SQL and attach answers
    for itm in tqdm(data, desc="Exec SQL"):
        raw_tables = {
            "table_names": itm["table_names"],
            "tables": itm["tables"],
        }
        ans = execute_sql(itm["SQL"], raw_tables)
        if is_meaningful(ans):
            itm["answer"] = ans
            sql_type = itm.get("sql_type", "UNKNOWN")
            grouped[sql_type].append(itm)

    # Sample within each sql_type
    final_samples = []
    import random
    for sql_type, items in grouped.items():
        if samples_per_type is not None and len(items) > samples_per_type:
            sampled = random.sample(items, samples_per_type)
        else:
            sampled = items
        final_samples.extend(sampled)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(safe_json(final_samples), f, ensure_ascii=False, indent=2)

    print(f"\n✅ Written {len(final_samples)} samples → {out_path}")

    type_counter = Counter()
    for item in final_samples:
        type_counter[item.get("sql_type", "UNKNOWN")] += 1

    print("\n[SQL Type Breakdown]")
    total_sql = sum(type_counter.values())
    for k, v in type_counter.most_common():
        pct = 100.0 * v / total_sql if total_sql else 0.0
        print(f"  {k:<20}: {v:>4}  ({pct:.2f}%)")


# =========================
# Argument parser
# =========================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="BIRD pipeline: build splits + execute SQL answers."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- build sub-command ----
    p_build = subparsers.add_parser("build", help="Construct BIRD splits by length + SQL type.")
    p_build.add_argument("--bird_root", type=str, default=".",
                         help="BIRD root directory (containing train/, dev_20240627/, etc.)")
    p_build.add_argument("--splits", type=str, default="train,dev_20240627",
                         help="Comma-separated splits, e.g., 'train,dev_20240627'")
    p_build.add_argument("--min_rows", type=int, default=0,
                         help="Minimum number of rows for a table (inclusive).")
    p_build.add_argument("--max_rows", type=int, default=None,
                         help="Maximum number of rows for a table (inclusive). Use None for no upper bound.")
    p_build.add_argument("--sql_mode", type=str, default="any",
                         choices=["any", "lookup", "compositional"],
                         help="SQL filter: any / lookup(SimpleProjection) / compositional(non-SimpleProjection).")
    p_build.add_argument("--single_table_only", action="store_true",
                         help="If set, keep only single-table SQL queries (len(tables)==1).")
    p_build.add_argument("--row_sample_cap", type=int, default=1000,
                         help="If a table is larger than this, randomly sample at most this many rows.")
    p_build.add_argument("--row_cache", type=str, default=".table_rows.pkl",
                         help="Path to the row-count cache file.")
    p_build.add_argument("--out", type=str, required=True,
                         help="Output JSON file path.")

    # ---- answers sub-command ----
    p_ans = subparsers.add_parser("answers", help="Execute SQL to attach answers and sample per sql_type.")
    p_ans.add_argument("--in", dest="in_file", type=str, required=True,
                       help="Input JSON file (produced by 'build').")
    p_ans.add_argument("--out", type=str, required=True,
                       help="Output JSON file with 'answer' field.")
    p_ans.add_argument("--samples_per_type", type=int, default=100,
                       help="Max number of samples to keep per sql_type. Use a large value to disable downsampling.")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    t0 = time.time()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "answers":
        cmd_answers(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    print(f"\nTime elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()