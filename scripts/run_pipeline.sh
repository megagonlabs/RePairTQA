#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Table-to-text pipeline (toy example)
#
# Steps:
#   1) Analyze table/SQL matches and column usage.
#   2) Use an LLM to classify columns and propose column combinations.
#   3) Use an LLM to generate natural-language templates for the combinations.
#   4) Verbalize tables into text using the generated templates.
#
# This script assumes the following Python entrypoints exist in ../src/:
#   - matched_columns_analysis_s1.py
#   - column_selection_s2.py
#   - template_generation_s3.py
#   - verbalization_col_s4.py
###############################################################################

#######################################
# Input configuration
#######################################

# Path to the input JSON data (your toy example).
# Make sure example.json is in the same directory as this script,
# or change the path accordingly.
DATA_JSON="../sample_data/example.json"

# OpenAI API key (recommended: export OPENAI_API_KEY in your shell instead of hard-coding)
#   export OPENAI_API_KEY="sk-xxxx"
API_KEY=${OPENAI_API_KEY}

if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Please export it before running this script."
  exit 1
fi

# Prompt templates
#   PROMPT_COL_SELECTION: used in column_selection_s2.py to decide which columns are important.
#   PROMPT_TEMPLATE_GEN:  used in template_generation_s3.py to generate natural language templates.
PROMPT_COL_SELECTION="../prompts/column_selection.txt"
PROMPT_TEMPLATE_GEN="../prompts/template_generation.txt"

#######################################
# Output configuration
#######################################

RUN_NAME="run_example"
OUT_DIR="../pipeline_results/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

echo "📂 All results will be saved to: ${OUT_DIR}"

###############################################################################
# Step 1: Analyze matched tables and columns
###############################################################################
# Script: matched_columns_analysis_s1.py
#
# Required args:
#   --input   : path to the input JSON file (DATA_JSON).
#   --output  : path to the CSV file storing table-level column stats.
#   --mode    : analysis mode, one of:
#               - table_match   : per-table matched/unmatched columns + examples.
#               - high_freq     : high-frequency vs low-frequency columns per table.
#               - unique_table  : counts of how often each table appears.
#
# Optional args (in the Python script):
#   --save_csv: for mode=unique_table, whether to write the CSV file.
###############################################################################
echo "🔍 Step 1: Analyzing matched tables and columns..."
python ../src/matched_columns_analysis_s1.py \
  --input  "${DATA_JSON}" \
  --output "${OUT_DIR}/example_table_match.csv" \
  --mode   table_match

###############################################################################
# Step 2: Column classification + combination generation
###############################################################################
# Script: column_selection_s2.py
#
# Required args:
#   --csv_in          : input CSV from Step 1 (table_match results).
#   --csv_cols_out    : output CSV with one row per (db, table, column, label).
#   --csv_tmpl_out    : output CSV with column combinations per (db, table).
#   --prompt_template_path :
#                        path to the prompt file used to classify columns and
#                        propose combinations.
#   --model           : OpenAI model name (e.g., gpt-4o-2024-11-20).
#   --api_key         : OpenAI API key string.
#
# Optional args:
#   --query_column       : if set, only use matched_columns (columns that appear in SQL).
#   --exclude_query_column :
#                          if set, only use unmatched_columns (columns that never appear in SQL).
#                          NOTE: cannot be used together with --query_column.
###############################################################################
echo "📝 Step 2: Classifying columns and generating combinations..."
python ../src/column_selection_s2.py \
  --csv_in               "${OUT_DIR}/example_table_match.csv" \
  --csv_cols_out         "${OUT_DIR}/example_col_labels.csv" \
  --csv_tmpl_out         "${OUT_DIR}/example_col_combinations.csv" \
  --prompt_template_path "${PROMPT_COL_SELECTION}" \
  --model                "gpt-4o-2024-11-20" \
  --api_key              "${API_KEY}"
  # Optional flags you might use later:
  # --query_column
  # --exclude_query_column

###############################################################################
# Step 3: Template generation
###############################################################################
# Script: template_generation_s3.py
#
# Required args:
#   --csv_templates        : input CSV with column combinations (from Step 2),
#                            or any CSV with db, table, column1..N.
#   --csv_matched          : CSV with table-level info from Step 1
#                            (used to fetch column_examples).
#   --prompt_template_path : path to the prompt file used to generate templates.
#   --api_key              : OpenAI API key.
#   --model                : OpenAI model name.
#
# Optional args:
#   --max_tokens           : max tokens for each LLM call (default: 512).
#   --all_columns          : if set, ignore the combinations in csv_templates and
#                            instead use *all* columns (matched + unmatched) from
#                            the matched CSV to build new combinations.
#                            The output file will then typically be
#                            ${csv_templates%.csv}_all_columns.csv
###############################################################################
echo "✏️ Step 3: Generating natural language templates..."
python ../src/template_generation_s3.py \
  --csv_templates        "${OUT_DIR}/example_col_combinations.csv" \
  --csv_matched          "${OUT_DIR}/example_table_match.csv" \
  --prompt_template_path "${PROMPT_TEMPLATE_GEN}" \
  --api_key              "${API_KEY}" \
  --model                "gpt-4o-2024-11-20" \
  --max_tokens           512
  # If you want templates that cover *all* columns for each table, use:
  # --all_columns

###############################################################################
# Step 4: Verbalize tables using the generated templates
###############################################################################
# Script: verbalization_col_s4.py
#
# Required args:
#   --data_path     : path to the original JSON dataset (same as DATA_JSON).
#   --template_path : path to the CSV with templates and column combinations
#                     (usually the output of Step 3).
#   --output_path   : path to save the verbalized JSON data.
#   --failed_path   : path to save items that failed verbalization / consistency checks.
#
# Optional args (depending on your Python implementation):
#   --no_sql        : if set, skip re-running SQL over cleaned raw_tables and
#                     keep the original "answer" field.
#                     Without this flag, the script will execute the SQL to
#                     recompute the answer.
###############################################################################
echo "💬 Step 4: Verbalizing tables..."
python ../src/verbalization_col_s4.py \
  --data_path     "${DATA_JSON}" \
  --template_path "${OUT_DIR}/example_col_combinations.csv" \
  --output_path   "${OUT_DIR}/example_verbalized.json" \
  --failed_path   "${OUT_DIR}/example_failed_items.json" \
  --no_sql
  # Remove --no_sql if you want to re-run SQL on cleaned tables to recompute answers.

echo "✅ Pipeline finished! All artifacts are in: ${OUT_DIR}"