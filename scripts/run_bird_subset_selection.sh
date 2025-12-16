#!/usr/bin/env bash
set -euo pipefail

#############################################
# BIRD split construction + answer checking
# ------------------------------------------
# S1: short  (<=100 rows), lookup (SimpleProjection), single-table
# S3: short  (<=100 rows), compositional (non-SimpleProjection), single-table
# S4: long   (>=101 rows), lookup, single-table
# S5: long   (>=101 rows), compositional, single-table
#
# For each split:
#   1) build  -> construct subset with table content
#   2) answers -> execute SQL, attach "answer", sample up to 100 per sql_type
#############################################

# Resolve paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="${SCRIPT_DIR}/.."

BIRD_ROOT="${REPO_ROOT}/data_selection/BIRD"
PIPELINE_PY="${BIRD_ROOT}/bird_pipeline.py"
OUT_DIR="${BIRD_ROOT}/splits"

# You can override python via: PYTHON=python3 bash scripts/run_bird_splits.sh
PYTHON="${PYTHON:-python}"

mkdir -p "${OUT_DIR}"

echo "[INFO] BIRD_ROOT   = ${BIRD_ROOT}"
echo "[INFO] PIPELINE_PY = ${PIPELINE_PY}"
echo "[INFO] OUT_DIR     = ${OUT_DIR}"
echo "[INFO] PYTHON      = ${PYTHON}"

##########################
# S1: short + lookup
##########################
S1_JSON="${OUT_DIR}/bird_S1_short_lookup.json"
S1_JSON_ANS="${OUT_DIR}/bird_S1_short_lookup_with_answer.json"

echo ""
echo "=== Building S1 (short, lookup) ==="
${PYTHON} "${PIPELINE_PY}" build \
  --bird_root "${BIRD_ROOT}" \
  --splits "train,dev_20240627" \
  --min_rows 0 \
  --max_rows 100 \
  --sql_mode "lookup" \
  --single_table_only \
  --out "${S1_JSON}"

echo "=== Executing SQL for S1 ==="
${PYTHON} "${PIPELINE_PY}" answers \
  --in "${S1_JSON}" \
  --out "${S1_JSON_ANS}" \
  --samples_per_type 100


##########################
# S3: short + compositional
##########################
S3_JSON="${OUT_DIR}/bird_S3_short_compositional.json"
S3_JSON_ANS="${OUT_DIR}/bird_S3_short_compositional_with_answer.json"

echo ""
echo "=== Building S3 (short, compositional) ==="
${PYTHON} "${PIPELINE_PY}" build \
  --bird_root "${BIRD_ROOT}" \
  --splits "train,dev_20240627" \
  --min_rows 0 \
  --max_rows 100 \
  --sql_mode "compositional" \
  --single_table_only \
  --out "${S3_JSON}"

echo "=== Executing SQL for S3 ==="
${PYTHON} "${PIPELINE_PY}" answers \
  --in "${S3_JSON}" \
  --out "${S3_JSON_ANS}" \
  --samples_per_type 100


##########################
# S4: long + lookup
##########################
S4_JSON="${OUT_DIR}/bird_S4_long_lookup.json"
S4_JSON_ANS="${OUT_DIR}/bird_S4_long_lookup_with_answer.json"

echo ""
echo "=== Building S4 (long, lookup) ==="
${PYTHON} "${PIPELINE_PY}" build \
  --bird_root "${BIRD_ROOT}" \
  --splits "train,dev_20240627" \
  --min_rows 101 \
  --sql_mode "lookup" \
  --single_table_only \
  --out "${S4_JSON}"

echo "=== Executing SQL for S4 ==="
${PYTHON} "${PIPELINE_PY}" answers \
  --in "${S4_JSON}" \
  --out "${S4_JSON_ANS}" \
  --samples_per_type 100


##########################
# S5: long + compositional
##########################
S5_JSON="${OUT_DIR}/bird_S5_long_compositional.json"
S5_JSON_ANS="${OUT_DIR}/bird_S5_long_compositional_with_answer.json"

echo ""
echo "=== Building S5 (long, compositional) ==="
${PYTHON} "${PIPELINE_PY}" build \
  --bird_root "${BIRD_ROOT}" \
  --splits "train,dev_20240627" \
  --min_rows 101 \
  --sql_mode "compositional" \
  --single_table_only \
  --out "${S5_JSON}"

echo "=== Executing SQL for S5 ==="
${PYTHON} "${PIPELINE_PY}" answers \
  --in "${S5_JSON}" \
  --out "${S5_JSON_ANS}" \
  --samples_per_type 100

echo ""
echo "✅All BIRD splits built and answers generated."