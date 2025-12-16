#!/usr/bin/env bash
set -euo pipefail

#############################################
# MMQA M-splits (M1 / M2 + optional No-Rules)
# -------------------------------------------
# 输入（默认假设）：
#   data_selection/MMQA/Synthesized_three_table.json
#   data_selection/MMQA/Synthesized_two_table.json
#
# 输出：
#   data_selection/MMQA/splits/filtered_three_table_No_Rules.json
#   data_selection/MMQA/splits/filtered_two_table_No_Rules.json
#   data_selection/MMQA/splits/MMQA_M1_short_multi.json
#   data_selection/MMQA/splits/MMQA_M2_long_multi.json
#############################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="${SCRIPT_DIR}/.."

MMQA_ROOT="${REPO_ROOT}/data_selection/MMQA"
PYTHON="${PYTHON:-python}"

THREE_JSON="${MMQA_ROOT}/Synthesized_three_table.json"
TWO_JSON="${MMQA_ROOT}/Synthesized_two_table.json"
OUT_DIR="${MMQA_ROOT}/splits"

PIPELINE_PY="${MMQA_ROOT}/data_selection.py"

echo "[INFO] MMQA_ROOT       = ${MMQA_ROOT}"
echo "[INFO] THREE_JSON      = ${THREE_JSON}"
echo "[INFO] TWO_JSON        = ${TWO_JSON}"
echo "[INFO] OUT_DIR         = ${OUT_DIR}"
echo "[INFO] PIPELINE_PY     = ${PIPELINE_PY}"
echo "[INFO] PYTHON          = ${PYTHON}"

mkdir -p "${OUT_DIR}"

# 你可以按需要调这些 threshold / 数量
SHORT_MAX_ROWS=100   # M1: 每张表 <= 100 行视为短表
LONG_MIN_ROWS=101    # M2: 每张表 >= 101 行视为长表
MIN_VALID_TABLES=2   # 至少两张表满足短/长条件
MAX_SAMPLES=100      # 每组（two/three）最多保留多少，用大一点就不抽样了
SEED=42

${PYTHON} "${PIPELINE_PY}" \
  --three_table_json "${THREE_JSON}" \
  --two_table_json "${TWO_JSON}" \
  --out_dir "${OUT_DIR}" \
  --short_min_rows 1 \
  --short_max_rows "${SHORT_MAX_ROWS}" \
  --long_min_rows "${LONG_MIN_ROWS}" \
  --min_valid_table_count "${MIN_VALID_TABLES}" \
  --max_samples "${MAX_SAMPLES}" \
  --seed "${SEED}" \
  --build_no_rules

echo ""
echo "MMQA splits generated under: ${OUT_DIR}"