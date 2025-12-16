#!/usr/bin/env bash
set -euo pipefail

#############################################
# TableEval S2 split: simple lookup + short + flat tables
#############################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="${SCRIPT_DIR}/.."

# Adjust this if your TableEval directory is elsewhere
TABLEEVAL_ROOT="${REPO_ROOT}/data_selection/TableEval"

PYTHON="${PYTHON:-python}"

META_FILE="${TABLEEVAL_ROOT}/TableEval-meta.jsonl"
TEST_FILE="${TABLEEVAL_ROOT}/TableEval-test.jsonl"
OUT_FILE="${TABLEEVAL_ROOT}/TableEval_S2_simple_short_flat.json"

PIPELINE_PY="${TABLEEVAL_ROOT}/data_selection.py"

echo "[INFO] TABLEEVAL_ROOT = ${TABLEEVAL_ROOT}"
echo "[INFO] META_FILE      = ${META_FILE}"
echo "[INFO] TEST_FILE      = ${TEST_FILE}"
echo "[INFO] OUT_FILE       = ${OUT_FILE}"
echo "[INFO] PYTHON         = ${PYTHON}"

mkdir -p "${TABLEEVAL_ROOT}"

${PYTHON} "${PIPELINE_PY}" \
  --meta_file "${META_FILE}" \
  --test_file "${TEST_FILE}" \
  --out "${OUT_FILE}" \
  --min_rows 0 \
  --max_rows 100

echo ""
echo "TableEval S2 split generated at: ${OUT_FILE}"