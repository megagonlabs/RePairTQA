#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# NL2SQL baseline (example)
# ================================================================

SCRIPT="../src/nl2sql_baseline_s6.py"      # entry script name
MODEL="gpt-4o-2024-11-20"           # or gpt-4o-mini
API_KEY=${OPENAI_API_KEY}

# === Example dataset path (after verbalization step) ===
INPUT_JSON="../pipeline_results/run_example/example_verbalized.json"

# === Output paths ===
OUT_DIR="../baseline_outputs/example_nl2sql"
mkdir -p "${OUT_DIR}"

OUT_RAW="${OUT_DIR}/example_nl2sql_raw_results.jsonl"
OUT_TBL="${OUT_DIR}/example_nl2sql_tables_results.jsonl"

# ================================================================
# Run NL2SQL (raw vs table modes)
# ================================================================

echo "💬 [RAW] ${INPUT_JSON} -> ${OUT_RAW}"
python "${SCRIPT}" \
  --input_json "${INPUT_JSON}" \
  --model "${MODEL}" \
  --output_file "${OUT_RAW}" \
  --openai_api_key "${API_KEY}"

echo "💬 [TABLES] ${INPUT_JSON} -> ${OUT_TBL}"
python "${SCRIPT}" \
  --input_json "${INPUT_JSON}" \
  --model "${MODEL}" \
  --no_raw \
  --output_file "${OUT_TBL}" \
  --openai_api_key "${API_KEY}"

echo "🎉 Example NL2SQL baseline completed!"
echo "   Raw-table mode  : ${OUT_RAW}"
echo "   Tables-only mode: ${OUT_TBL}"