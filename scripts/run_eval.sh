#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# LLM-based Evaluation (example)
# ================================================================

SCRIPT="../src/gpt_evaluator_s7.py"     # evaluation script
MODEL="gpt-4o-2024-11-20"               # or gpt-4o-mini
API_KEY=${OPENAI_API_KEY}

# === Input: model prediction vs ground truth ===
INPUT_FILE="../baseline_outputs/example_nl2sql/example_nl2sql_raw_results.jsonl"

# === Output: GPT-evaluated result JSONL ===
OUT_DIR="../evaluation_outputs/example_eval"
mkdir -p "${OUT_DIR}"

OUTPUT_FILE="${OUT_DIR}/example_eval_gpt.jsonl"

# === Optional: specify total length (for progress bar) ===
TOTAL_LENGTH=100

# ================================================================
# Run GPT-based evaluator
# ================================================================

echo "🧠 Evaluating predictions using ${MODEL}"
python "${SCRIPT}" \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --model "${MODEL}" \
  --api_key "${API_KEY}" \
  --total_length "${TOTAL_LENGTH}"

echo "✅ GPT evaluation completed!"
echo "   Output: ${OUTPUT_FILE}"