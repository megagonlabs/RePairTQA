#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Step 5: LLM QA inference (Example Dataset)
#
# Runs GPT-based QA on the verbalized dataset and (optionally) a structured
# baseline using raw_tables.
#
# This script assumes the following Python entrypoint exists in ../src/:
#   - llm_infer_s5.py   (API compatible with gpt_qa_inference.py)
###############################################################################

#######################################
# Input & Output configuration
#######################################

OUT_DIR="../baseline_outputs/example_gpt"
mkdir -p "${OUT_DIR}"

# your API key here
API_KEY=${OPENAI_API_KEY}

# Check if key is filled
if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: Please provide your OpenAI API key in this script (variable API_KEY)." >&2
  exit 1
fi

#######################################
# Inference configuration
#######################################

# Path to your verbalized output from Step 4
INPUT_JSON="../pipeline_results/run_example/example_verbalized.json"

# Output files
OUT_SEMI="${OUT_DIR}/example_semi_gpt4o.jsonl"
OUT_STRUCTURED="${OUT_DIR}/example_structured_rawtable_gpt4o.jsonl"

MODEL="gpt-4o-2024-11-20"
PROVIDER="gpt"   # uses the official OpenAI client in llm_infer_s5.py

###############################################################################
# Run inference
###############################################################################

echo "💬 Step 5: LLM inference (example dataset, semi-structured + structured baseline)..."

python ../src/llm_infer_s5.py \
  --input_file "${INPUT_JSON}" \
  --output_file "${OUT_SEMI}" \
  --structured_output_file "${OUT_STRUCTURED}" \
  --data_type semi-structured \
  --model "${MODEL}" \
  --provider "${PROVIDER}" \
  --api_key "${API_KEY}"

echo "✅ LLM inference done!"
echo "   Semi-structured results : ${OUT_SEMI}"
echo "   Structured baseline     : ${OUT_STRUCTURED}"