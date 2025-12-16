#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Gemini QA on example.json (both semi-structured and structured views)
#
# This script runs llm_infer_s5.py twice:
#   1) data_type=semi-structured  -> uses tables + verbalized text (if any)
#   2) data_type=structured       -> uses tables only
#
# It assumes:
#   - example.json is in the current directory
#   - llm_infer_s5.py is in ../src/ directory
###############################################################################

############################
# 1. Basic configuration
############################

# TODO: put your  Gemini API key here or export it in the environment variable GEMINI_API_KEY
GEMINI_API_KEY=${GEMINI_API_KEY}

# Python entry script
PY_SCRIPT="../src/llm_infer_s5.py"

# Input dataset (your example file)
INPUT_JSON="../pipeline_results/run_example/example_verbalized.json"

# Output directory
OUT_DIR="../baseline_outputs/example_gemini"

# Gemini model (via LiteLLM)
MODEL="gemini/gemini-2.5-flash"   # or: gemini/gemini-2.5-pro

# Optional: limit the number of samples (leave empty to run all)
MAX_SAMPLES=""    # e.g. "10"


############################
# 2. Sanity checks
############################

if [[ -z "${GEMINI_API_KEY}" ]]; then
  echo "ERROR: Please set GEMINI_API_KEY in this script." >&2
  exit 1
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "ERROR: Python script not found: ${PY_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${INPUT_JSON}" ]]; then
  echo "ERROR: Input JSON not found: ${INPUT_JSON}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

# LiteLLM reads the key from the environment, so export it as well
export GEMINI_API_KEY


############################
# 3. Helper: run one config
############################

run_one() {
  local DATA_TYPE="$1"   # semi-structured | structured | unstructured
  local SUFFIX="$2"      # used in output file name

  local OUT_FILE="${OUT_DIR}/example_${SUFFIX}_gemini.jsonl"

  echo "💬 Running Gemini QA on ${INPUT_JSON} (data_type=${DATA_TYPE})"
  set -x
  python "${PY_SCRIPT}" \
    --input_file "${INPUT_JSON}" \
    --output_file "${OUT_FILE}" \
    --data_type "${DATA_TYPE}" \
    --model "${MODEL}" \
    --provider gemini \
    --api_key "${GEMINI_API_KEY}" \
    $( [[ -n "${MAX_SAMPLES}" ]] && echo --max_samples "${MAX_SAMPLES}" )
  set +x

  echo "✅ Done: ${OUT_FILE}"
}

############################
# 4. Run both views
############################

# 4.1 semi-structured: tables + verbalized text (if present, otherwise just tables)
run_one "semi-structured" "semi"

# 4.2 structured: tables only
run_one "structured" "structured"

echo "🎉 All done. Results are in: ${OUT_DIR}"