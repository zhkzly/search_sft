#!/bin/bash

set -euo pipefail

# Run this script from the project root.
# Modes:
#   gold                -> sanity check the evaluation logic with gold targets
#   base                -> evaluate a base model
#   model <checkpoint>  -> evaluate a trained model checkpoint

PYTHON_BIN="/data/zkl/miniconda3/envs/simpdsu/bin/python"

MODE="${1:-gold}"
MODEL_PATH="${2:-}"

DATA_PATH="${DATA_PATH:-sft/data/step_level_compact_k2_state_on_eval.jsonl}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-Qwen/Qwen2.5-Coder-0.5B-Instruct}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Coder-0.5B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
TEMPERATURE="${TEMPERATURE:-0.0}"
RUN_NAME="${RUN_NAME:-}"

export HF_HOME="${HF_HOME:-.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-.cache/hf_datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-.cache/matplotlib}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$MPLCONFIGDIR" "eval/results/tool_use_step_eval"

if [ ! -f "eval/tool_use_step_eval.py" ]; then
  echo "Please run this script from the project root."
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"

case "$MODE" in
  gold)
    run_tag="${RUN_NAME:-gold_${timestamp}}"
    cmd=(
      "$PYTHON_BIN" eval/tool_use_step_eval.py
      --data_path "$DATA_PATH"
      --use_gold_targets
      --max_samples "$MAX_SAMPLES"
      --output_path "eval/results/tool_use_step_eval/${run_tag}_results.jsonl"
      --metrics_path "eval/results/tool_use_step_eval/${run_tag}_metrics.json"
    )
    ;;
  base)
    run_tag="${RUN_NAME:-base_${timestamp}}"
    cmd=(
      "$PYTHON_BIN" eval/tool_use_step_eval.py
      --data_path "$DATA_PATH"
      --model_name_or_path "$BASE_MODEL_NAME_OR_PATH"
      --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH"
      --max_samples "$MAX_SAMPLES"
      --max_new_tokens "$MAX_NEW_TOKENS"
      --temperature "$TEMPERATURE"
      --output_path "eval/results/tool_use_step_eval/${run_tag}_results.jsonl"
      --metrics_path "eval/results/tool_use_step_eval/${run_tag}_metrics.json"
    )
    ;;
  model)
    if [ -z "$MODEL_PATH" ]; then
      echo "Usage: bash eval/scripts/run_tool_use_step_eval.sh model <checkpoint_path>"
      exit 1
    fi
    run_tag="${RUN_NAME:-model_${timestamp}}"
    cmd=(
      "$PYTHON_BIN" eval/tool_use_step_eval.py
      --data_path "$DATA_PATH"
      --model_name_or_path "$MODEL_PATH"
      --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH"
      --max_samples "$MAX_SAMPLES"
      --max_new_tokens "$MAX_NEW_TOKENS"
      --temperature "$TEMPERATURE"
      --output_path "eval/results/tool_use_step_eval/${run_tag}_results.jsonl"
      --metrics_path "eval/results/tool_use_step_eval/${run_tag}_metrics.json"
    )
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage:"
    echo "  bash eval/scripts/run_tool_use_step_eval.sh gold"
    echo "  bash eval/scripts/run_tool_use_step_eval.sh base"
    echo "  bash eval/scripts/run_tool_use_step_eval.sh model <checkpoint_path>"
    exit 1
    ;;
esac

echo "=========================================="
echo "Mode: ${MODE}"
echo "Data Path: ${DATA_PATH}"
echo "Max Samples: ${MAX_SAMPLES}"
if [ "$MODE" = "base" ]; then
  echo "Model: ${BASE_MODEL_NAME_OR_PATH}"
fi
if [ "$MODE" = "model" ]; then
  echo "Model: ${MODEL_PATH}"
fi
echo "Results Prefix: eval/results/tool_use_step_eval/${run_tag}"
echo "=========================================="

"${cmd[@]}"
