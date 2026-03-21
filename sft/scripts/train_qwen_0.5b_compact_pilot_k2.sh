#!/bin/bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/zkl/miniconda3/envs/simpdsu/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-.cache/hf_datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-.cache/matplotlib}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

lr="${LR:-2e-5}"
base="${BASE_MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Coder-0.5B-Instruct}"
tokenizer="${TOKENIZER_NAME_OR_PATH:-Qwen/Qwen2.5-Coder-0.5B-Instruct}"
train_data="${TRAIN_DATA:-sft/data/step_level_compact_pilot_k2_state_on_train.jsonl}"
bsz="${PER_DEVICE_BATCH_SIZE:-1}"
acc="${GRAD_ACC_STEPS:-8}"
epochs="${NUM_TRAIN_EPOCHS:-1}"
max_len="${MODEL_MAX_LENGTH:-4096}"
job_name="${JOB_NAME:-qwen2.5-coder-0.5b-compact-pilot-k2}"

JOB_ID=$(( RANDOM % 100000 ))
output_dir="${OUTPUT_DIR:-sft/output/JOB:${JOB_ID}#${job_name}}"
mkdir -p ".cache/huggingface" ".cache/hf_datasets" ".cache/matplotlib" "$output_dir"

echo "=========================================="
echo "Job Name: ${job_name}"
echo "Output Dir: ${output_dir}"
echo "Train Data: ${train_data}"
echo "Base Model: ${base}"
echo "Tokenizer: ${tokenizer}"
echo "LR: ${lr}"
echo "Batch Size: ${bsz}"
echo "Grad Accumulation: ${acc}"
echo "Epochs: ${epochs}"
echo "=========================================="

"${PYTHON_BIN}" sft/sft.py \
    --model_name_or_path "${base}" \
    --tokenizer_name_or_path "${tokenizer}" \
    --do_train \
    --data_path "${train_data}" \
    --lr_scheduler_type cosine \
    --output_dir "${output_dir}" \
    --warmup_ratio 0.03 \
    --gradient_checkpointing true \
    --per_device_train_batch_size "${bsz}" \
    --gradient_accumulation_steps "${acc}" \
    --logging_steps 1 \
    --learning_rate "${lr}" \
    --num_train_epochs "${epochs}" \
    --save_strategy epoch \
    --save_only_model true \
    --model_max_length "${max_len}" \
    --save_total_limit 2 \
    --bf16

echo "Pilot training completed! Model saved to: ${output_dir}"
