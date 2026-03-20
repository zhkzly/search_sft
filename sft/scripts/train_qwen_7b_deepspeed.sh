#!/bin/bash
# DeepSpeed ZeRO-3 + CPU Offload 版本 - 显存优化更激进

export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=2,3,4,6,7

lr=2e-5
base="Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer="Qwen/Qwen2.5-Coder-7B-Instruct"
train_data="sft/data/trajectory_training_data.json"
bsz=1
acc=8

JOB_ID=$(( RANDOM % 100000 ))
JOB_NAME="qwen2.5-coder-7b-deepspeed"
output_dir="sft/output/JOB:${JOB_ID}#${JOB_NAME}"
mkdir -p "$output_dir"

echo "=========================================="
echo "Job Name: ${JOB_NAME}"
echo "Output Dir: ${output_dir}"
echo "=========================================="

deepspeed --num_gpus=5 \
    --master_port=9944 \
    sft/sft.py \
    --deepspeed sft/configs/ds_zero3_offload.json \
    --model_name_or_path $base \
    --tokenizer_name_or_path $tokenizer \
    --do_train \
    --data_path $train_data \
    --lr_scheduler_type cosine \
    --output_dir $output_dir \
    --warmup_ratio 0.03 \
    --gradient_checkpointing true \
    --per_device_train_batch_size $bsz \
    --gradient_accumulation_steps $acc \
    --logging_steps 1 \
    --learning_rate "$lr" \
    --num_train_epochs 6 \
    --save_strategy epoch \
    --save_only_model true \
    --model_max_length 20000 \
    --save_total_limit 5 \
    --bf16 || exit 1

echo "Training completed! Model saved to: ${output_dir}"
