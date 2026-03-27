#!/bin/bash

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0  # 只用一张卡

# 定义参数
lr=2e-5
base="Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ"  # 0.5B 模型
tokenizer="Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ"
train_data="sft/data/training_data_871.json"
bsz=1                     # 0.5B 模型可以用更大的 batch size
acc=8                     # 梯度累积，有效 batch size = 64

# 创建输出目录
JOB_ID=$(( RANDOM % 100000 ))
JOB_NAME="qwen2.5-coder-0.5b-sft"
output_dir="sft/output/JOB:${JOB_ID}#${JOB_NAME}"
mkdir -p "$output_dir"

echo "=========================================="
echo "Job Name: ${JOB_NAME}"
echo "Output Dir: ${output_dir}"
echo "Model: ${base}"
echo "Effective Batch Size: $(( bsz * acc ))"
echo "=========================================="

# 不使用 DeepSpeed，直接用 Trainer
python sft/sft.py \
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
    --model_max_length 3000 \
    --save_total_limit 5 \
    --bf16 || exit 1

echo "Training completed! Model saved to: ${output_dir}"
