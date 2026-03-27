#!/bin/bash

export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0,1  # 使用两张卡

# 定义参数
lr=2e-5
base="Qwen/Qwen2.5-Coder-1.5B-Instruct"
tokenizer="Qwen/Qwen2.5-Coder-1.5B-Instruct"
train_data="sft/data/training_data_871.json"
bsz=4                     # 每张卡的 batch size
acc=8                     # 梯度累积，有效 batch size = 4 * 8 * 2 = 64

# 创建输出目录
JOB_ID=$(( RANDOM % 100000 ))
JOB_NAME="qwen2.5-coder-1.5b-fsdp-2gpu"
output_dir="sft/output/JOB:${JOB_ID}#${JOB_NAME}"
mkdir -p "$output_dir"

echo "=========================================="
echo "Job Name: ${JOB_NAME}"
echo "Output Dir: ${output_dir}"
echo "Model: ${base}"
echo "Effective Batch Size: $(( bsz * acc * 2 ))"
echo "=========================================="

# 使用 torchrun 启动 FSDP
torchrun --nproc_per_node=2 \
    --master_port=9944 \
    sft/sft.py \
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
    --model_max_length 30000 \
    --save_total_limit 5 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "Qwen2DecoderLayer" \
    --bf16 || exit 1

echo "Training completed! Model saved to: ${output_dir}"
