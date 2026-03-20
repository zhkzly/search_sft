# Trajectory数据训练指南

## 1. 数据转换

将您的trajectory数据转换为训练格式：

```bash
python process_data/trajectory_coldstart/convert_trajectory_legacy.py \
  --input your_trajectories.json \
  --output sft/data/training_data.json
```

## 2. 训练

```bash
bash sft/scripts/train_qwen_7b_deepspeed.sh

# or edit and run a script under sft/scripts/ from the project root
export CUDA_VISIBLE_DEVICES=0,1,2,3

lr=1e-5
base="your_base_model_path"
tokenizer="your_tokenizer_path"
train_data="sft/data/training_data.json"
bsz=2
acc=4

output_dir="sft/output/trajectory_sft"
mkdir -p "$output_dir"

deepspeed \
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
    --model_max_length 30000 \
    --save_total_limit 5 \
    --bf16
```

## 数据格式说明

### 输入格式（您的trajectory.json）
```json
{
  "task": "任务描述",
  "llm_interactions": [
    {
      "input_messages": [{"role": "system", "content": "..."}],
      "response": {
        "content": "推理内容",
        "tool_calls": [{"name": "bash", "arguments": {...}}]
      },
      "tool_result": {"result": "..."}
    }
  ]
}
```

### 转换后格式
```json
{
  "input": "任务描述\n\nsystem prompt",
  "output": [
    {"gen": "推理内容"},
    {"gen": "<tool_call>bash</tool_call><args>{...}</args>"},
    {"doc_gen": "<tool_result>...</tool_result>"}
  ]
}
```

### Loss计算规则
- ✅ **计算loss**: `"gen"` 字段（推理 + tool call）
- ❌ **不计算loss**: `"doc_gen"` 字段（tool result）
