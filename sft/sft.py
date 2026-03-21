import os
import copy
import json
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
import importlib.util as _importlib_util

_real_find_spec = _importlib_util.find_spec


def _patched_find_spec(name, package=None):
    if name == "torchvision":
        return None
    return _real_find_spec(name, package)


if _real_find_spec("torchvision") is not None:
    _importlib_util.find_spec = _patched_find_spec
    try:
        import transformers
    finally:
        _importlib_util.find_spec = _real_find_spec
else:
    import transformers
from torch.utils.data import Dataset
import random
from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
import shutil

# from liger_kernel.transformers import AutoLigerKernelForCausalLM


import matplotlib.pyplot as plt
import numpy as np

from process_data.trajectory_coldstart.tool_schemas import (
    CANONICAL_TOOL_SCHEMAS,
    normalize_message_for_tool_template,
    normalize_messages_for_tool_template,
)

IGNORE_INDEX = -100
MAX_LENGTH = 2000


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    # flash_attention: Optional[bool] = field(default=False)
    tokenizer_name_or_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    prompt_type: Optional[str] = field(default="instruction")
    dailog_augmentation: Optional[bool] = field(default=False)
    other_type_data: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path of other type of data"
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=MAX_LENGTH,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def process_legacy(sample, tokenizer):
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    source = sample["input"]

    source = tokenizer.apply_chat_template(
        [
            {'role': 'user', 'content': source}
        ],
        tokenize=False, add_generation_prompt=True
    )

    source = tokenizer(source, add_special_tokens=False)["input_ids"]
    target = [IGNORE_INDEX] * len(source)
    for output in sample["output"]:
        for k, v in output.items():
            if v is None:
                continue
            v_tokens = tokenizer(v, add_special_tokens=False)["input_ids"]
            if k in ["gen"]:
                source += v_tokens
                target += v_tokens
            elif k in ["doc_gen"]:
                source += v_tokens
                target += [IGNORE_INDEX] * len(v_tokens)
    input_ids = source
    labels = target

    if not input_ids or input_ids[-1] != tokenizer.eos_token_id:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)

    result = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
    # print(result)
    return result


def build_compact_prefix_messages(sample):
    task_block = sample["task_block"]
    state_block = sample["state_block"]
    recent_window_messages = sample["recent_window_messages"]

    messages = []
    system_text = task_block.get("system") or ""
    user_text = task_block.get("user") or ""
    if system_text:
        messages.append({"role": "system", "content": system_text})
    if user_text:
        messages.append({"role": "user", "content": user_text})

    state_text = "[COMPACT_STATE]\n" + json.dumps(state_block, ensure_ascii=False, indent=2) + "\n[/COMPACT_STATE]"
    messages.append({"role": "user", "content": state_text})
    messages.extend(normalize_messages_for_tool_template(recent_window_messages))
    return messages


def render_and_tokenize_messages(tokenizer, messages, add_generation_prompt):
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=CANONICAL_TOOL_SCHEMAS,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    input_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return rendered, input_ids


def process_compact(sample, tokenizer):
    prefix_messages = build_compact_prefix_messages(sample)
    target_message = normalize_message_for_tool_template(sample["target_message"])

    prefix_rendered, prefix_ids = render_and_tokenize_messages(
        tokenizer, prefix_messages, add_generation_prompt=True
    )
    full_messages = prefix_messages + [target_message]
    full_rendered, full_ids = render_and_tokenize_messages(
        tokenizer, full_messages, add_generation_prompt=False
    )

    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "Compact sample prefix tokens are not a prefix of the full sequence. "
            f"sample_id={sample.get('sample_id')}"
        )

    labels = [IGNORE_INDEX] * len(prefix_ids) + full_ids[len(prefix_ids):]
    input_ids = list(full_ids)
    if not input_ids or input_ids[-1] != tokenizer.eos_token_id:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)

    result = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
    return result


def process_other_data(sample, tokenizer): # without mask

    source = sample["prompt"]
    source = tokenizer.apply_chat_template(
        [
            {'role': 'user', 'content': source}
        ],
        tokenize=False, add_generation_prompt=True
    )

    source = tokenizer(source, add_special_tokens=False)["input_ids"]
    target = [IGNORE_INDEX] * len(source)

    output = sample["output"]
    output = tokenizer(output, add_special_tokens=False)["input_ids"]

    source += output
    target += output
    
    input_ids = source
    labels = target

    if not input_ids or input_ids[-1] != tokenizer.eos_token_id:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)

    result = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
    # print(result)
    return result


def process(sample, tokenizer):
    if "task_block" in sample and "state_block" in sample and "recent_window_messages" in sample:
        return process_compact(sample, tokenizer)
    return process_legacy(sample, tokenizer)



def print_function(example, tokenizer):
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print("labels:\n{}".format(
        tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
    ))


def resolve_dataset_cache_dir(explicit_cache_dir: Optional[str] = None) -> str:
    if explicit_cache_dir:
        return explicit_cache_dir
    env_cache_dir = os.environ.get("HF_DATASETS_CACHE")
    if env_cache_dir:
        return env_cache_dir
    return ".cache/hf_datasets"


def get_dataset(file_path, tokenizer, other_dataset=False, cache_dir: Optional[str] = None):
    max_length = getattr(tokenizer, "model_max_length", MAX_LENGTH)
    dataset = load_dataset('json', data_files=file_path, cache_dir=resolve_dataset_cache_dir(cache_dir))
    train_dataset = dataset["train"]
    file_name = os.path.basename(file_path)
    dataset_name = os.path.splitext(file_name)[0]

    if other_dataset:
        tokenized_dataset = train_dataset.map(
            process_other_data,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=1,
            load_from_cache_file=False,
        )
    else:
        tokenized_dataset = train_dataset.map(
            process,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=1,
            load_from_cache_file=False,
        )
    print_function(next(iter(tokenized_dataset)), tokenizer)
    print(f"len of dataset before filter: {len(tokenized_dataset)}")
    
    filtered_dataset = []
    for item in tokenized_dataset:
        if len(item["input_ids"]) <= max_length:
            filtered_dataset.append(item)
    print(f"len of dataset after filter: {len(filtered_dataset)}")
    return filtered_dataset


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("==========Model Args=========")
    print(model_args)
    print("==========Data Args=========")
    print(data_args)
    print("==========Training Args=========")
    print(training_args)

    use_cache = True
    if training_args.gradient_checkpointing:
        use_cache = False # use_cache与gradient_checkpointing不能同时设置为true
    model_dtype = None
    if torch.cuda.is_available():
        if training_args.bf16:
            model_dtype = torch.bfloat16
        elif training_args.fp16:
            model_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        _attn_implementation="sdpa",  # 注释掉，使用默认的 eager attention
        use_cache=use_cache,
        torch_dtype=model_dtype,
        #  save_only_model=True
    )

    if not torch.cuda.is_available():
        model = model.float()

    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path, model_max_length=training_args.model_max_length
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = get_dataset(data_args.data_path, tokenizer, cache_dir=training_args.cache_dir)
    if data_args.other_type_data:
        dataset_other = get_dataset(data_args.other_type_data, tokenizer, True, cache_dir=training_args.cache_dir)
        dataset = dataset + dataset_other
    print(f"dataset length: {len(dataset)}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    torch.manual_seed(42)
    train()
