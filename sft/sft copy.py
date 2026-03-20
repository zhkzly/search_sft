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
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import random
from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
import shutil

# from liger_kernel.transformers import AutoLigerKernelForCausalLM


import matplotlib.pyplot as plt
import numpy as np


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
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


IGNORE_INDEX = -100
MAX_LENGTH = 2000

def process(sample, tokenizer):
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

    input_ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)

    result = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
    # print(result)
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

    input_ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)

    result = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
    # print(result)
    return result



def print_function(example, tokenizer):
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print("labels:\n{}".format(
        tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
    ))


def get_dataset(file_path, tokenizer, other_dataset=False):
    dataset = load_dataset('json', data_files=file_path)
    train_dataset = dataset["train"]
    file_name = os.path.basename(file_path)
    dataset_name = os.path.splitext(file_name)[0]

    if other_dataset:
        tokenized_dataset = train_dataset.map(process_other_data, fn_kwargs={'tokenizer': tokenizer}, num_proc=1, load_from_cache_file=False)
    else:
        tokenized_dataset = train_dataset.map(process, fn_kwargs={'tokenizer': tokenizer}, num_proc=1, load_from_cache_file=False)
    print_function(next(iter(tokenized_dataset)), tokenizer)
    print(f"len of dataset before filter: {len(tokenized_dataset)}")
    
    filtered_dataset = []
    for item in tokenized_dataset:
        if len(item["input_ids"]) <= 5000:
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
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        _attn_implementation="sdpa",  # 注释掉，使用默认的 eager attention
        use_cache=use_cache, 
        #  save_only_model=True
    ).float()

    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path, model_max_length=training_args.model_max_length
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = get_dataset(data_args.data_path, tokenizer)
    if data_args.other_type_data:
        dataset_other = get_dataset(data_args.other_type_data, tokenizer, True)
        dataset = dataset + dataset_other
    print(f"dataset length: {len(dataset)}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
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
