import argparse
import copy
import importlib.util as _importlib_util
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from tqdm.auto import tqdm

_real_find_spec = _importlib_util.find_spec


def _patched_find_spec(name, package=None):
    if name == "torchvision":
        return None
    return _real_find_spec(name, package)


if _real_find_spec("torchvision") is not None:
    _importlib_util.find_spec = _patched_find_spec
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    finally:
        _importlib_util.find_spec = _real_find_spec
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer

from process_data.trajectory_coldstart.tool_schemas import (
    ALLOWED_TOOL_SCHEMAS,
    CANONICAL_TOOL_SCHEMAS,
    normalize_message_for_tool_template,
    normalize_messages_for_tool_template,
)


TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate next-step tool-use quality on compact step data.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", default="eval/tool_use_step_eval_results.jsonl")
    parser.add_argument("--metrics_path", default="eval/tool_use_step_eval_metrics.json")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--tokenizer_name_or_path", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use_gold_targets", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_main_process_from_env() -> bool:
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank) == 0
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0
    return True


def build_compact_prefix_messages(sample: Dict) -> List[Dict]:
    task_block = sample["task_block"]
    state_block = sample["state_block"]
    recent_window_messages = sample["recent_window_messages"]

    messages = []
    if task_block.get("system"):
        messages.append({"role": "system", "content": task_block["system"]})
    if task_block.get("user"):
        messages.append({"role": "user", "content": task_block["user"]})
    state_text = "[COMPACT_STATE]\n" + json.dumps(state_block, ensure_ascii=False, indent=2) + "\n[/COMPACT_STATE]"
    messages.append({"role": "user", "content": state_text})
    messages.extend(normalize_messages_for_tool_template(recent_window_messages))
    return messages


def render_prefix(tokenizer, sample: Dict) -> str:
    prefix_messages = build_compact_prefix_messages(sample)
    return tokenizer.apply_chat_template(
        prefix_messages,
        tools=CANONICAL_TOOL_SCHEMAS,
        tokenize=False,
        add_generation_prompt=True,
    )


def _coerce_tool_args(raw_args) -> Tuple[Optional[Dict], Optional[str]]:
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args), None
        except json.JSONDecodeError:
            return None, "invalid_arguments_json"
    if isinstance(raw_args, dict):
        return raw_args, None
    return None, "arguments_not_object_or_string"


def _build_parsed_tool_message(content: str, tool_name: Optional[str], tool_args: Dict) -> Dict:
    return {
        "role": "assistant",
        "content": content.strip(),
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": tool_args,
                }
            }
        ],
    }


def parse_generated_tool_call(text: str) -> Tuple[Optional[Dict], Dict]:
    content = text.strip()
    matches = TOOL_CALL_BLOCK_RE.findall(content)
    if not matches:
        return None, {"format_validity": False, "reason": "missing_tool_call_block"}

    raw_block = matches[0]
    try:
        parsed = json.loads(raw_block)
    except json.JSONDecodeError:
        return None, {"format_validity": False, "reason": "invalid_tool_call_json"}

    tool_name = parsed.get("name")
    raw_args = parsed.get("arguments", {})
    tool_args, error_reason = _coerce_tool_args(raw_args)
    if error_reason is not None:
        return None, {"format_validity": False, "reason": error_reason}

    return (
        _build_parsed_tool_message(
            content=content.split("<tool_call>", 1)[0],
            tool_name=tool_name,
            tool_args=tool_args,
        ),
        {"format_validity": True, "reason": "parsed"},
    )


def parse_generated_bare_json_tool_call(text: str) -> Tuple[Optional[Dict], Dict]:
    content = text.strip()
    if "<tool_call>" in content:
        return None, {"format_validity": False, "reason": "wrapped_tool_call_present"}

    decoder = json.JSONDecoder()
    for start_idx, ch in enumerate(content):
        if ch != "{":
            continue
        try:
            parsed, end_idx = decoder.raw_decode(content, start_idx)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if "name" not in parsed or "arguments" not in parsed:
            continue
        tool_args, error_reason = _coerce_tool_args(parsed.get("arguments", {}))
        if error_reason is not None:
            return None, {"format_validity": False, "reason": error_reason}
        prefix_content = content[:start_idx].strip()
        suffix_content = content[end_idx:].strip()
        if suffix_content:
            prefix_content = (prefix_content + "\n" + suffix_content).strip() if prefix_content else suffix_content
        return (
            _build_parsed_tool_message(
                content=prefix_content,
                tool_name=parsed.get("name"),
                tool_args=tool_args,
            ),
            {"format_validity": True, "reason": "parsed"},
        )

    return None, {"format_validity": False, "reason": "missing_bare_json_tool_call"}


def validate_schema(tool_name: Optional[str], args: Dict) -> Tuple[bool, str]:
    if tool_name not in ALLOWED_TOOL_SCHEMAS:
        return False, "unknown_tool_name"
    schema = ALLOWED_TOOL_SCHEMAS[tool_name]
    for key, expected_type in schema["required"].items():
        if key not in args:
            return False, f"missing_required::{key}"
        if not isinstance(args[key], expected_type):
            return False, f"invalid_type::{key}"
    allowed_keys = set(schema["required"]) | set(schema["optional"])
    for key, value in args.items():
        if key not in allowed_keys:
            return False, f"unexpected_arg::{key}"
        if key in schema["optional"] and not isinstance(value, schema["optional"][key]):
            return False, f"invalid_type::{key}"
    return True, "valid"


def normalize_text(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def path_soft_match(pred: Optional[str], gold: Optional[str]) -> bool:
    if not pred or not gold:
        return False
    pred_n = pred.rstrip("/")
    gold_n = gold.rstrip("/")
    return pred_n == gold_n or pred_n.endswith(gold_n) or gold_n.endswith(pred_n)


def string_soft_match(pred: Optional[str], gold: Optional[str]) -> bool:
    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)
    if not pred_n or not gold_n:
        return False
    return pred_n == gold_n or pred_n in gold_n or gold_n in pred_n


def range_soft_match(pred_start, pred_end, gold_start, gold_end) -> bool:
    if pred_start is None or gold_start is None:
        return False
    if gold_end is None:
        gold_end = gold_start
    if pred_end is None:
        pred_end = pred_start
    return not (pred_end < gold_start or gold_end < pred_start)


def args_soft_match(tool_name: Optional[str], pred_args: Dict, gold_args: Dict) -> bool:
    if tool_name == "task_done":
        return True
    if tool_name == "repo_find_files":
        return path_soft_match(pred_args.get("root_path"), gold_args.get("root_path")) and (
            string_soft_match(pred_args.get("glob"), gold_args.get("glob"))
            or string_soft_match(pred_args.get("name_pattern"), gold_args.get("name_pattern"))
        )
    if tool_name == "repo_list_dir":
        return path_soft_match(pred_args.get("dir_path"), gold_args.get("dir_path"))
    if tool_name == "repo_search_text":
        return path_soft_match(pred_args.get("path"), gold_args.get("path")) and string_soft_match(
            pred_args.get("pattern"), gold_args.get("pattern")
        )
    if tool_name == "repo_read_head":
        return path_soft_match(pred_args.get("file_path"), gold_args.get("file_path"))
    if tool_name == "repo_read_range":
        return path_soft_match(pred_args.get("file_path"), gold_args.get("file_path")) and range_soft_match(
            pred_args.get("start_line"),
            pred_args.get("end_line"),
            gold_args.get("start_line"),
            gold_args.get("end_line"),
        )
    if tool_name == "repo_read_file":
        return path_soft_match(pred_args.get("file_path"), gold_args.get("file_path"))
    return False


def make_gold_record(sample: Dict) -> Dict:
    tool_calls = sample["target_message"].get("tool_calls") or []
    if tool_calls:
        function = normalize_message_for_tool_template(sample["target_message"])["tool_calls"][0]["function"]
        return {
            "tool_name": function["name"],
            "tool_args": function["arguments"],
        }
    return {"tool_name": None, "tool_args": {}}


def predict_with_model(model, tokenizer, sample: Dict, max_new_tokens: int, temperature: float) -> str:
    prefix_text = render_prefix(tokenizer, sample)
    inputs = tokenizer(prefix_text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def build_variant_record(
    *,
    sample: Dict,
    predicted_message: Optional[Dict],
    format_meta: Dict,
    prefix: str,
) -> Dict:
    gold = make_gold_record(sample)
    pred_tool_name = None
    pred_tool_args = {}
    if predicted_message and (predicted_message.get("tool_calls") or []):
        function = predicted_message["tool_calls"][0]["function"]
        pred_tool_name = function.get("name")
        pred_tool_args = function.get("arguments") or {}

    format_validity = format_meta.get("format_validity", False)
    schema_validity = False
    schema_reason = "not_checked"
    if format_validity and pred_tool_name is not None:
        schema_validity, schema_reason = validate_schema(pred_tool_name, pred_tool_args)

    tool_family_correct = pred_tool_name == gold["tool_name"]
    exact_args = pred_tool_args == gold["tool_args"] if pred_tool_name == gold["tool_name"] else False
    soft_args = args_soft_match(pred_tool_name, pred_tool_args, gold["tool_args"]) if tool_family_correct else False

    gold_is_stop = sample["target_kind"] == "stop_step"
    pred_is_stop = pred_tool_name == "task_done"
    stop_step_accuracy = pred_is_stop == gold_is_stop

    return {
        f"{prefix}_pred_tool_name": pred_tool_name,
        f"{prefix}_pred_tool_args": pred_tool_args,
        f"{prefix}_format_validity": format_validity,
        f"{prefix}_format_reason": format_meta.get("reason"),
        f"{prefix}_schema_validity": schema_validity,
        f"{prefix}_schema_reason": schema_reason,
        f"{prefix}_tool_family_correct": tool_family_correct,
        f"{prefix}_tool_args_exact_match": exact_args,
        f"{prefix}_tool_args_soft_match": soft_args,
        f"{prefix}_pred_is_stop": pred_is_stop,
        f"{prefix}_stop_step_accuracy": stop_step_accuracy,
    }


def evaluate_prediction(
    sample: Dict,
    strict_message: Optional[Dict],
    strict_meta: Dict,
    bare_message: Optional[Dict],
    bare_meta: Dict,
    raw_prediction_text: str,
) -> Dict:
    gold = make_gold_record(sample)
    relaxed_message = strict_message if strict_meta.get("format_validity", False) else bare_message
    relaxed_meta = strict_meta if strict_meta.get("format_validity", False) else bare_meta

    strict_record = build_variant_record(
        sample=sample,
        predicted_message=strict_message,
        format_meta=strict_meta,
        prefix="strict",
    )
    bare_record = build_variant_record(
        sample=sample,
        predicted_message=bare_message,
        format_meta=bare_meta,
        prefix="bare_json",
    )
    relaxed_record = build_variant_record(
        sample=sample,
        predicted_message=relaxed_message,
        format_meta=relaxed_meta,
        prefix="relaxed",
    )

    return {
        "sample_id": sample["sample_id"],
        "repo": sample["repo"],
        "target_kind": sample["target_kind"],
        "gold_tool_name": gold["tool_name"],
        "gold_tool_args": gold["tool_args"],
        "raw_prediction_text": raw_prediction_text,
        # Backward-compatible aliases: keep old top-level fields mapped to strict metrics.
        "pred_tool_name": strict_record["strict_pred_tool_name"],
        "pred_tool_args": strict_record["strict_pred_tool_args"],
        "format_validity": strict_record["strict_format_validity"],
        "format_reason": strict_record["strict_format_reason"],
        "schema_validity": strict_record["strict_schema_validity"],
        "schema_reason": strict_record["strict_schema_reason"],
        "tool_family_correct": strict_record["strict_tool_family_correct"],
        "tool_args_exact_match": strict_record["strict_tool_args_exact_match"],
        "tool_args_soft_match": strict_record["strict_tool_args_soft_match"],
        "gold_is_stop": sample["target_kind"] == "stop_step",
        "pred_is_stop": strict_record["strict_pred_is_stop"],
        "stop_step_accuracy": strict_record["strict_stop_step_accuracy"],
        **strict_record,
        **bare_record,
        **relaxed_record,
    }


def summarize_variant(records: List[Dict], prefix: str) -> Dict:
    metrics = {}
    total = len(records)
    if total == 0:
        return {"overall": metrics, "per_tool": {}}

    def avg(field: str) -> float:
        return sum(1 for record in records if record.get(field)) / total

    metrics["num_samples"] = total
    metrics["format_validity"] = avg(f"{prefix}_format_validity")
    metrics["schema_validity"] = avg(f"{prefix}_schema_validity")
    metrics["tool_family_accuracy"] = avg(f"{prefix}_tool_family_correct")
    metrics["tool_args_exact_match"] = avg(f"{prefix}_tool_args_exact_match")
    metrics["tool_args_soft_match"] = avg(f"{prefix}_tool_args_soft_match")
    metrics["stop_step_accuracy"] = avg(f"{prefix}_stop_step_accuracy")

    per_tool = defaultdict(list)
    for record in records:
        per_tool[record["gold_tool_name"]].append(record)

    per_tool_metrics = {}
    for tool_name, tool_records in per_tool.items():
        n = len(tool_records)
        per_tool_metrics[tool_name] = {
            "num_samples": n,
            "format_validity": sum(1 for x in tool_records if x[f"{prefix}_format_validity"]) / n,
            "schema_validity": sum(1 for x in tool_records if x[f"{prefix}_schema_validity"]) / n,
            "tool_family_accuracy": sum(1 for x in tool_records if x[f"{prefix}_tool_family_correct"]) / n,
            "tool_args_soft_match": sum(1 for x in tool_records if x[f"{prefix}_tool_args_soft_match"]) / n,
        }

    return {"overall": metrics, "per_tool": per_tool_metrics}


def summarize(records: List[Dict]) -> Dict:
    strict_metrics = summarize_variant(records, prefix="strict")
    bare_metrics = summarize_variant(records, prefix="bare_json")
    relaxed_metrics = summarize_variant(records, prefix="relaxed")
    return {
        # Backward-compatible aliases: keep old top-level overall/per_tool mapped to strict metrics.
        "overall": strict_metrics["overall"],
        "per_tool": strict_metrics["per_tool"],
        "overall_strict": strict_metrics["overall"],
        "per_tool_strict": strict_metrics["per_tool"],
        "overall_bare_json": bare_metrics["overall"],
        "per_tool_bare_json": bare_metrics["per_tool"],
        "overall_relaxed": relaxed_metrics["overall"],
        "per_tool_relaxed": relaxed_metrics["per_tool"],
    }


def evaluate_samples(
    samples: List[Dict],
    *,
    model=None,
    tokenizer=None,
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    use_gold_targets: bool = False,
    show_progress: bool = False,
    progress_desc: str = "tool_eval",
) -> Tuple[List[Dict], Dict]:
    if not use_gold_targets and (model is None or tokenizer is None):
        raise ValueError("model and tokenizer are required when use_gold_targets is False.")

    records = []
    progress_bar = tqdm(
        samples,
        total=len(samples),
        desc=progress_desc,
        dynamic_ncols=True,
        disable=not (show_progress and sys.stderr.isatty()),
    )
    for sample in progress_bar:
        if use_gold_targets:
            gold = normalize_message_for_tool_template(sample["target_message"])
            function = gold["tool_calls"][0]["function"] if gold.get("tool_calls") else {"name": None, "arguments": "{}"}
            raw_prediction_text = (
                (gold.get("content") or "").strip()
                + ("\n\n" if gold.get("content") else "")
                + "<tool_call>\n"
                + json.dumps({"name": function["name"], "arguments": function["arguments"]}, ensure_ascii=False)
                + "\n</tool_call>"
            )
            strict_message, strict_meta = parse_generated_tool_call(raw_prediction_text)
            bare_message, bare_meta = parse_generated_bare_json_tool_call(raw_prediction_text)
        else:
            raw_prediction_text = predict_with_model(
                model,
                tokenizer,
                sample,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            strict_message, strict_meta = parse_generated_tool_call(raw_prediction_text)
            bare_message, bare_meta = parse_generated_bare_json_tool_call(raw_prediction_text)

        records.append(
            evaluate_prediction(
                sample,
                strict_message,
                strict_meta,
                bare_message,
                bare_meta,
                raw_prediction_text,
            )
        )

    metrics = summarize(records)
    return records, metrics


def run_tool_use_eval(
    *,
    data_path: str,
    model=None,
    tokenizer=None,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    use_gold_targets: bool = False,
    output_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
    show_progress: bool = False,
    progress_desc: str = "tool_eval",
) -> Tuple[List[Dict], Dict]:
    samples = list(iter_jsonl(data_path))
    if max_samples is not None:
        samples = samples[:max_samples]

    records, metrics = evaluate_samples(
        samples,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_gold_targets=use_gold_targets,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )

    if output_path is not None:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

    if metrics_path is not None:
        metrics_path_obj = Path(metrics_path)
        metrics_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path_obj, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)

    return records, metrics


def main() -> None:
    args = parse_args()
    if not args.use_gold_targets and not args.model_name_or_path:
        raise ValueError("Either --use_gold_targets or --model_name_or_path must be provided.")

    model = None
    tokenizer = None
    if args.model_name_or_path:
        tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            model = model.float()

    records, metrics = run_tool_use_eval(
        data_path=args.data_path,
        model=model,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_gold_targets=args.use_gold_targets,
        output_path=args.output_path,
        metrics_path=args.metrics_path,
        show_progress=is_main_process_from_env(),
        progress_desc="tool_eval",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"results_saved_to: {args.output_path}")
    print(f"metrics_saved_to: {args.metrics_path}")


if __name__ == "__main__":
    main()
