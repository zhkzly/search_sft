import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from process_data.trajectory_coldstart.common import (
    DEFAULT_CATEGORIES,
    DEFAULT_LENGTH_BUCKETS,
    clean_tool_result,
    collect_trajectory_files,
    derive_instance_metadata,
    format_tool_call,
    parse_csv_arg,
    write_jsonl,
)


def convert_input_messages(messages: List[Dict]) -> List[Dict]:
    converted: List[Dict] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "user"} and content:
            converted.append({"role": role, "content": content})
        tool_result = message.get("tool_result")
        if tool_result:
            tool_call_id = tool_result.get("call_id") or f"tool_{len(converted)}"
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": clean_tool_result(tool_result.get("result", "")),
                }
            )
    return converted


def convert_response_to_assistant(response: Dict, step_idx: int) -> Dict:
    assistant_message = {"role": "assistant", "content": response.get("content") or ""}
    raw_tool_calls = response.get("tool_calls") or []
    if raw_tool_calls:
        assistant_message["tool_calls"] = [
            format_tool_call(raw_tool_call, fallback_id=f"call_{step_idx}_{idx}")
            for idx, raw_tool_call in enumerate(raw_tool_calls)
        ]
    return assistant_message


def classify_target(assistant_message: Dict) -> str:
    tool_calls = assistant_message.get("tool_calls") or []
    tool_names = [tool_call["function"]["name"] for tool_call in tool_calls]
    if "task_done" in tool_names:
        return "stop_step"
    if tool_calls:
        return "action_step"
    return "thought_step"


def extract_raw_command_metadata(assistant_message: Dict) -> Dict:
    tool_calls = assistant_message.get("tool_calls") or []
    raw_tool_names: List[str] = []
    raw_commands: List[str] = []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        raw_tool_names.append(function.get("name", ""))
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        command = arguments.get("command")
        if command:
            raw_commands.append(command)
    return {
        "raw_tool_names": raw_tool_names,
        "raw_commands": raw_commands,
        "raw_command": raw_commands[0] if len(raw_commands) == 1 else None,
    }


def build_samples(args: argparse.Namespace) -> List[Dict]:
    categories = parse_csv_arg(args.categories, DEFAULT_CATEGORIES)
    length_buckets = parse_csv_arg(args.length_buckets, DEFAULT_LENGTH_BUCKETS)
    trajectory_files = collect_trajectory_files(
        base_dir=args.base_dir,
        categories=categories,
        length_buckets=length_buckets,
        max_trajectories=args.max_trajectories,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    samples: List[Dict] = []
    for traj_path, quality_bucket, length_bucket in trajectory_files:
        with open(traj_path, "r", encoding="utf-8") as handle:
            traj_data = json.load(handle)

        interactions = traj_data.get("llm_interactions") or []
        if not interactions:
            continue

        instance_meta = derive_instance_metadata(args.base_dir, traj_path)
        conversation: List[Dict] = convert_input_messages(interactions[0].get("input_messages") or [])

        for step_idx, interaction in enumerate(interactions):
            if step_idx > 0:
                conversation.extend(convert_input_messages(interaction.get("input_messages") or []))

            target_message = convert_response_to_assistant(interaction.get("response") or {}, step_idx)
            if not target_message.get("content") and not target_message.get("tool_calls"):
                continue

            target_kind = classify_target(target_message)
            raw_meta = extract_raw_command_metadata(target_message)
            samples.append(
                {
                    "sample_id": f"{instance_meta['instance_id']}__step_{step_idx:03d}",
                    "message_format": "raw",
                    "repo": instance_meta["repo"],
                    "instance_id": instance_meta["instance_id"],
                    "quality_bucket": quality_bucket,
                    "length_bucket": length_bucket,
                    "source_path": instance_meta["source_path"],
                    "step_idx": step_idx,
                    "target_kind": target_kind,
                    "prefix_messages": copy.deepcopy(conversation),
                    "target_message": target_message,
                    "metadata": {
                        "trajectory_success": traj_data.get("success"),
                        "max_steps": traj_data.get("max_steps"),
                        "is_terminal_step": target_kind == "stop_step",
                        "num_prior_messages": len(conversation),
                        "num_prior_tool_messages": sum(1 for msg in conversation if msg.get("role") == "tool"),
                        **raw_meta,
                    },
                }
            )
            conversation.append(copy.deepcopy(target_message))
    return samples


def print_stats(samples: List[Dict]) -> None:
    repo_counter = Counter(sample["repo"] for sample in samples)
    target_counter = Counter(sample["target_kind"] for sample in samples)
    raw_tool_counter = Counter()
    prefix_lengths = []
    target_lengths = []
    for sample in samples:
        prefix_lengths.append(len(sample["prefix_messages"]))
        target_lengths.append(len(sample["target_message"].get("content", "")))
        raw_tool_counter.update(sample["metadata"].get("raw_tool_names") or [])

    print(f"samples: {len(samples)}")
    print(f"repos: {len(repo_counter)}")
    print(f"target_kinds: {dict(target_counter)}")
    print(f"raw_tool_names: {dict(raw_tool_counter)}")
    if prefix_lengths:
        print(f"avg_prefix_messages: {sum(prefix_lengths) / len(prefix_lengths):.2f}")
    if target_lengths:
        print(f"avg_target_content_chars: {sum(target_lengths) / len(target_lengths):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build step-level raw pilot data from localization trajectories.")
    parser.add_argument("--base_dir", default="classified_by_f1")
    parser.add_argument("--output_path", default="sft/data/step_level_raw_pilot.jsonl")
    parser.add_argument("--categories", default="2_medium_f1_50-100,3_perfect_match_100")
    parser.add_argument("--length_buckets", default="short_lt100k,medium_100k_300k")
    parser.add_argument("--max_trajectories", type=int, default=100)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = build_samples(args)
    write_jsonl(args.output_path, samples)
    print_stats(samples)
    print(f"saved_to: {Path(args.output_path)}")


if __name__ == "__main__":
    main()
