import argparse
import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from process_data.trajectory_coldstart.canonicalize import canonicalize_assistant_message
from process_data.trajectory_coldstart.common import iter_jsonl, json_dumps, write_jsonl


WORKSPACE_PATH_RE = re.compile(r"(/workspace/[^\s:`]+)")
MATCH_LINE_RE = re.compile(r"^(?P<path>/workspace/[^:\s]+):(?P<line>\d+):(.*)$")
SYMBOL_RE_LIST = [
    re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"\bfunction:\s*([A-Za-z_][A-Za-z0-9_.]*)"),
]
ERROR_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"traceback",
        r"error",
        r"exception",
        r"no such file",
        r"command not found",
        r"permission denied",
        r"not found",
        r"failed",
    ]
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact prefix variants for step-level localization samples.")
    parser.add_argument("--input_path", default="sft/data/step_level_raw.jsonl")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--recent_pairs", type=int, default=2)
    parser.add_argument("--state_mode", choices=["on", "off"], default="on")
    parser.add_argument("--preview_path", default=None)
    return parser.parse_args()


def extract_task_block(prefix_messages: List[Dict]) -> Dict:
    system = ""
    user = ""
    for message in prefix_messages:
        if message.get("role") == "system" and not system:
            system = message.get("content") or ""
        elif message.get("role") == "user" and not user:
            user = message.get("content") or ""
        if system and user:
            break
    return {"system": system, "user": user}


def extract_tool_turns(prefix_messages: List[Dict]) -> List[Dict]:
    turns: List[Dict] = []
    current_turn: Dict | None = None

    for message in prefix_messages:
        role = message.get("role")
        if role in {"system", "user"}:
            continue
        if role == "assistant":
            if current_turn is not None:
                turns.append(current_turn)
            current_turn = {
                "assistant": copy.deepcopy(message),
                "tools": [],
            }
            if not (message.get("tool_calls") or []):
                turns.append(current_turn)
                current_turn = None
        elif role == "tool":
            if current_turn is None:
                continue
            current_turn["tools"].append(copy.deepcopy(message))

    if current_turn is not None:
        turns.append(current_turn)
    return turns


def safe_json_loads(raw: str | None) -> Dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def dedup_keep_order(items: List) -> List:
    seen = set()
    output = []
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if not isinstance(item, str) else item
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def clip_lines(lines: List[str], head: int, tail: int = 0, marker: str | None = None) -> List[str]:
    if len(lines) <= head + tail:
        return lines
    kept = lines[:head]
    omitted = len(lines) - head - tail
    if marker:
        kept.append(marker.format(omitted=omitted, total=len(lines)))
    if tail:
        kept.extend(lines[-tail:])
    return kept


def extract_paths(text: str) -> List[str]:
    paths = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = MATCH_LINE_RE.match(stripped)
        if match:
            paths.append(match.group("path"))
            continue
        path_match = WORKSPACE_PATH_RE.search(stripped)
        if path_match:
            paths.append(path_match.group(1))
    return dedup_keep_order(paths)


def extract_symbols(text: str) -> List[str]:
    symbols: List[str] = []
    for line in text.splitlines():
        for pattern in SYMBOL_RE_LIST:
            for match in pattern.findall(line):
                symbols.append(match)
    return dedup_keep_order(symbols)


def infer_error_summary(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if any(pattern.search(line) for pattern in ERROR_PATTERNS):
            return line[:240]
    return None


def clip_tool_result(tool_name: str, tool_args: Dict, content: str) -> Tuple[str, Dict]:
    lines = [line.rstrip() for line in content.splitlines() if line.strip()]
    original_line_count = len(lines)
    metadata = {
        "tool_name": tool_name,
        "original_line_count": original_line_count,
        "clipped": False,
    }

    if tool_name in {"repo_find_files", "repo_list_dir"}:
        path_lines = [line for line in lines if WORKSPACE_PATH_RE.search(line)]
        kept = clip_lines(
            path_lines or lines,
            head=12,
            tail=4,
            marker="TOOL_OUTPUT_TRUNCATED omitted={omitted} total_lines={total}",
        )
    elif tool_name == "repo_search_text":
        symbol_lines = [line for line in lines if any(pattern.search(line) for pattern in SYMBOL_RE_LIST)]
        first_lines = lines[:8]
        last_lines = lines[-2:] if len(lines) > 10 else []
        kept = dedup_keep_order(first_lines + symbol_lines + last_lines)
        if len(kept) < len(lines):
            kept = kept[:10] + [f"TOOL_OUTPUT_TRUNCATED omitted={max(len(lines) - len(kept), 0)} total_lines={len(lines)}"]
    elif tool_name in {"repo_read_range", "repo_read_head", "repo_read_file"}:
        kept = clip_lines(
            lines,
            head=40,
            tail=20,
            marker="TOOL_OUTPUT_TRUNCATED omitted={omitted} total_lines={total}",
        )
    else:
        kept = clip_lines(
            lines,
            head=20,
            tail=5,
            marker="TOOL_OUTPUT_TRUNCATED omitted={omitted} total_lines={total}",
        )

    metadata["clipped"] = len(kept) < len(lines)
    metadata["kept_line_count"] = len(kept)
    return "\n".join(kept), metadata


def select_recent_and_older_turns(turns: List[Dict], recent_pairs: int) -> Tuple[List[Dict], List[Dict]]:
    if recent_pairs <= 0:
        return turns, []
    tool_turns = [turn for turn in turns if (turn["assistant"].get("tool_calls") or [])]
    recent_tool_turns = tool_turns[-recent_pairs:]
    recent_ids = {id(turn) for turn in recent_tool_turns}
    older_turns = []
    recent_turns = []
    for turn in turns:
        if id(turn) in recent_ids:
            recent_turns.append(turn)
        else:
            older_turns.append(turn)
    return older_turns, recent_turns


def recent_turn_to_canonical_messages(turn: Dict) -> Tuple[List[Dict] | None, Dict]:
    canonical_assistant, mappings = canonicalize_assistant_message(turn["assistant"])
    if canonical_assistant is None:
        reason = mappings[-1]["reason"] if mappings else "recent_unmapped"
        return None, {"reason": reason}

    messages = [canonical_assistant]
    tool_calls = canonical_assistant.get("tool_calls") or []
    tool_name_by_id = {
        tool_call["id"]: tool_call["function"]["name"]
        for tool_call in tool_calls
    }
    tool_args_by_id = {
        tool_call["id"]: safe_json_loads(tool_call["function"]["arguments"])
        for tool_call in tool_calls
    }
    clip_stats = []
    for tool_message in turn.get("tools", []):
        tool_call_id = tool_message.get("tool_call_id")
        tool_name = tool_name_by_id.get(tool_call_id)
        tool_args = tool_args_by_id.get(tool_call_id, {})
        if not tool_name:
            continue
        clipped_content, clip_meta = clip_tool_result(tool_name, tool_args, tool_message.get("content") or "")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": clipped_content,
            }
        )
        clip_stats.append(clip_meta)
    return messages, {
        "reason": "mapped",
        "tool_name": mappings[0]["tool_name"] if mappings else None,
        "clip_stats": clip_stats,
    }


def initial_state() -> Dict:
    return {
        "repo_root": "/workspace",
        "candidate_files_topk": [],
        "candidate_symbols_topk": [],
        "search_patterns_used": [],
        "opened_files": [],
        "recent_read_ranges": [],
        "evidence": [],
        "last_error": None,
        "stop_ready": False,
    }


def update_ranked_counter(counter: Dict[str, Dict], key: str, step_idx: int, weight: int) -> None:
    if not key:
        return
    entry = counter.setdefault(key, {"score": 0, "last_step": -1})
    entry["score"] += weight
    entry["last_step"] = max(entry["last_step"], step_idx)


def make_evidence_from_search(pattern: str, path: str, line: str | None = None) -> str:
    if line:
        return f"{path}:{line} matches pattern '{pattern}'"
    return f"{path} matches pattern '{pattern}'"


def build_state_block(older_turns: List[Dict], include_state: bool) -> Tuple[Dict, Dict]:
    state = initial_state()
    stats = {
        "older_turn_count": len(older_turns),
        "older_canonical_turn_count": 0,
        "older_unmapped_turn_count": 0,
    }
    if not include_state:
        return state, stats

    file_counter: Dict[str, Dict] = {}
    symbol_counter: Dict[str, Dict] = {}
    search_patterns: List[str] = []
    opened_files: List[str] = []
    recent_ranges: List[Dict] = []
    evidence: List[str] = []
    last_error = None

    for older_idx, turn in enumerate(older_turns):
        assistant = turn["assistant"]
        if not (assistant.get("tool_calls") or []):
            continue
        canonical_assistant, mappings = canonicalize_assistant_message(assistant)
        if canonical_assistant is None:
            stats["older_unmapped_turn_count"] += 1
            continue
        stats["older_canonical_turn_count"] += 1

        tool_calls = canonical_assistant.get("tool_calls") or []
        tool_name_by_id = {tool_call["id"]: tool_call["function"]["name"] for tool_call in tool_calls}
        tool_args_by_id = {tool_call["id"]: safe_json_loads(tool_call["function"]["arguments"]) for tool_call in tool_calls}
        tool_result_by_id = {tool.get("tool_call_id"): tool.get("content") or "" for tool in turn.get("tools", [])}

        for tool_call in tool_calls:
            call_id = tool_call["id"]
            tool_name = tool_name_by_id.get(call_id)
            tool_args = tool_args_by_id.get(call_id, {})
            tool_result = tool_result_by_id.get(call_id, "")

            if tool_name == "repo_find_files":
                for path in extract_paths(tool_result):
                    update_ranked_counter(file_counter, path, older_idx, weight=1)
                if tool_args.get("name_pattern"):
                    search_patterns.append(tool_args["name_pattern"])

            elif tool_name == "repo_list_dir":
                dir_path = tool_args.get("dir_path")
                if dir_path:
                    update_ranked_counter(file_counter, dir_path, older_idx, weight=1)
                for path in extract_paths(tool_result):
                    update_ranked_counter(file_counter, path, older_idx, weight=1)
                if tool_args.get("filter_pattern"):
                    search_patterns.append(tool_args["filter_pattern"])

            elif tool_name == "repo_search_text":
                pattern = tool_args.get("pattern")
                if pattern:
                    search_patterns.append(pattern)
                lines = [line for line in tool_result.splitlines() if line.strip()]
                for line in lines:
                    match = MATCH_LINE_RE.match(line.strip())
                    if match:
                        path = match.group("path")
                        update_ranked_counter(file_counter, path, older_idx, weight=2)
                        if pattern:
                            evidence.append(make_evidence_from_search(pattern, path, match.group("line")))
                for path in extract_paths(tool_result):
                    update_ranked_counter(file_counter, path, older_idx, weight=1)
                for symbol in extract_symbols(tool_result):
                    update_ranked_counter(symbol_counter, symbol, older_idx, weight=2)

            elif tool_name in {"repo_read_range", "repo_read_head", "repo_read_file"}:
                file_path = tool_args.get("file_path")
                if file_path:
                    opened_files.append(file_path)
                    update_ranked_counter(file_counter, file_path, older_idx, weight=3)
                start_line = tool_args.get("start_line")
                end_line = tool_args.get("end_line")
                if file_path and (start_line is not None or end_line is not None):
                    recent_ranges.append(
                        {
                            "file_path": file_path,
                            "start_line": start_line,
                            "end_line": end_line,
                        }
                    )
                symbols = extract_symbols(tool_result)
                for symbol in symbols:
                    update_ranked_counter(symbol_counter, symbol, older_idx, weight=2)
                if file_path:
                    if symbols:
                        evidence.append(f"read {file_path} reveals symbols: {', '.join(symbols[:2])}")
                    elif start_line is not None or end_line is not None:
                        evidence.append(f"read {file_path}:{start_line}-{end_line}")
                    else:
                        evidence.append(f"read {file_path}")

            error_summary = infer_error_summary(tool_result)
            if error_summary:
                last_error = {
                    "tool_name": tool_name,
                    "args": tool_args,
                    "error_summary": error_summary,
                    "older_turn_index": older_idx,
                }

    ranked_files = sorted(file_counter.items(), key=lambda item: (-item[1]["score"], -item[1]["last_step"], item[0]))
    ranked_symbols = sorted(symbol_counter.items(), key=lambda item: (-item[1]["score"], -item[1]["last_step"], item[0]))

    state["candidate_files_topk"] = [path for path, _ in ranked_files[:10]]
    state["candidate_symbols_topk"] = [symbol for symbol, _ in ranked_symbols[:10]]
    state["search_patterns_used"] = dedup_keep_order(search_patterns)[:10]
    state["opened_files"] = dedup_keep_order(opened_files)[-10:]
    state["recent_read_ranges"] = dedup_keep_order(recent_ranges)[-5:]
    state["evidence"] = dedup_keep_order(evidence)[:8]
    state["last_error"] = last_error
    state["stop_ready"] = bool(state["candidate_files_topk"] and (state["opened_files"] or state["candidate_symbols_topk"]))

    return state, stats


def compact_sample(sample: Dict, recent_pairs: int, include_state: bool) -> Tuple[Dict | None, Dict]:
    task_block = extract_task_block(sample.get("prefix_messages") or [])
    turns = extract_tool_turns(sample.get("prefix_messages") or [])
    older_turns, recent_turns = select_recent_and_older_turns(turns, recent_pairs=recent_pairs)

    recent_window_messages: List[Dict] = []
    recent_clip_stats: List[Dict] = []
    for turn in recent_turns:
        canonical_messages, meta = recent_turn_to_canonical_messages(turn)
        if canonical_messages is None:
            return None, {"status": "dropped", "reason": f"recent_{meta['reason']}"}
        recent_window_messages.extend(canonical_messages)
        recent_clip_stats.extend(meta.get("clip_stats", []))

    canonical_target, target_mappings = canonicalize_assistant_message(sample.get("target_message") or {})
    if canonical_target is None:
        reason = target_mappings[-1]["reason"] if target_mappings else "unmapped_target"
        return None, {"status": "dropped", "reason": reason}

    state_block, state_stats = build_state_block(older_turns, include_state=include_state)

    compact = copy.deepcopy(sample)
    compact["message_format"] = "compact_v0"
    compact["task_block"] = task_block
    compact["state_block"] = state_block
    compact["recent_window_messages"] = recent_window_messages
    compact["target_message"] = canonical_target
    compact["metadata"] = {
        **sample.get("metadata", {}),
        "canonical_tool_name": target_mappings[0]["tool_name"] if target_mappings else None,
        "canonical_tool_args": json_dumps(target_mappings[0]["tool_args"]) if target_mappings else None,
        "mapping_status": "mapped",
        "mapping_reason": target_mappings[0]["reason"] if target_mappings else None,
        "compact_recent_pairs": recent_pairs,
        "state_mode": "on" if include_state else "off",
        "recent_window_message_count": len(recent_window_messages),
        "recent_tool_result_clips": recent_clip_stats,
        **state_stats,
    }
    compact.pop("prefix_messages", None)
    return compact, {"status": "mapped"}


def summarize_stats(samples: List[Dict], dropped: Counter) -> Dict:
    tool_counter = Counter(sample["metadata"].get("canonical_tool_name") for sample in samples if sample["metadata"].get("canonical_tool_name"))
    return {
        "kept_samples": len(samples),
        "canonical_tool_distribution": dict(tool_counter),
        "drop_reasons": dict(dropped),
        "avg_recent_window_message_count": round(
            sum(sample["metadata"].get("recent_window_message_count", 0) for sample in samples) / len(samples), 2
        ) if samples else 0.0,
        "avg_older_unmapped_turns": round(
            sum(sample["metadata"].get("older_unmapped_turn_count", 0) for sample in samples) / len(samples), 2
        ) if samples else 0.0,
    }


def build_preview(samples: List[Dict], dropped_examples: List[Dict], output_path: str, recent_pairs: int, state_mode: str) -> None:
    preview = {
        "description": "Representative compact prefix examples.",
        "variant": {
            "recent_pairs": recent_pairs,
            "state_mode": state_mode,
        },
        "kept_examples": [],
        "dropped_examples": dropped_examples,
    }

    for key, predicate in [
        ("compact_action_start", lambda s: s["target_kind"] == "action_step" and s["metadata"].get("num_prior_tool_messages", 0) == 0),
        ("compact_action_mid", lambda s: s["target_kind"] == "action_step" and s["metadata"].get("num_prior_tool_messages", 0) > 0),
        ("compact_stop", lambda s: s["target_kind"] == "stop_step"),
    ]:
        for sample in samples:
            if predicate(sample):
                preview["kept_examples"].append({"example_type": key, "sample": sample})
                break

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    include_state = args.state_mode == "on"

    kept_samples: List[Dict] = []
    dropped_counter: Counter = Counter()
    dropped_examples: List[Dict] = []
    seen_drop_reasons = set()

    for sample in iter_jsonl(args.input_path):
        compact, status = compact_sample(sample, recent_pairs=args.recent_pairs, include_state=include_state)
        if compact is None:
            reason = status.get("reason", "unknown")
            dropped_counter[reason] += 1
            if reason not in seen_drop_reasons and len(dropped_examples) < 6:
                dropped_examples.append(
                    {
                        "reason": reason,
                        "sample_id": sample["sample_id"],
                        "target_kind": sample["target_kind"],
                        "raw_command": sample.get("metadata", {}).get("raw_command"),
                        "metadata": sample.get("metadata", {}),
                    }
                )
                seen_drop_reasons.add(reason)
            continue
        kept_samples.append(compact)

    write_jsonl(args.output_path, kept_samples)
    stats = summarize_stats(kept_samples, dropped_counter)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"saved_to: {Path(args.output_path)}")

    if args.preview_path:
        build_preview(kept_samples, dropped_examples, args.preview_path, args.recent_pairs, args.state_mode)
        print(f"preview_saved_to: {Path(args.preview_path)}")


if __name__ == "__main__":
    main()
