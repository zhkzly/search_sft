import argparse
import copy
import json
import re
import shlex
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from process_data.trajectory_coldstart.common import iter_jsonl, json_dumps, write_jsonl


REGEX_HINT_PATTERN = re.compile(r"(\\\||\.\*|[\[\]\(\)\?\+\{\}])")


def looks_like_regex(pattern: str | None) -> bool:
    if not pattern:
        return False
    return bool(REGEX_HINT_PATTERN.search(pattern))


def shell_split(command: str) -> List[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|")
    lexer.whitespace_split = True
    return list(lexer)


def split_pipeline(command: str) -> Tuple[List[List[str]] | None, str | None]:
    if any(token in command for token in ("&&", "||", ";", "$(", "`")):
        return None, "unsupported_shell_control"
    try:
        tokens = shell_split(command)
    except ValueError:
        return None, "shell_parse_error"
    if not tokens:
        return None, "empty_command"
    segments: List[List[str]] = []
    current: List[str] = []
    for token in tokens:
        if token == "|":
            if not current:
                return None, "empty_pipeline_segment"
            segments.append(current)
            current = []
        else:
            current.append(token)
    if not current:
        return None, "empty_pipeline_segment"
    segments.append(current)
    return segments, None


def parse_head_tokens(tokens: List[str]) -> Dict | None:
    if not tokens or tokens[0] != "head":
        return None
    max_lines = None
    file_path = None
    idx = 1
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("-") and token[1:].isdigit():
            max_lines = int(token[1:])
        elif token.startswith("-"):
            return None
        else:
            file_path = token
        idx += 1
    if max_lines is None:
        max_lines = 10
    return {"max_lines": max_lines, "file_path": file_path}


def parse_ls_tokens(tokens: List[str]) -> Dict | None:
    if not tokens or tokens[0] != "ls":
        return None
    dir_path = None
    for token in tokens[1:]:
        if token.startswith("/"):
            dir_path = token
    if not dir_path:
        return None
    return {"dir_path": dir_path}


def parse_cat_tokens(tokens: List[str]) -> Dict | None:
    if len(tokens) != 2 or tokens[0] != "cat":
        return None
    if not tokens[1].startswith("/"):
        return None
    return {"file_path": tokens[1]}


def parse_find_tokens(tokens: List[str]) -> Dict | None:
    if not tokens or tokens[0] != "find":
        return None
    root_path = tokens[1] if len(tokens) > 1 and tokens[1].startswith("/") else None
    if not root_path:
        return None
    glob = None
    case_insensitive = False
    idx = 2
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"-name", "-iname"} and idx + 1 < len(tokens):
            glob = tokens[idx + 1]
            case_insensitive = token == "-iname"
            idx += 2
            continue
        idx += 1
    return {
        "root_path": root_path,
        "glob": glob or "*",
        "case_insensitive": case_insensitive,
    }


def parse_sed_tokens(tokens: List[str]) -> Dict | None:
    if len(tokens) < 4 or tokens[0] != "sed" or tokens[1] != "-n":
        return None
    expr = tokens[2]
    file_path = tokens[3]
    if not file_path.startswith("/"):
        return None
    match = re.fullmatch(r"(\d+),(\d+)p", expr)
    if match:
        return {
            "file_path": file_path,
            "start_line": int(match.group(1)),
            "end_line": int(match.group(2)),
        }
    match = re.fullmatch(r"(\d+),\$p", expr)
    if match:
        return {
            "file_path": file_path,
            "start_line": int(match.group(1)),
            "end_line": None,
        }
    return None


def parse_grep_tokens(tokens: List[str], default_path: str | None = None) -> Dict | None:
    if not tokens or tokens[0] != "grep":
        return None
    recursive = False
    case_insensitive = False
    is_regex = False
    before_context = 0
    after_context = 0
    glob = None
    positional: List[str] = []

    idx = 1
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"-r", "-R"}:
            recursive = True
        elif token == "-i":
            case_insensitive = True
        elif token == "-E":
            is_regex = True
        elif token == "-n":
            pass
        elif token in {"-A", "-B", "-C"} and idx + 1 < len(tokens):
            value = int(tokens[idx + 1])
            if token == "-A":
                after_context = value
            elif token == "-B":
                before_context = value
            else:
                before_context = value
                after_context = value
            idx += 1
        elif re.fullmatch(r"-A\d+", token):
            after_context = int(token[2:])
        elif re.fullmatch(r"-B\d+", token):
            before_context = int(token[2:])
        elif re.fullmatch(r"-C\d+", token):
            value = int(token[2:])
            before_context = value
            after_context = value
        elif token.startswith("--include="):
            glob = token.split("=", 1)[1]
        elif token == "--include" and idx + 1 < len(tokens):
            glob = tokens[idx + 1]
            idx += 1
        elif token.startswith("-") and set(token[1:]).issubset({"r", "R", "i", "n", "E"}):
            recursive = recursive or "r" in token[1:] or "R" in token[1:]
            case_insensitive = case_insensitive or "i" in token[1:]
            is_regex = is_regex or "E" in token[1:]
        elif token.startswith("-"):
            return None
        else:
            positional.append(token)
        idx += 1

    if not positional:
        return None
    pattern = positional[0]
    path = positional[1] if len(positional) > 1 else default_path
    if not path:
        return None
    return {
        "pattern": pattern,
        "path": path,
        "recursive": recursive,
        "is_regex": is_regex or looks_like_regex(pattern),
        "case_insensitive": case_insensitive,
        "before_context": before_context,
        "after_context": after_context,
        "glob": glob,
    }


def make_mapped(tool_name: str, args: Dict, reason: str) -> Dict:
    cleaned_args = {key: value for key, value in args.items() if value is not None}
    return {
        "status": "mapped",
        "tool_name": tool_name,
        "tool_args": cleaned_args,
        "reason": reason,
    }


def make_unmapped(reason: str) -> Dict:
    return {"status": "unmapped", "reason": reason}


def canonicalize_command(command: str) -> Dict:
    segments, error = split_pipeline(command)
    if error:
        return make_unmapped(error)
    assert segments is not None

    first = segments[0]
    if first[0] == "task_done":
        return make_mapped("task_done", {}, "already_task_done")

    if first[0] == "find":
        parsed_find = parse_find_tokens(first)
        if not parsed_find:
            return make_unmapped("find_parse_failed")
        if len(segments) == 1:
            return make_mapped("repo_find_files", parsed_find, "find_only")
        if len(segments) == 2 and segments[1][0] == "head":
            parsed_head = parse_head_tokens(segments[1])
            if not parsed_head:
                return make_unmapped("find_head_parse_failed")
            parsed_find["max_results"] = parsed_head["max_lines"]
            return make_mapped("repo_find_files", parsed_find, "find_head")
        if segments[1][0] == "grep":
            parsed_grep = parse_grep_tokens(segments[1], default_path=parsed_find["root_path"])
            if not parsed_grep:
                return make_unmapped("find_grep_parse_failed")
            parsed_find.update(
                {
                    "name_pattern": parsed_grep["pattern"],
                    "filter_is_regex": parsed_grep["is_regex"],
                    "filter_case_insensitive": parsed_grep["case_insensitive"],
                }
            )
            if len(segments) == 2:
                return make_mapped("repo_find_files", parsed_find, "find_grep")
            if len(segments) == 3 and segments[2][0] == "head":
                parsed_head = parse_head_tokens(segments[2])
                if not parsed_head:
                    return make_unmapped("find_grep_head_parse_failed")
                parsed_find["max_results"] = parsed_head["max_lines"]
                return make_mapped("repo_find_files", parsed_find, "find_grep_head")
        return make_unmapped("unsupported_find_pipeline")

    if first[0] == "ls":
        parsed_ls = parse_ls_tokens(first)
        if not parsed_ls:
            return make_unmapped("ls_parse_failed")
        if len(segments) == 1:
            return make_mapped("repo_list_dir", parsed_ls, "ls_only")
        if len(segments) >= 2 and segments[1][0] == "grep":
            parsed_grep = parse_grep_tokens(segments[1], default_path=parsed_ls["dir_path"])
            if not parsed_grep:
                return make_unmapped("ls_grep_parse_failed")
            parsed_ls.update(
                {
                    "filter_pattern": parsed_grep["pattern"],
                    "is_regex": parsed_grep["is_regex"],
                    "case_insensitive": parsed_grep["case_insensitive"],
                }
            )
            if len(segments) == 2:
                return make_mapped("repo_list_dir", parsed_ls, "ls_grep")
            if len(segments) == 3 and segments[2][0] == "head":
                parsed_head = parse_head_tokens(segments[2])
                if not parsed_head:
                    return make_unmapped("ls_grep_head_parse_failed")
                parsed_ls["max_results"] = parsed_head["max_lines"]
                return make_mapped("repo_list_dir", parsed_ls, "ls_grep_head")
        if len(segments) == 2 and segments[1][0] == "head":
            parsed_head = parse_head_tokens(segments[1])
            if not parsed_head:
                return make_unmapped("ls_head_parse_failed")
            parsed_ls["max_results"] = parsed_head["max_lines"]
            return make_mapped("repo_list_dir", parsed_ls, "ls_head")
        return make_unmapped("unsupported_ls_pipeline")

    if first[0] == "grep":
        parsed_grep = parse_grep_tokens(first)
        if not parsed_grep:
            return make_unmapped("grep_parse_failed")
        if len(segments) == 1:
            return make_mapped("repo_search_text", parsed_grep, "grep_only")
        if len(segments) == 2 and segments[1][0] == "head":
            parsed_head = parse_head_tokens(segments[1])
            if not parsed_head:
                return make_unmapped("grep_head_parse_failed")
            parsed_grep["max_results"] = parsed_head["max_lines"]
            return make_mapped("repo_search_text", parsed_grep, "grep_head")
        return make_unmapped("unsupported_grep_pipeline")

    if first[0] == "cat":
        parsed_cat = parse_cat_tokens(first)
        if not parsed_cat:
            return make_unmapped("cat_parse_failed")
        if len(segments) == 1:
            return make_mapped("repo_read_file", parsed_cat, "cat_only")
        if len(segments) == 2 and segments[1][0] == "head":
            parsed_head = parse_head_tokens(segments[1])
            if not parsed_head:
                return make_unmapped("cat_head_parse_failed")
            return make_mapped(
                "repo_read_head",
                {"file_path": parsed_cat["file_path"], "max_lines": parsed_head["max_lines"]},
                "cat_head",
            )
        if len(segments) == 2 and segments[1][0] == "grep":
            parsed_grep = parse_grep_tokens(segments[1], default_path=parsed_cat["file_path"])
            if not parsed_grep:
                return make_unmapped("cat_grep_parse_failed")
            return make_mapped("repo_search_text", parsed_grep, "cat_grep")
        return make_unmapped("unsupported_cat_pipeline")

    if first[0] == "head":
        parsed_head = parse_head_tokens(first)
        if not parsed_head or not parsed_head.get("file_path"):
            return make_unmapped("head_parse_failed")
        if len(segments) == 1:
            return make_mapped("repo_read_head", parsed_head, "head_only")
        if len(segments) == 2 and segments[1][0] == "grep":
            parsed_grep = parse_grep_tokens(segments[1], default_path=parsed_head["file_path"])
            if not parsed_grep:
                return make_unmapped("head_grep_parse_failed")
            parsed_grep["max_scan_lines"] = parsed_head["max_lines"]
            return make_mapped("repo_search_text", parsed_grep, "head_grep")
        return make_unmapped("unsupported_head_pipeline")

    if first[0] == "sed":
        parsed_sed = parse_sed_tokens(first)
        if not parsed_sed:
            return make_unmapped("sed_parse_failed")
        if len(segments) == 1:
            return make_mapped("repo_read_range", parsed_sed, "sed_only")
        if len(segments) == 2 and segments[1][0] == "head" and parsed_sed["end_line"] is None:
            parsed_head = parse_head_tokens(segments[1])
            if not parsed_head:
                return make_unmapped("sed_head_parse_failed")
            start_line = parsed_sed["start_line"]
            parsed_sed["end_line"] = start_line + parsed_head["max_lines"] - 1
            return make_mapped("repo_read_range", parsed_sed, "sed_head")
        return make_unmapped("unsupported_sed_pipeline")

    return make_unmapped("unsupported_command_family")


def canonicalize_tool_call(tool_call: Dict) -> Tuple[Dict | None, Dict]:
    function = tool_call.get("function", {})
    tool_name = function.get("name", "")
    if tool_name == "task_done":
        return (
            {
                "id": tool_call.get("id"),
                "type": "function",
                "function": {"name": "task_done", "arguments": "{}"},
            },
            make_mapped("task_done", {}, "task_done"),
        )
    if tool_name != "bash":
        return None, make_unmapped("unsupported_source_tool")

    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        return None, make_unmapped("invalid_tool_arguments_json")
    command = arguments.get("command")
    if not command:
        return None, make_unmapped("missing_command")

    mapping = canonicalize_command(command)
    if mapping["status"] != "mapped":
        return None, mapping
    return (
        {
            "id": tool_call.get("id"),
            "type": "function",
            "function": {
                "name": mapping["tool_name"],
                "arguments": json_dumps(mapping["tool_args"]),
            },
        },
        mapping,
    )


def canonicalize_assistant_message(message: Dict) -> Tuple[Dict | None, List[Dict]]:
    canonical_message = {"role": "assistant", "content": message.get("content", "")}
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return canonical_message, []

    canonical_tool_calls = []
    mappings = []
    for tool_call in tool_calls:
        canonical_tool_call, mapping = canonicalize_tool_call(tool_call)
        mappings.append(mapping)
        if not canonical_tool_call:
            return None, mappings
        canonical_tool_calls.append(canonical_tool_call)
    canonical_message["tool_calls"] = canonical_tool_calls
    return canonical_message, mappings


def canonicalize_sample(sample: Dict, require_mapped_prefix: bool) -> Tuple[Dict | None, Dict]:
    prefix_messages = []
    prefix_mappings: List[Dict] = []
    for message in sample.get("prefix_messages", []):
        if message.get("role") != "assistant":
            prefix_messages.append(copy.deepcopy(message))
            continue
        canonical_message, mappings = canonicalize_assistant_message(message)
        prefix_mappings.extend(mappings)
        if canonical_message is None:
            if require_mapped_prefix:
                return None, {"status": "dropped", "reason": "unmapped_prefix"}
            return None, {"status": "dropped", "reason": "mixed_prefix"}
        prefix_messages.append(canonical_message)

    canonical_target, target_mappings = canonicalize_assistant_message(sample.get("target_message") or {})
    if canonical_target is None:
        reason = target_mappings[-1]["reason"] if target_mappings else "unmapped_target"
        return None, {"status": "dropped", "reason": reason}

    canonical_sample = copy.deepcopy(sample)
    canonical_sample["message_format"] = "canonical_v0"
    canonical_sample["prefix_messages"] = prefix_messages
    canonical_sample["target_message"] = canonical_target
    canonical_sample["metadata"] = {
        **sample.get("metadata", {}),
        "canonical_tool_name": target_mappings[0]["tool_name"] if target_mappings else None,
        "canonical_tool_args": json_dumps(target_mappings[0]["tool_args"]) if target_mappings else None,
        "mapping_status": "mapped",
        "mapping_reason": target_mappings[0]["reason"] if target_mappings else None,
        "prefix_tool_calls_canonicalized": bool(prefix_mappings or not sample.get("prefix_messages")),
    }
    return canonical_sample, {"status": "mapped", "tool_name": canonical_sample["metadata"]["canonical_tool_name"]}


def print_stats(kept_samples: List[Dict], counters: Counter) -> None:
    tool_counter = Counter(sample["metadata"].get("canonical_tool_name") for sample in kept_samples if sample["metadata"].get("canonical_tool_name"))
    print(f"kept_samples: {len(kept_samples)}")
    print(f"canonical_tool_distribution: {dict(tool_counter)}")
    print(f"mapping_outcomes: {dict(counters)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map raw step-level bash samples into canonical localization tools.")
    parser.add_argument("--input_path", default="sft/data/step_level_raw.jsonl")
    parser.add_argument("--output_path", default="sft/data/step_level_canonical_v0.jsonl")
    parser.add_argument("--require_mapped_prefix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kept_samples: List[Dict] = []
    counters: Counter = Counter()

    for sample in iter_jsonl(args.input_path):
        canonical_sample, status = canonicalize_sample(sample, require_mapped_prefix=args.require_mapped_prefix)
        counters[status["status"] if status else "unknown"] += 1
        if status.get("reason"):
            counters[f"reason::{status['reason']}"] += 1
        if canonical_sample is None:
            continue
        kept_samples.append(canonical_sample)

    write_jsonl(args.output_path, kept_samples)
    print_stats(kept_samples, counters)
    print(f"saved_to: {Path(args.output_path)}")


if __name__ == "__main__":
    main()
