import copy
import json


CANONICAL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "repo_find_files",
            "description": "Search for candidate repository file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "root_path": {"type": "string"},
                    "glob": {"type": "string"},
                    "case_insensitive": {"type": "boolean"},
                    "name_pattern": {"type": "string"},
                    "filter_is_regex": {"type": "boolean"},
                    "filter_case_insensitive": {"type": "boolean"},
                    "max_results": {"type": "integer"},
                },
                "required": ["root_path", "glob"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_list_dir",
            "description": "Inspect a repository directory and list entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string"},
                    "filter_pattern": {"type": "string"},
                    "is_regex": {"type": "boolean"},
                    "case_insensitive": {"type": "boolean"},
                    "max_results": {"type": "integer"},
                },
                "required": ["dir_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_search_text",
            "description": "Search file contents inside the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "is_regex": {"type": "boolean"},
                    "case_insensitive": {"type": "boolean"},
                    "before_context": {"type": "integer"},
                    "after_context": {"type": "integer"},
                    "glob": {"type": "string"},
                    "max_results": {"type": "integer"},
                    "max_scan_lines": {"type": "integer"},
                },
                "required": ["pattern", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_read_head",
            "description": "Read the first N lines of a repository file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "max_lines": {"type": "integer"},
                },
                "required": ["file_path", "max_lines"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_read_range",
            "description": "Read a line range from a repository file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["file_path", "start_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_read_file",
            "description": "Read a whole repository file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_done",
            "description": "Finish localization when enough evidence has been collected.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


ALLOWED_TOOL_SCHEMAS = {
    "repo_find_files": {
        "required": {"root_path": str, "glob": str},
        "optional": {
            "case_insensitive": bool,
            "name_pattern": str,
            "filter_is_regex": bool,
            "filter_case_insensitive": bool,
            "max_results": int,
        },
    },
    "repo_list_dir": {
        "required": {"dir_path": str},
        "optional": {
            "filter_pattern": str,
            "is_regex": bool,
            "case_insensitive": bool,
            "max_results": int,
        },
    },
    "repo_search_text": {
        "required": {"pattern": str, "path": str},
        "optional": {
            "recursive": bool,
            "is_regex": bool,
            "case_insensitive": bool,
            "before_context": int,
            "after_context": int,
            "glob": str,
            "max_results": int,
            "max_scan_lines": int,
        },
    },
    "repo_read_head": {
        "required": {"file_path": str, "max_lines": int},
        "optional": {},
    },
    "repo_read_range": {
        "required": {"file_path": str, "start_line": int},
        "optional": {"end_line": int},
    },
    "repo_read_file": {
        "required": {"file_path": str},
        "optional": {},
    },
    "task_done": {
        "required": {},
        "optional": {},
    },
}


def _normalize_tool_call(tool_call):
    normalized = copy.deepcopy(tool_call)
    function = normalized.get("function") or {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            function["arguments"] = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    normalized["function"] = function
    return normalized


def normalize_message_for_tool_template(message):
    normalized = copy.deepcopy(message)
    tool_calls = normalized.get("tool_calls") or []
    if tool_calls:
        normalized["tool_calls"] = [_normalize_tool_call(tool_call) for tool_call in tool_calls]
    return normalized


def normalize_messages_for_tool_template(messages):
    return [normalize_message_for_tool_template(message) for message in messages]
