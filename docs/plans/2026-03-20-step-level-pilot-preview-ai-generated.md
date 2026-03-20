# Step-Level Pilot Preview (AI-Generated)

## Files Saved

The current pilot artifacts have been saved to:

- [step_level_raw_pilot.jsonl](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/data/step_level_raw_pilot.jsonl)
- [step_level_canonical_pilot_v0.jsonl](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/data/step_level_canonical_pilot_v0.jsonl)

These are pilot files for inspection only.
They are not yet the final training dataset.

## Pilot Stats

Raw pilot:

- trajectory subset: 20
- step samples: 324
- repositories: 10
- target kinds:
  - `action_step`: 300
  - `stop_step`: 20
  - `thought_step`: 4

Canonical pilot `v0`:

- kept samples: 146
- repositories: 10
- canonical tool distribution:
  - `repo_search_text`: 53
  - `repo_read_range`: 44
  - `repo_find_files`: 23
  - `repo_list_dir`: 9
  - `repo_read_head`: 6
  - `task_done`: 6
  - `repo_read_file`: 2

Important note:

- The current canonical pilot keeps only samples whose target is cleanly mappable and whose prefix passes the current strict mapping rule.
- This is why `146 / 324` samples remain.
- The next likely refinement is `prefix_compact`, not trainer changes.

## Raw Sample Shape

Representative raw step sample:

```json
{
  "sample_id": "localstack__localstack-1066__step_000",
  "message_format": "raw",
  "repo": "localstack__localstack",
  "quality_bucket": "2_medium_f1_50-100",
  "step_idx": 0,
  "target_kind": "action_step",
  "prefix_messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "...issue text..."}
  ],
  "target_message": {
    "role": "assistant",
    "content": "Let me analyze this GitHub issue step by step ...",
    "tool_calls": [
      {
        "function": {
          "name": "bash",
          "arguments": "{\"command\":\"find /workspace -type f -name \\\"*.py\\\" | grep -E \\\"(ssl|cert|tls)\\\" | head -20\"}"
        }
      }
    ]
  },
  "metadata": {
    "raw_command": "find /workspace -type f -name \"*.py\" | grep -E \"(ssl|cert|tls)\" | head -20"
  }
}
```

Interpretation:

- `prefix_messages` contains the history before this step
- `target_message` is the next assistant turn to predict
- `metadata.raw_command` preserves the original shell form

## Canonical Sample Shape

Representative canonical step sample:

```json
{
  "sample_id": "localstack__localstack-1066__step_000",
  "message_format": "canonical_v0",
  "repo": "localstack__localstack",
  "quality_bucket": "2_medium_f1_50-100",
  "step_idx": 0,
  "target_kind": "action_step",
  "prefix_messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "...issue text..."}
  ],
  "target_message": {
    "role": "assistant",
    "content": "Let me analyze this GitHub issue step by step ...",
    "tool_calls": [
      {
        "function": {
          "name": "repo_find_files",
          "arguments": "{\"root_path\":\"/workspace\",\"glob\":\"*.py\",\"case_insensitive\":false,\"name_pattern\":\"(ssl|cert|tls)\",\"filter_is_regex\":true,\"filter_case_insensitive\":false,\"max_results\":20}"
        }
      }
    ]
  },
  "metadata": {
    "raw_command": "find /workspace -type f -name \"*.py\" | grep -E \"(ssl|cert|tls)\" | head -20",
    "canonical_tool_name": "repo_find_files",
    "canonical_tool_args": {
      "root_path": "/workspace",
      "glob": "*.py",
      "name_pattern": "(ssl|cert|tls)",
      "max_results": 20
    },
    "mapping_reason": "find_grep_head"
  }
}
```

Interpretation:

- raw bash has been converted into a structured tool family
- the original command is still preserved in metadata
- this is the version intended for `v0` training experiments

## Stop-Step Sample Shape

Representative stop sample:

```json
{
  "sample_id": "huggingface__transformers-1327__step_008",
  "target_kind": "stop_step",
  "target_message": {
    "role": "assistant",
    "content": "Now I have enough information to identify the exact locations ...",
    "tool_calls": [
      {
        "function": {
          "name": "task_done",
          "arguments": "{}"
        }
      }
    ]
  },
  "metadata": {
    "canonical_tool_name": "task_done",
    "is_terminal_step": true
  }
}
```

Interpretation:

- the stop decision is modeled as a normal assistant step
- this can later be used to train `continue` vs `task_done`

## Current Judgment

The saved pilot already proves three things:

1. step-level extraction works and the schema is stable
2. canonicalization works for a substantial subset of real search behavior
3. the main blocker is now prefix normalization, not whether the overall direction is valid
