# Prefix Compact Execution Plan (AI-Generated)

## Goal

Build a deterministic `prefix_compact` pipeline for step-level localization training data.

The purpose is:

- preserve tool-use supervision
- reduce noisy or non-canonicalizable history
- improve canonical sample retention
- keep the training interface inspectable

This plan does **not** introduce LM summarization in `v0`.

## Non-Goals

This plan is not trying to:

- optimize final model quality in one pass
- solve all parser coverage issues first
- redesign the trainer before data is stable
- introduce free-form natural-language memory summaries

## Core Principle

The compact prefix should be:

1. `Task Block`
2. `State Block`
3. `Recent Evidence Window`

Where:

- `Task Block` preserves the task and system constraints
- `State Block` stores structured facts extracted from older history
- `Recent Evidence Window` keeps the most recent canonical assistant/tool traces

## Execution Order

The correct order is:

1. freeze compaction schema
2. implement deterministic compactors
3. generate compact pilot variants
4. inspect sample content manually
5. compare retention and compactness metrics
6. choose one `v0` compact format
7. only then wire the trainer to the compact data

Do not modify the trainer before step 6.

## Phase 1: Freeze the `prefix_compact` schema

### Decision 1: final compact sample structure

Each compact sample should contain:

```json
{
  "sample_id": "...",
  "message_format": "compact_v0",
  "repo": "...",
  "instance_id": "...",
  "quality_bucket": "...",
  "length_bucket": "...",
  "step_idx": 0,
  "target_kind": "action_step",
  "task_block": {
    "system": "...",
    "user": "..."
  },
  "state_block": {
    "repo_root": "/workspace",
    "candidate_files_topk": [],
    "candidate_symbols_topk": [],
    "search_patterns_used": [],
    "opened_files": [],
    "recent_read_ranges": [],
    "evidence": [],
    "last_error": null,
    "stop_ready": false
  },
  "recent_window_messages": [...],
  "target_message": {...},
  "metadata": {...}
}
```

This is the schema to freeze first.

### Decision 2: recent window variants

The initial ablation set should be:

- `K=1`
- `K=2`
- `K=3`

Where `K` is the number of recent canonical `assistant(tool_call) + tool(result)` pairs.

### Decision 3: no LM summarizer in `v0`

Use deterministic compaction only.

### Outputs

- schema spec documented
- naming convention fixed
- compact variant names fixed

Recommended names:

- `compact_k1_state_on`
- `compact_k2_state_on`
- `compact_k3_state_on`
- optional baseline: `compact_k2_state_off`

## Phase 2: Implement the compaction building blocks

This should be implemented under:

- `process_data/trajectory_coldstart/prefix_compact.py`

But implementation should be modular inside the file.

### Component A: `extract_task_block`

Input:

- raw or canonical sample

Responsibility:

- keep the first `system`
- keep the first `user`
- optionally add light issue trimming later, but not in `v0`

Output:

- `task_block`

### Component B: `canonical_recent_window`

Input:

- full prefix messages

Responsibility:

- canonicalize assistant tool calls in prefix
- collect the most recent `K` assistant/tool pairs
- preserve ordering
- discard older raw assistant/tool trace from the recent window

Output:

- `recent_window_messages`

Failure behavior:

- if recent messages cannot be canonicalized, mark sample as dropped

### Component C: `clip_tool_result`

Input:

- canonical tool family
- tool result content

Responsibility:

- clip the tool result according to tool-family-specific rules
- attach a truncation marker if text was clipped

Rules for `v0`:

- `repo_find_files` / `repo_list_dir`
  - keep first 20 paths
  - force-keep keyword-matching paths if found
- `repo_search_text`
  - keep first 10 matches
  - each match reduced to `path:line:text`
- `repo_read_range` / `repo_read_head` / `repo_read_file`
  - keep up to 80 lines
  - if longer, keep first 40 and last 20 lines from the requested span
- failed outputs
  - keep short error summary

Output:

- clipped tool message

### Component D: `build_state_block`

Input:

- older prefix history outside the recent window

Responsibility:

- extract structured facts from older history
- do not preserve raw transcripts

State fields for `v0`:

- `repo_root`
- `candidate_files_topk`
- `candidate_symbols_topk`
- `search_patterns_used`
- `opened_files`
- `recent_read_ranges`
- `evidence`
- `last_error`
- `stop_ready`

Update rules:

- from `repo_find_files` / `repo_list_dir`
  - update `candidate_files_topk`
- from `repo_search_text`
  - update `search_patterns_used`, `candidate_files_topk`, `candidate_symbols_topk`
- from `repo_read_range` / `repo_read_head` / `repo_read_file`
  - update `opened_files`, `recent_read_ranges`, `evidence`
- from failed tool results
  - update `last_error`

### Component E: `assemble_compact_sample`

Input:

- original sample
- task block
- state block
- recent window
- target message

Output:

- compact sample with `message_format = compact_v0`

## Phase 3: Generate compact pilot variants

Use as source:

- [step_level_raw_pilot.jsonl](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/data/step_level_raw_pilot.jsonl)

Generate:

- `sft/data/step_level_compact_pilot_k1_state_on.jsonl`
- `sft/data/step_level_compact_pilot_k2_state_on.jsonl`
- `sft/data/step_level_compact_pilot_k3_state_on.jsonl`
- optional:
  - `sft/data/step_level_compact_pilot_k2_state_off.jsonl`

For each variant also generate a human-readable preview:

- `sft/data/step_level_compact_pilot_k1_examples.json`
- `sft/data/step_level_compact_pilot_k2_examples.json`
- `sft/data/step_level_compact_pilot_k3_examples.json`

## Phase 4: Quantitative inspection

For each compact variant, compute:

- total kept samples
- kept rate
- kept action-step rate
- kept stop-step rate
- repo coverage
- average number of recent window messages
- average clipped tool-result size
- average number of state facts
- canonical tool distribution
- top drop reasons

This should be saved in a short report file, for example:

- `docs/plans/2026-03-20-prefix-compact-pilot-report-ai-generated.md`

## Phase 5: Qualitative inspection

For each compact variant, manually inspect at least:

- one `action_step` at step 0
- one mid-trajectory `action_step`
- one `stop_step`
- one previously dropped sample that is now kept
- one still-dropped sample

Inspection questions:

- does the recent window preserve enough local evidence?
- does the state block preserve the important old evidence?
- is the clipped tool result still decision-relevant?
- does the sample remain readable and inspectable?

This phase is mandatory.

## Phase 6: Selection criteria for `v0`

Choose the winning compact format using these criteria in order:

1. highest retained action-step count
2. acceptable stop-step retention
3. compact prefix shape is still semantically interpretable
4. clipped results still preserve decision-relevant evidence
5. no mixed tool formats

My current expectation is:

- `K=2 + state_on` is likely the best default

But this must be confirmed, not assumed.

## Phase 7: Trainer integration

Only after a compact variant is selected:

- adapt the trainer input pipeline to consume compact samples
- keep target supervision exactly on the target assistant turn
- do not change the optimization objective and compaction strategy in the same step

This separation is important for debugging.

## Phase 8: Evaluation after integration

Once the trainer consumes compact samples, evaluate in two stages.

### Offline

- next-tool-family accuracy
- argument match
- stop-vs-continue accuracy

### Closed-loop

- file recall@k
- all-correct
- average steps to first relevant file
- average total steps
- task_done timing quality

## Acceptance Criteria

The compact pipeline should be considered good enough for `v0` if:

- compact kept-rate is meaningfully above the current strict canonical rate
- sample content remains understandable to humans
- state block contains useful older-history facts
- no major loss of local decision evidence is observed in manual inspection

## Practical Recommendation

The next concrete implementation step should be:

1. create `prefix_compact.py`
2. implement `K=1/2/3` variants
3. save compact pilot files and example previews
4. compare them before touching the trainer
