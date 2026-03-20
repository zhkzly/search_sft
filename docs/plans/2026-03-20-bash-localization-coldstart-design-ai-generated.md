# Bash Localization Cold-Start Design (AI-Generated)

## 1. Goal

This project is not trying to train a general shell agent.
The immediate target is narrower:

- Input: GitHub issue / localization task / repository context
- Output: a short sequence of search-oriented tool calls that quickly narrows to the right files
- Main skill: repo localization through bash-like search actions

The current data already supports this direction:

- raw trajectories: about 740
- repos: 25
- bash calls: about 12907
- dominant commands: `grep`, `sed`, `find`, `cat`, `ls`, `head`

This means the fastest cold-start path is:

1. change the training unit from full trajectory to step-level decisions
2. reduce the action space from open-ended `bash` to a few canonical search families
3. train with native multi-turn tool-call formatting instead of XML-like text tags

## 2. Current Pipeline and Main Problems

Current pipeline:

- [convert_trajectory_legacy.py](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/process_data/trajectory_coldstart/convert_trajectory_legacy.py) converts one full trajectory into one long sample
- [sft/sft.py](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/sft.py) flattens the sample into one user turn plus appended text targets
- tool results are masked from loss, which is correct in principle

Main problems:

### 2.1 One trajectory = one sample

This wastes supervision density.
A 15-step successful localization trajectory contains about 15 tool-decision points, but the current format treats it as one long generation target.

### 2.2 Open-ended bash is too hard for small models

The model is being asked to emit arbitrary shell strings.
For 0.6B or 1.5B models, this is harder than necessary.
The dominant behaviors are already highly concentrated into a few command families, so the model should first learn those families explicitly.

### 2.3 XML-style tool-call strings are not native tool states

The model currently learns a text convention such as:

```text
<tool_call>bash</tool_call><args>{"command": "..."}</args>
```

This is easy to parse, but it is not the same as learning the actual multi-turn state transition:

`assistant(tool_calls) -> tool(result) -> assistant(next decision)`

For tool-use ability, the second form is the one that matters.

## 3. Target End State

The target cold-start system should have three layers.

### Layer A: File-level localization

Given an issue, predict:

- likely directories
- likely files
- likely keywords

This layer should be cheap and short.
It reduces search space before long reasoning starts.

### Layer B: Step-level repo search policy

Given the current prefix of the interaction, predict:

- next short thought
- next tool call

This is the core cold-start stage.

### Layer C: Stop decision

Given the current evidence, decide whether to:

- continue searching
- call `task_done`

This is necessary because high-quality trajectories are shorter than low-quality ones.

## 4. Data Reform: Full Trajectory -> Step-Level

### 4.1 New training unit

Each training sample should correspond to one assistant decision point.

Recommended sample schema:

```json
{
  "task_id": "django__django-1301",
  "quality_bucket": "3_perfect_match_100",
  "step_idx": 7,
  "prefix_messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "...task..."},
    {"role": "assistant", "content": "short reasoning", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_1", "content": "...tool result..."},
    {"role": "assistant", "content": "next reasoning", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_2", "content": "...tool result..."}
  ],
  "target_message": {
    "role": "assistant",
    "content": "the next short reasoning",
    "tool_calls": [
      {
        "id": "call_3",
        "type": "function",
        "function": {
          "name": "repo_search_text",
          "arguments": "{\"pattern\":\"csrf\",\"path\":\"/workspace/django\"}"
        }
      }
    ]
  },
  "metadata": {
    "repo": "django__django",
    "raw_tool_name": "bash",
    "raw_command": "grep -n \"csrf\" /workspace/django/middleware/csrf.py"
  }
}
```

### 4.2 Conversion rule

For each interaction `i` in a trajectory:

1. collect all previous system, user, assistant, and tool messages into `prefix_messages`
2. take the current assistant response as `target_message`
3. create one training sample

This turns one long trajectory into many supervised next-step decisions.

### 4.3 Which trajectories to use first

Recommended cold-start training set:

- first choice: all `3_perfect_match_100`
- second choice: all `2_medium_f1_50-100`
- do not use `0_no_match` and `1_low_f1_0-50` in the first SFT stage

Reason:

- perfect and medium already give about 4702 bash decision points
- low-quality trajectories are better used as negative or preference data later

### 4.4 Length policy

Recommended order:

1. `short_lt100k`
2. `medium_100k_300k`
3. `long_gte300k`

Do not start from long trajectories.
The model first needs to learn the policy shape, not extreme context handling.

## 5. Action Space Compression

### 5.1 Do not cold-start with raw unrestricted bash

Recommended first-stage action families:

- `repo_find_files`
- `repo_list_dir`
- `repo_search_text`
- `repo_read_head`
- `repo_read_range`
- `repo_read_file`
- `task_done`

These are enough to cover most of the current data distribution.

### 5.2 Canonical mapping from raw bash

Map the dominant raw commands into canonical tool families:

- `find /workspace -type f -name "*.py"` -> `repo_find_files`
- `ls -la /workspace/foo` -> `repo_list_dir`
- `grep -n "pattern" path` or `grep -r` -> `repo_search_text`
- `head -50 file` -> `repo_read_head`
- `sed -n '20,80p' file` -> `repo_read_range`
- `cat file` -> `repo_read_file`

Discard or postpone uncommon actions in stage 1:

- `pip install`
- `apt-get`
- `curl`
- arbitrary `python -c`
- environment mutation commands

For localization, these are either irrelevant or too noisy.

### 5.3 Why structured proxy tools are better

Even if the final runtime only exposes one `bash` tool, cold-start should not start there.

Recommended strategy:

- training stage 1: structured proxy tools
- training stage 2: distill back into raw bash strings if needed

This gives the model a smaller decision space first.
Later, if the deployment environment insists on a single `bash` tool, a thin transpiler can map the structured tool call back into canonical bash.

Example:

```json
{
  "name": "repo_read_range",
  "arguments": {
    "file_path": "/workspace/django/middleware/csrf.py",
    "start_line": 1,
    "end_line": 120
  }
}
```

can be compiled to:

```bash
sed -n '1,120p' /workspace/django/middleware/csrf.py
```

## 6. Native Multi-Turn Tool-Call Format

### 6.1 Recommended message format

Use OpenAI-compatible message objects:

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {
    "role": "assistant",
    "content": "I should inspect the CSRF middleware.",
    "tool_calls": [
      {
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "repo_search_text",
          "arguments": "{\"pattern\":\"csrf\",\"path\":\"/workspace/django\"}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_1",
    "content": "...search result..."
  }
]
```

### 6.2 Loss masking rule

Recommended masking:

- prefix messages: no loss
- target assistant message content: loss
- target assistant tool call JSON: loss
- tool result messages: no loss

This preserves the useful part of the current masking idea, but in a native multi-turn setting.

### 6.3 Training-time rendering

For each step-level sample:

1. render `prefix_messages` with `add_generation_prompt=True`
2. render `prefix_messages + [target_message]`
3. mask all tokens from the prefix render
4. unmask only the assistant target continuation

Implementation note:

- if the tokenizer supports assistant masks directly, use that
- otherwise compute masks from the template-aware rendered sequence

The important point is to let the model predict the next assistant turn, not a flattened pseudo-format.

## 7. Two-Stage Cold-Start Curriculum

### Stage 1: File localization warmup

Task:

- input: issue text
- output: top-k file paths or top-k directories plus keywords

Purpose:

- reduce repository search entropy
- give the model a rough map before multi-step tool use

Data source:

- final `task_done` outputs from successful trajectories
- files mentioned in final localization block

Recommended target format:

```text
repo: django__django
keywords: csrf, token, middleware
files:
- /workspace/django/middleware/csrf.py
- /workspace/django/template/defaulttags.py
- /workspace/tests/csrf_tests/tests.py
```

### Stage 2: Step-level search policy SFT

Task:

- input: prefix messages
- output: next assistant thought plus next tool call

Purpose:

- learn when to search
- learn which search family to use
- learn short-horizon repo exploration

Recommended thought style:

- short
- local
- action-oriented

Do not train long free-form chain-of-thought.
For small models, long thoughts raise sequence length and reduce decision clarity.

### Stage 3: Stop decision and task completion

Task:

- input: current evidence state
- output: either another search action or `task_done`

Purpose:

- suppress over-searching
- match the behavior of high-quality short successful trajectories

## 8. Preference Stage After SFT

After the first SFT converges, use the existing bucket labels as preference signals.

Recommended pairs:

- `perfect` > `medium`
- `perfect` > `low`
- `medium` > `low`
- shorter successful trajectory > longer successful trajectory for the same issue

Preference objective can be:

- DPO
- multi-turn DPO
- other pairwise preference optimization

Primary behaviors to reward:

- fewer redundant searches
- faster convergence to relevant files
- correct use of `task_done`
- lower rate of irrelevant file reads

## 9. Evaluation Plan

Do not evaluate only with final F1.

Use at least these metrics:

- `file_recall@k`
- `file_precision@k`
- `all_correct`
- average bash steps
- average steps before first relevant file hit
- `task_done` overrun rate
- percentage of commands belonging to canonical high-value families
- invalid tool-call rate

Recommended evaluation splits:

- in-project split across existing trajectory repos
- held-out repo split
- optional external benchmark: LCA bug localization

Be careful with contamination if using SWE-bench style data.

## 10. Proposed Implementation Tasks

Recommended minimal code changes:

### Task A: New step-level converter

Add a new script, for example:

- `process_data/trajectory_stepify.py`

Responsibilities:

- read raw `trajectory.json`
- emit step-level multi-turn samples
- preserve metadata such as repo, quality bucket, step index, raw command

### Task B: Canonical action mapper

Add:

- `process_data/bash_canonicalize.py`

Responsibilities:

- map raw `bash` commands to one of the canonical action families
- keep both canonical action and raw command
- filter unsupported or low-value commands from stage 1

### Task C: New SFT preprocessor

Replace or extend the current flat processing logic in:

- [sft/sft.py](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/sft.py)

Responsibilities:

- consume `prefix_messages` and `target_message`
- render native chat template
- mask only assistant target tokens

### Task D: Evaluation script

Add:

- `eval/eval_localization_policy.py`

Responsibilities:

- replay predicted actions
- compute step efficiency and file localization metrics

## 11. Concrete Execution Order

Recommended order of work:

1. build the step-level converter
2. build the command-family canonicalizer
3. export `perfect + medium` short and medium-length step data
4. train stage-2 search policy SFT on canonical actions
5. add the stop-decision subset
6. run held-out evaluation
7. only then consider raw `bash` distillation or preference optimization

## 12. Detailed Execution Procedure

Yes, the correct order is:

1. data processing
2. dataset audit
3. training pipeline changes
4. smoke training
5. evaluation
6. only then scale up

Do not start by modifying the training loop first.
If the data schema is not stable, the training code will be revised repeatedly.

### Step 0: Freeze the target task definition

Before any processing, define the exact first-stage task:

- task type: repository localization
- allowed action families: only the canonical search actions
- training objective: predict the next assistant turn
- success criterion: reach relevant files faster with fewer steps

Output of this step:

- a fixed list of allowed tool names
- a fixed sample schema
- a fixed evaluation split policy

This design document now plays that role.

### Step 1: Build the step-level raw dataset

Input:

- raw `trajectory.json` files under `classified_by_f1`

Filter:

- only `2_medium_f1_50-100`
- only `3_perfect_match_100`
- first pass: only `short_lt100k` and `medium_100k_300k`

Operation:

- iterate over each `llm_interaction`
- collect all prior messages as `prefix_messages`
- convert the current assistant response into `target_message`
- keep original raw tool call and tool result in metadata

Recommended saved file:

- `sft/data/step_level_raw_medium_perfect.jsonl`

Recommended record fields per line:

```json
{
  "task_id": "django__django-1301",
  "repo": "django__django",
  "quality_bucket": "3_perfect_match_100",
  "length_bucket": "short_lt100k",
  "step_idx": 5,
  "prefix_messages": [...],
  "target_message": {...},
  "metadata": {
    "raw_tool_name": "bash",
    "raw_command": "grep -n \"csrf\" /workspace/django/middleware/csrf.py",
    "is_terminal_step": false
  }
}
```

Checks after Step 1:

- number of tasks
- number of step samples
- number of `bash` targets
- number of `task_done` targets
- average prefix length
- average target length

This is the first audit point.
Do not move on until these statistics are printed and inspected.

### Step 2: Canonicalize the action space

Input:

- `step_level_raw_medium_perfect.jsonl`

Operation:

- parse `metadata.raw_command`
- map raw `bash` into a canonical tool family
- keep unsupported commands marked as `unmapped`
- optionally drop `unmapped` from version `v0`

Recommended saved file:

- `sft/data/step_level_canonical_v0.jsonl`

Recommended added fields:

```json
{
  "target_message": {
    "role": "assistant",
    "content": "I should inspect the CSRF-related files first.",
    "tool_calls": [
      {
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "repo_search_text",
          "arguments": "{\"pattern\":\"csrf\",\"path\":\"/workspace/django\"}"
        }
      }
    ]
  },
  "metadata": {
    "raw_command": "grep -n \"csrf\" /workspace/django/middleware/csrf.py",
    "canonical_tool_name": "repo_search_text",
    "canonical_tool_args": {
      "pattern": "csrf",
      "path": "/workspace/django/middleware/csrf.py"
    },
    "mapping_status": "mapped"
  }
}
```

Checks after Step 2:

- mapping coverage
- tool-family distribution
- examples of each mapped family
- examples of `unmapped`

Target for `v0`:

- high coverage on `find/grep/sed/cat/ls/head`
- low-value commands excluded cleanly

### Step 3: Create train/dev/test splits

This split should be repository-aware.
Do not randomly split steps from the same repo into train and test.

Recommended split policy:

- train: most repositories
- dev: held-out tasks from a few repositories
- test: fully held-out repositories if possible

Minimum saved artifacts:

- `sft/data/step_level_canonical_v0_train.jsonl`
- `sft/data/step_level_canonical_v0_dev.jsonl`
- `sft/data/step_level_canonical_v0_test.jsonl`

Checks after Step 3:

- no repo leakage across train and test
- similar distribution of canonical tools
- similar step length statistics

### Step 4: Add a dataset inspector

Before touching training code, add a lightweight inspector script.

Responsibilities:

- load one split
- print counts and distributions
- render a few samples exactly as the trainer will see them

The purpose is simple:

- verify the multi-turn message schema is correct
- verify tool call JSON is valid
- verify target masking boundaries are easy to compute

Without this step, trainer bugs become hard to diagnose.

### Step 5: Modify the training pipeline

Only now should the training code be changed.

Input:

- canonical multi-turn step data

Operation:

- replace the old flat `input` + `output` logic
- render `prefix_messages` with `apply_chat_template(..., add_generation_prompt=True)`
- render `prefix_messages + [target_message]`
- compute labels so only the target assistant continuation contributes loss

Required outputs from this stage:

- one debug print of rendered prefix
- one debug print of rendered full target sequence
- one debug print of decoded supervised target tokens only

Checks after Step 5:

- assistant target is the only supervised span
- tool result content is masked
- no duplicated messages
- tool call JSON appears in the expected template format

### Step 6: Smoke training

Before full training, run a very small training job.

Recommended smoke setup:

- 100 to 500 samples
- 1 epoch
- small max length
- frequent logging

What to inspect:

- loss goes down
- no template corruption
- generated samples contain valid canonical tool calls
- model does not emit old XML tags

Only after this passes should large-scale training begin.

### Step 7: Offline evaluation

Run the model on the test split without tool execution first.

Measure:

- next-tool accuracy
- next-tool-family accuracy
- argument exact match or soft match
- stop-vs-continue accuracy

This stage answers:

- can the model choose the right action class?
- can it fill arguments correctly?

### Step 8: Closed-loop evaluation

After offline evaluation, run a small closed-loop replay.

Two options:

- exact replay against recorded tool results
- execution against a real or simulated repo environment

Metrics:

- file hit rate
- average steps to first relevant file
- average total steps
- terminal success rate
- `task_done` timing quality

This is the main measure of whether the policy is actually useful.

### Step 9: Expand to raw bash or preference training

Only after the canonical-action model is stable:

- distill canonical actions back to raw bash if deployment needs it
- add `low/no_match` for preference training
- extend to long trajectories

## 13. Immediate Next Version

The first practical version should be intentionally narrow.

Version `v0` should:

- use only `perfect + medium`
- use only canonical search families
- ignore uncommon shell commands
- use short assistant thoughts
- use step-level next-action prediction
- evaluate on file hit rate and average search length

If `v0` works, then `v1` can add:

- raw bash reconstruction
- longer contexts
- preference optimization with low-quality trajectories
- external benchmark transfer

## 14. Decision Summary

The recommended cold-start recipe is:

1. train on step-level decisions, not whole trajectories
2. reduce raw `bash` into a small set of canonical search tools
3. represent data as native multi-turn tool-calling conversations
4. use `perfect` and `medium` as SFT data
5. use `low` and `no_match` later as negative or preference data

This should improve learnability more than simply adding more raw trajectory text.

## 15. Revised Recommended Plan

This section is the final recommended execution plan after challenging the initial assumptions.

### Priority Clarification

The primary goal of this project is:

- give a small model a real tool-use prior for repository localization

The primary goal is not:

- rely only on context engineering
- solve localization only by retrieving better context for a tool-naive model

This distinction matters.

If the goal is tool-use cold start, then the model should learn:

- when to call a tool
- which tool family to call
- how to fill arguments
- when to stop and emit `task_done`

Context engineering is still useful at inference time, but it is secondary.
It should support the trained policy, not replace it.

### Plan Overview

The correct execution order is:

1. freeze evaluation protocol first
2. run a small pilot data pipeline
3. inspect and refine the sample schema
4. process the full dataset
5. adapt the training pipeline
6. run smoke training
7. run offline evaluation
8. run closed-loop evaluation
9. only then expand scope

### Phase 1: Freeze evaluation before data processing

First define:

- train/dev/test repository split
- core metrics
- what counts as a successful localization step

Why:

- if evaluation is defined after data processing, it is easy to leak repository information into training
- if metrics are vague, data design will drift toward convenience instead of utility

Required outputs:

- a repo-heldout split file
- a metric definition file or a short markdown spec

### Phase 2: Build a pilot dataset, not the full dataset

Start with a small subset, for example:

- 50 to 100 trajectories
- only `perfect` and `medium`
- only `short` and `medium`

Why:

- the sample schema will likely need revision
- canonicalization rules will almost certainly need debugging
- target masking bugs are easier to catch on a small set

Pilot output files:

- `step_level_raw_pilot.jsonl`
- `step_level_canonical_pilot.jsonl`

Pilot audit:

- step count
- tool-family count
- unmapped command count
- average prefix length
- average target length
- examples rendered exactly as training inputs

### Phase 3: Refine the sample schema

Do not assume the initial schema is optimal.

Three schema variants should be compared:

- `action-only`
- `short-thought + action`
- `full-thought + action`

The recommended starting point is:

- `v0 = short-thought + action`

where `short-thought` means one short local planning sentence only.

Why not default to long reasoning:

- small models overfit style faster than policy
- long thoughts increase sequence length and reduce supervision density
- the goal is not beautiful explanation, it is correct next action

### Phase 4: Full data processing

After the pilot format is stable:

- process all `perfect` and `medium`
- still keep only `short` and `medium` first
- keep repo-level split boundaries fixed

Output files:

- `step_level_raw_full.jsonl`
- `step_level_canonical_full.jsonl`
- train/dev/test JSONL files

### Phase 5: Training pipeline changes

Only after the full step-level canonical dataset is stable:

- modify the trainer
- render native multi-turn messages
- supervise only the target assistant turn

Why training changes come later:

- if data schema changes first, trainer changes made early will be invalidated
- trainer debugging is much easier when the dataset has already been audited

### Phase 6: Smoke training

Run a very small training job first.

Success criteria:

- no format corruption
- valid canonical tool calls are generated
- no old XML-style tool tags appear
- loss decreases normally

### Phase 7: Offline evaluation

Before tool execution, evaluate the next-step prediction quality.

Metrics:

- next tool-family accuracy
- next tool-argument match
- stop-vs-continue accuracy

Why:

- this tells whether the model learned the local search policy at all
- it isolates policy learning from environment execution noise

### Phase 8: Closed-loop evaluation

Only after offline metrics are acceptable:

- replay against recorded tool results
- or execute in a controlled repository environment

Metrics:

- file recall@k
- all-correct
- average steps to first relevant file
- average total steps
- task_done timing quality

### Phase 9: Expand the scope

Only after the canonical-action policy is stable:

- add longer trajectories
- add raw bash distillation
- add preference training using low-quality trajectories

## 17. Prefix Compact Detailed Design

This section refines the earlier `prefix_compact_v0` idea.
It is written as a design decision record, not as final code.

### 17.1 What prefix compact is actually solving

The point of `prefix_compact` is not generic long-context management.
It is a task-specific transformation for tool-use supervision.

The actual problem observed in the pilot is:

- current-step target commands are mostly parseable
- strict sample retention collapses mainly because older prefix history is too noisy or not canonicalizable

So `prefix_compact` should be understood as:

- keep the parts of history that are locally useful for next-step tool choice
- compress older history into a stable state representation
- remove historical detail that the model does not need to imitate token-by-token

### 17.2 Why not just keep the full prefix

Keeping the full prefix is attractive in theory but weak in this dataset for three reasons:

1. older history often contains low-value shell detail
2. older tool results are extremely long and noisy
3. older assistant actions frequently use command forms outside the current canonicalizer coverage

So full-prefix training currently optimizes for transcript fidelity more than tool-policy clarity.

### 17.3 Why `K=2` was proposed, and why it should not be treated as fixed truth

`K` means the number of recent `assistant(tool_call) + tool(result)` pairs kept verbatim.

`K=2` is a default, not a theorem.

It was proposed because pilot retention experiments showed:

- full prefix: about 45%
- recent 1 pair: about 77%
- recent 2 pairs: about 70%
- recent 3 pairs: about 66%

So the local data already shows that smaller windows dramatically improve sample retention.

However, `K=1` may be too aggressive because:

- some steps depend on a short chain of “search -> inspect -> refine search”
- the immediately previous observation may not be enough to explain the current action

And `K>=3` may be too expensive because:

- much of the third and older pair content can be moved into state
- extra recent history quickly reintroduces noisy prefix actions

Recommended interpretation:

- `K=2` is the best default compromise
- `K=1` should be run as a retention-heavy ablation
- `K=3` should be run as a fidelity-heavy ablation

Do not hard-code `K=2` as ideology.
Treat it as the default point in a small ablation grid.

### 17.4 Why deterministic compaction first, not LM summarization first

The reason is not that LM summarization is useless.
The reason is that the project is still deciding the tool-learning interface.

If summarization is introduced too early:

- it adds a second model and a second source of noise
- it becomes harder to tell whether failures come from summarization or policy learning
- it weakens inspectability of the training data

Deterministic compaction is preferable for `v0` because:

- every transformation is inspectable
- it can be reproduced exactly at training and evaluation time
- errors can be traced back to a rule instead of a hidden summarizer behavior

LM summarization can be introduced later when:

- the canonical tool interface is already stable
- the state schema is already validated
- deterministic compaction is shown to lose important information

### 17.5 Why tool-result clipping is necessary

Tool results in this dataset are often:

- long
- noisy
- partially malformed
- mixed with terminal echo
- much more detailed than the model needs for the next local action

So clipping is not optional.
Without clipping, recent-window retention alone still feeds large amounts of low-value text.

But clipping should not mean “always keep only the beginning”.
That would be too naive.

### 17.6 What clipping should look like

Clipping should be tool-family aware.

#### `repo_find_files` and `repo_list_dir`

Do not keep arbitrary head-only text by default.

Recommended output shape:

- first `N_head` lines
- plus any lines matching the current issue keywords
- plus a truncation marker if omitted

Why:

- path discovery is mostly about candidate path coverage
- the most important signal is “which files/dirs appeared”, not every line

Recommended `v0` rule:

- keep up to 20 paths
- if keyword hits exist, force-keep them

#### `repo_search_text`

Do not keep arbitrary head-only text.
For grep-like outputs, the important unit is “match”, not line count at the top.

Recommended output shape:

- first `M` matches
- each match stored as `path:line:text`
- force-keep matches containing symbol-like patterns such as `def`, `class`, `function`
- add truncation marker

Why:

- the next action often depends on the matched file and line, not on all matches

Recommended `v0` rule:

- keep first 10 matches
- truncate text payload per match to a fixed width

#### `repo_read_range`, `repo_read_file`, `repo_read_head`

For code reading, “head + tail” is not always the right default.

Challenge:

- `head + tail` is reasonable for logs
- but for code, the useful part is usually the requested span itself

So the default should be:

- keep the requested code span
- if too long, keep the exact requested start area and the exact requested end area
- preserve line numbers

Recommended `v0` rule:

- keep up to 80 lines
- if longer: keep first 40 and last 20 from the span, with a marker

#### Failed tool results

These should not be dropped.

Keep:

- command family
- short error text
- failure marker

This is required for recovery behavior.

### 17.7 What the State Block should be

The `State Block` should not be a free-form natural language summary.
It should be a structured, task-specific memory object.

The purpose is:

- represent older history in a stable schema
- carry forward only decision-relevant facts
- avoid replaying raw tool outputs

Recommended `v0` state schema:

```json
{
  "repo_root": "/workspace",
  "candidate_files_topk": [],
  "candidate_symbols_topk": [],
  "search_patterns_used": [],
  "opened_files": [],
  "recent_read_ranges": [],
  "evidence": [],
  "last_error": null,
  "stop_ready": false
}
```

### 17.8 What each State Block field means

#### `repo_root`

Usually fixed as `/workspace`.
Included mainly for explicitness and portability.

#### `candidate_files_topk`

Paths currently believed to be most relevant.

How to populate:

- count repeated hits from `find`, `grep`, `ls`
- prefer files observed in multiple steps
- keep top-K unique paths

This is one of the most important fields.

#### `candidate_symbols_topk`

Functions, classes, constants, or methods that appear relevant.

How to populate:

- extract from grep/read outputs if patterns like `def`, `class`, method names, or explicit localization strings appear

This is useful because many final `task_done` outputs are symbol-oriented, not file-only.

#### `search_patterns_used`

The grep/find patterns already tried.

Why keep it:

- avoids repeated low-value searches
- helps the model know the current search frontier

#### `opened_files`

Files already explicitly opened or read.

Why keep it:

- helps the model avoid reopening the same file without a reason
- marks which candidates have already been inspected

#### `recent_read_ranges`

Pairs such as:

- file path
- start line
- end line

Why keep it:

- code-localization often depends on whether a file was only skimmed or actually inspected near the relevant lines

#### `evidence`

A short list of structured evidence items.

Each item should be extremely short, for example:

- `localstack/utils/common.py contains generate_ssl_cert and hardcoded CN`
- `transformers/tokenization_utils_base.py line 2227 mentions max_length warning`

These are not chain-of-thought.
They are compressed factual justifications.

#### `last_error`

A short structure such as:

```json
{
  "tool_name": "repo_search_text",
  "args": {"pattern": "...", "path": "..."},
  "error_summary": "...",
  "step_idx": 5
}
```

Why keep it:

- recovery is part of tool-use ability
- failed attempts should shape the next action

#### `stop_ready`

A coarse boolean indicating whether the current evidence looks sufficient for localization.

For `v0`, this can be heuristically computed.
Later, it can become a learned supervision target.

### 17.9 What should not go into the State Block

The State Block should not contain:

- full raw tool results
- entire code snippets unless tiny
- repeated natural-language assistant thoughts
- low-level shell syntax details already represented by canonical tools

It should represent facts, not transcripts.

### 17.10 Should old history become summary text or state

For `v0`, older history should become `State Block`, not free-form summary text.

Reason:

- state is easier to inspect
- state is schema-stable
- state is easier to compute deterministically
- state is easier to learn from consistently

Free-form summary text can be added later as an auxiliary field if needed.

### 17.11 Should tool names be kept in state

Yes, but indirectly and selectively.

Do not keep a raw transcript of all tool names.
Instead, preserve the effect of the tools through fields like:

- `search_patterns_used`
- `opened_files`
- `recent_read_ranges`
- `last_error`

If needed, add a compact field:

- `recent_tool_families`: last 2-3 canonical tool names

But only if ablations show it helps.
This should not be the default first choice because the recent window already exposes recent tool names explicitly.

### 17.12 Should tool results be kept in state

Not as raw text.

Only the extracted facts should be carried into state.

Examples:

- from `find` result: keep candidate file paths
- from `grep` result: keep matching paths, lines, symbols
- from `sed` result: keep opened file span and short evidence

So the answer is:

- raw tool result belongs in the recent window, clipped
- distilled facts from old tool results belong in the state

### 17.13 Recommended `v0` compaction template

The final compact prefix should be:

1. `system`
2. `user`
3. `state_block_json`
4. recent canonical assistant/tool pair 1
5. recent canonical assistant/tool pair 2

If fewer than 2 recent pairs exist, use however many exist.

### 17.14 Recommended `v0` ablation plan

Do not trust one design point.
At minimum compare:

- `K=1`, state on
- `K=2`, state on
- `K=3`, state on

And if resources allow:

- `K=2`, state off

This tells whether the state actually replaces older history effectively.

### 17.15 Final recommendation

The current best `v0` is:

- deterministic compaction
- `K=2` recent canonical pairs
- tool-family-aware clipping
- older history rewritten into structured state
- explicit `last_error`

But this should be treated as the default experiment, not as unquestionable truth.

## 16. What This Plan Challenges

This plan intentionally rejects several tempting but weak assumptions.

### Assumption 1: every interaction is a good supervised sample

This is false.
Some interactions are redundant, noisy, or already off-policy.

Consequence:

- every interaction should first be treated as a candidate sample, not an automatic positive sample

### Assumption 2: more prefix history is always better

This is false for small models.
Long noisy prefixes reduce effective capacity and can drown out the local decision state.

Consequence:

- keep raw full history in data storage
- but compare training with compact context variants

### Assumption 3: thought plus tool call is always better than action-only

This is not guaranteed.
For small models, the extra thought tokens may hurt more than help.

Consequence:

- compare `action-only` and `short-thought + action` in the pilot

### Assumption 4: trajectory-level quality means step-level quality

This is false.
A perfect trajectory may still contain wasteful intermediate steps.
A medium trajectory may contain many excellent steps and only fail near the end.

Consequence:

- use trajectory bucket as filtering and weighting, not as perfect step-level truth

### Assumption 5: the trainer should be changed first

This is a bad engineering order.
If the data schema is unstable, trainer work will be redone.

Consequence:

- freeze data format first
- change the trainer second

### Assumption 6: evaluation can be defined at the end

This is risky.
Without early split and metric freezing, leakage and hindsight optimization are very likely.

Consequence:

- define held-out repo splits and metrics before full data generation
