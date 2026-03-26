# Tool-Use Evaluation Plan (AI-Generated)

## Goal

Evaluate whether the trained model has actually learned to use tools correctly for repository localization.

The core question is not:

- "does the final answer sometimes look right?"

The core question is:

- "does the model choose the right next tool?"
- "does it format the tool call correctly?"
- "does it provide usable arguments?"
- "does the tool execution produce useful evidence?"
- "does this improve final localization?"

## Why this evaluation is necessary

For this project, final localization quality alone is not enough.

A model can look superficially good while still failing tool use in one of these ways:

- emits invalid tool format
- selects the wrong tool family
- picks bad arguments
- stops too early or too late
- repeatedly searches without progress

Therefore evaluation must be layered.

## Evaluation Stack

Evaluation should be staged.

For the current current experiment stage, only the first two layers are mandatory:

1. `Tool Call Correctness`
2. `Tool Execution Quality`

`Localization Outcome` is a later, optional downstream layer that only becomes necessary once the model can be put into a stable closed-loop tool-using evaluation.

This order matters.
Do not start from final recall only.

## Layer 1: Tool Call Correctness

This layer evaluates the next predicted step against the target step.

### What to measure

- `format_validity`
  - whether the predicted tool call can be parsed
- `schema_validity`
  - whether required fields exist and have the correct types
- `tool_family_accuracy`
  - whether the predicted tool family matches the gold step
- `tool_args_exact_match`
  - whether the arguments exactly match the gold arguments
- `tool_args_soft_match`
  - whether the arguments are semantically close enough to be useful
- `stop_step_accuracy`
  - whether `task_done` is predicted when appropriate

### Why this layer comes first

If the model cannot even produce valid or semantically correct tool calls, then any downstream localization success is unstable.

## Layer 2: Tool Execution Quality

This layer checks whether the predicted tool call, once executed, produces useful information.

### What to measure

- `execution_success_rate`
  - whether the predicted tool call executes without error
- `useful_observation_rate`
  - whether the returned observation contains evidence useful for the task
- `first_hit_step`
  - the first step where a relevant file is hit
- `redundant_search_ratio`
  - how often the model repeats low-value searches

### Why this layer matters

Even if tool family and argument format are technically correct, the step may still be unhelpful.

For tool-use learning, "executable" is weaker than "useful".

## Layer 3: Localization Outcome (Optional Later Stage)

This layer evaluates the final localization result.

For the current current experiment stage, this layer is not the main gate.
It should only be activated after the model already shows strong next-step tool behavior.

### What to measure

- `file_precision`
- `file_recall`
- `num_matched`
- `all_correct`
- `num_pred_files`
- `avg_total_steps`
- `task_done_timing_quality`

### Existing supervision already available

The trajectory files already contain:

- `formatted_localization`
- `formatted_localization_readable`
- `analysis.metrics`

These should be reused as the reference signal for localization outcome evaluation.

## Evaluation Datasets

Evaluation should be run on initial experiment first.

### Recommended first evaluation data

- compact dataset `K=2`
- optionally compare with `K=1` and `K=3`

Candidate files:

- [step_level_compact_k1_state_on.jsonl](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/data/step_level_compact_k1_state_on.jsonl)
- [step_level_compact_k2_state_on.jsonl](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/data/step_level_compact_k2_state_on.jsonl)
- [step_level_compact_k3_state_on.jsonl](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/data/step_level_compact_k3_state_on.jsonl)

### Recommended split

Create a repository-aware held-out split with repository awareness.

Do not randomly mix steps from the same repository into train and eval.

Suggested split:

- train: majority of repositories in the current experiment set
- eval: held-out repositories inside the current experiment split

## Comparison Baselines

At minimum compare these systems:

1. `base model`
2. `legacy trajectory SFT`
3. `compact_k1`
4. `compact_k2`
5. optional `compact_k3`

This is necessary to answer two questions:

- did compact training improve tool use over the base model?
- did compact training improve over legacy trajectory SFT?

If legacy trajectory SFT is not immediately available, it can be deferred.
The minimum viable comparison for the current stage is:

1. `base model`
2. `compact_k2`
3. optional `compact_k1`

## Detailed Metric Definitions

## `format_validity`

A predicted step is format-valid if:

- assistant output can be parsed
- `tool_calls` exists when expected
- tool call JSON is syntactically valid

## `schema_validity`

A predicted step is schema-valid if:

- tool name is in the allowed canonical set
- required args exist
- arg types are correct

Examples:

- `repo_read_range.start_line` must be integer
- `repo_search_text.path` must be string
- `task_done` should not contain unrelated arguments

## `tool_family_accuracy`

Compare the predicted tool family to the gold tool family.

Examples:

- `repo_search_text`
- `repo_find_files`
- `repo_read_range`
- `repo_list_dir`
- `repo_read_file`
- `repo_read_head`
- `task_done`

This is the most important next-step metric.

## `tool_args_exact_match`

Exact string or exact JSON match on arguments.

This metric is strict and should not be used alone.

## `tool_args_soft_match`

A relaxed semantic match.

Examples:

- same file path but slightly different line range
- same search intent with equivalent pattern wording
- same directory but different result limit

This metric better reflects tool usefulness.

## `execution_success_rate`

Whether the predicted tool call executes successfully.

This should be computed only after a step passes schema validation.

## `useful_observation_rate`

A tool call is useful if the resulting observation:

- contains a ground-truth file
- contains a relevant symbol
- or clearly advances the search frontier

This should be heuristically defined first, then refined later.

## `task_done_timing_quality`

This is not just whether the model predicts `task_done`.

It must also answer:

- did the model stop too early?
- did the model over-search before stopping?

## Step-Level Evaluation Record

Each evaluated step should be saved in a record like:

```json
{
  "sample_id": "...",
  "repo": "...",
  "target_kind": "action_step",
  "gold_tool_name": "repo_search_text",
  "gold_tool_args": {...},
  "pred_tool_name": "repo_search_text",
  "pred_tool_args": {...},
  "format_validity": true,
  "schema_validity": true,
  "tool_family_correct": true,
  "tool_args_exact_match": false,
  "tool_args_soft_match": true,
  "execution_success": true,
  "observation_useful": true
}
```

This record should be saved per step for later analysis.

## Stage-Specific Execution Plan

This section defines what should be done now, in order.

### Stage 0: Freeze the repository-aware eval split

Before running any evaluation, create a repository-aware held-out split on the compact dataset set.

Required properties:

- no repository leakage between train and eval
- same split reused across all compared models

Suggested artifacts:

- `sft/data/step_level_compact_k2_state_on_train.jsonl`
- `sft/data/step_level_compact_k2_state_on_eval.jsonl`
- optional:
  - `sft/data/step_level_compact_k1_state_on_eval.jsonl`

### Stage 1: Evaluate the base model on the held-out split

Do not wait until after training.
Run the base model first to establish the floor.

Outputs:

- per-step prediction records
- summary metrics

Focus only on:

- `format_validity`
- `schema_validity`
- `tool_family_accuracy`
- `tool_args_soft_match`
- `stop_step_accuracy`

### Stage 2: Train the compact model

Use:

- compact training data
- compact eval split fixed in stage 0

Current recommended default:

- `compact_k2`

This training run only needs to be a initial experiment run.
It does not need to be long or expensive.

### Stage 3: Re-run the same step-level eval on the trained compact model

Compare directly against the base model.

Primary questions:

- did tool-family accuracy improve?
- did schema validity remain high?
- did stop-step accuracy improve or at least not collapse?
- did tool-argument soft match improve?

### Stage 4: Add one-step execution evaluation

Once step-level predictions are available:

1. execute the predicted canonical tool call
2. judge whether it succeeds
3. judge whether the returned observation is useful

This can still be done step-by-step and does not yet require a full closed-loop rollout.

Metrics to compute:

- `execution_success_rate`
- `useful_observation_rate`

### Stage 5: Decide whether compact training is effective

At this stage, the current experiment should be considered effective if:

- `tool_family_accuracy` improves over base
- `schema_validity` stays high
- `tool_args_soft_match` improves
- `execution_success_rate` is acceptable
- `useful_observation_rate` improves

If these conditions are not met, do not move to closed-loop localization outcome evaluation yet.

### Stage 6: Optional later-stage closed-loop evaluation

Only after stages 1-5 are satisfactory:

- run multi-step tool use
- measure final localization outcome
- measure `task_done` timing
- measure step efficiency

This stage is useful, but it is not the first success criterion for the current experiment.

## Immediate Detailed To-Do

The next concrete implementation order should be:

1. create a repo-aware eval split for `compact_k2`
2. implement `eval/tool_use_step_eval.py`
3. run the base model on the eval split
4. train the compact dataset model
5. run the trained model on the same eval split
6. compare step-level tool-use metrics
7. if promising, implement one-step execution evaluation

## Immediate Success Criteria

For the current current experiment stage, the minimum convincing result is:

- higher `tool_family_accuracy`
- non-trivial `tool_args_soft_match`
- high `schema_validity`
- acceptable `stop_step_accuracy`

Notably, final file recall is not required yet to declare the experiment useful.

## Recommended Evaluation Pipeline

### Phase 1: Offline next-step evaluation

Input:

- held-out compact step samples

Procedure:

1. feed compact prefix to model
2. decode next assistant step
3. parse predicted tool call
4. compare with gold target step

Outputs:

- per-step JSONL results
- summary metrics

### Phase 2: One-step execution evaluation

Input:

- predicted next tool calls from phase 1

Procedure:

1. execute predicted tool via canonical tool adapter
2. inspect whether the result is useful

Outputs:

- execution success rate
- useful observation rate

### Phase 3: Closed-loop replay evaluation

Input:

- experiment repositories
- trained model

Procedure:

1. start from a task
2. repeatedly predict next step
3. execute canonical tool
4. feed result back
5. stop at `task_done` or max steps

Outputs:

- first hit step
- final file recall
- average steps
- stop timing quality

## What counts as "effective"

The training should be considered effective only if:

- tool family accuracy improves over the base model
- schema validity remains high
- stop-step accuracy does not collapse
- final localization recall improves or at least does not regress
- average number of steps does not increase substantially

This is a joint condition.

Low loss alone is not enough.

## Recommended First Implementation

The next concrete evaluation implementation should be:

1. `eval/tool_use_step_eval.py`
   - next-step tool correctness
2. a small report script that compares:
   - base model
   - compact_k1
   - compact_k2

`eval/tool_use_replay_eval.py` should be treated as a later-stage extension, not the first required step.

## Practical Default

For the first real validation run, use:

- training data: `compact_k2`
- evaluation mode: offline next-step first
- metrics to look at immediately:
  - `tool_family_accuracy`
  - `schema_validity`
  - `tool_args_soft_match`
  - `stop_step_accuracy`

Only after these are acceptable should you spend time on one-step execution eval and then closed-loop replay.
