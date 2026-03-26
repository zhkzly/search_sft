# Repo Context Finder Research Plan (ai_generated)

Date: 2026-03-26

## 1. Goal

Your target is not another full answer model. It is a **small repo-context policy model** that helps a stronger CLI agent such as Codex or Claude Code decide:

- whether to search at all,
- which repo tool to call,
- what query / path / symbol to use,
- when enough context has already been found.

The cleanest formulation is:

> train a small model to map `(task, repo state, current evidence, budget)` to a sequence of **context-finding actions** that maximizes downstream success while minimizing search cost.

This is closer to **tool-routing + retrieval planning + stopping** than to generic code generation.

## 2. Main Research Thesis

Based on the literature, the strongest route is a **three-stage recipe**:

1. **Trajectory collection + filtering**
   Collect real repo interaction traces from strong agents and turn them into verifiable context-finding episodes.
2. **SFT for tool correctness and search behavior**
   First teach the model to emit valid tool calls and stop decisions, then teach it to prefer short effective traces.
3. **RL for efficiency under quality constraints**
   Optimize for fewer search steps only after the model already knows how to find the right context.

This follows the direction of:

- [SimpleDeepSearcher](https://arxiv.org/abs/2505.16834)
- [R1-Searcher](https://arxiv.org/abs/2503.05592)
- [R1-Searcher++](https://arxiv.org/abs/2505.17005)
- [Search-R1](https://arxiv.org/abs/2503.09516)
- [Repoformer](https://arxiv.org/abs/2403.10059)
- [OTC: Optimal Tool Calls via Reinforcement Learning](https://arxiv.org/abs/2504.14870)

The recurring pattern is the same:

- pure end-to-end imitation is not enough,
- pure outcome-only RL is too sparse,
- efficiency must be optimized explicitly,
- search should become **conditional**, not default.

## 3. Problem Formulation

### 3.1 State

The model should not observe raw full repo text. It should observe a structured state:

- user task / issue / question,
- repo summary:
  - directory tree sketch,
  - language,
  - build/test entrypoints,
  - optional symbol index summary,
- current evidence memory:
  - candidate files,
  - candidate symbols,
  - snippets already read,
  - extracted facts,
- recent tool trajectory:
  - previous tool call,
  - previous tool result summary,
  - repeated-query markers,
- remaining budget:
  - search budget,
  - total tool-call budget,
  - latency/token budget.

This matches the lesson from [Search Wisely](https://arxiv.org/abs/2505.17281), [Self-RAG](https://arxiv.org/abs/2310.11511), [Repoformer](https://arxiv.org/abs/2403.10059), and [R1-Searcher++](https://arxiv.org/abs/2505.17005): the agent must know both its uncertainty and its remaining budget.

### 3.2 Action Space

Keep the action space **small, typed, and stable**. Recommended actions:

- `SEARCH_FILES(query, glob?, top_k?)`
- `SEARCH_SYMBOL(symbol, kind?, file_hint?)`
- `OPEN_FILE(path, start_line?, end_line?)`
- `OPEN_SYMBOL(path_or_symbol_id)`
- `EXPAND_GRAPH(node_id, edge_type?, budget?)`
- `RERANK(candidates, reason?)`
- `WRITE_MEMORY(summary)`
- `STOP_SEARCH`

This is much better than letting the small model emit arbitrary shell.

The best supporting literature here is:

- [Toolformer](https://arxiv.org/abs/2302.04761)
- [ToolLLM](https://arxiv.org/abs/2307.16789)
- [APIGen](https://arxiv.org/abs/2406.18518)
- [BFCL](https://proceedings.mlr.press/v267/patil25a.html)
- [GraphCoder](https://arxiv.org/abs/2406.07003)
- [RANGER](https://arxiv.org/abs/2509.25257)

### 3.3 Output

The small model can output one of two things:

1. **next action only**
   Better for tight control and RL.
2. **short action plan + next action**
   Better for interpretability and SFT.

I recommend:

- during SFT: `reasoning + next action`
- during RL: optimize only the action policy and a compact scratchpad

## 4. Data Collection Strategy

### 4.1 Where traces should come from

You need three data sources.

#### A. Real strong-agent traces

Run Codex / Claude Code / strong internal agents on:

- SWE-bench Verified style issue resolution,
- repo QA,
- bug localization,
- function/path lookup,
- refactor tasks,
- test-writing tasks.

This is the highest-value source because it reflects real CLI behavior.

#### B. Programmatically generated retrieval tasks

Build tasks from repo artifacts:

- given a changed patch, recover touched files,
- given a symbol description, recover its implementation file,
- given a failing test, recover the bug-related module,
- given a commit message, recover impacted files.

This is useful for large-scale cheap trajectory generation.

Relevant papers:

- [RepoCoder](https://arxiv.org/abs/2303.12570)
- [RepoBench](https://arxiv.org/abs/2306.03091)
- [CrossCodeEval](https://arxiv.org/abs/2310.11248)
- [RepoQA](https://arxiv.org/abs/2406.06025)
- [Repository-level Code Search with Neural Retrieval Methods](https://arxiv.org/abs/2502.07067)
- [Repository-Aware File Path Retrieval via Fine-Tuned LLMs](https://arxiv.org/abs/2510.08850)

#### C. Synthetic teacher rollouts

Use a strong teacher to generate multiple candidate context-finding traces per task, then verify them.

This is exactly the spirit of:

- [SimpleDeepSearcher](https://arxiv.org/abs/2505.16834)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [ToolAlpaca](https://arxiv.org/abs/2306.05301)
- [APIGen](https://arxiv.org/abs/2406.18518)
- [ZeroSearch](https://arxiv.org/abs/2505.04588)

### 4.2 What must be logged

At minimum, log:

- task id,
- repo id / commit hash / branch,
- user instruction,
- full tool schema available to the agent,
- every assistant step,
- tool name,
- arguments,
- exit code,
- stdout / stderr or hashes + truncated content,
- which later step referenced this tool result,
- final outcome,
- final patch or final selected context set,
- total search/tool counts,
- wall-clock time,
- token usage.

### 4.3 What makes a trace high quality

A good trace is not simply a successful trace. It must be:

- executable,
- grounded,
- non-redundant,
- recoverable,
- short enough to teach search efficiency.

Good filtering heuristics:

- drop tool hallucinations,
- drop environment failures unrelated to reasoning,
- downweight long repetitive search loops,
- keep successful recovery traces,
- keep both “good minimal” and “bad wandering” trajectories for the same task.

This aligns with:

- [FireAct](https://arxiv.org/abs/2310.05915)
- [T-Eval](https://arxiv.org/abs/2312.14033)
- [ToolSandbox](https://arxiv.org/abs/2408.04682)
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)

## 5. How to Adapt This Repo’s Current Trajectory Format

Your current format already separates:

- assistant generation / tool call tokens,
- tool result tokens,
- and computes loss only on model-generated parts.

That is correct and should be preserved.

For the current `README_TRAJECTORY.md` format, I recommend adding optional fields:

```json
{
  "task": "...",
  "repo_meta": {
    "repo": "...",
    "commit": "...",
    "language": "...",
    "budget": {
      "max_search_calls": 4,
      "max_tool_calls": 8
    }
  },
  "llm_interactions": [
    {
      "step_id": 0,
      "input_messages": [...],
      "response": {
        "content": "...reasoning...",
        "tool_calls": [...]
      },
      "tool_result": {...},
      "labels": {
        "valid_call": 1,
        "useful": 1,
        "redundant": 0,
        "stop_now": 0,
        "delta_progress": 0.7
      }
    }
  ],
  "final_labels": {
    "success": 1,
    "context_hit_at_5": 1,
    "search_calls": 2,
    "tool_calls": 4
  }
}
```

Key rule:

- continue to **not** compute loss on tool outputs,
- compute loss on reasoning, tool-call arguments, and explicit stop decisions.

## 6. SFT Stage Design

### 6.1 SFT-0: Tool syntax and interface correctness

The first SFT phase should not optimize “fewest searches”.

It should only optimize:

- tool name correctness,
- argument correctness,
- stop-token format,
- result integration format.

Use:

- [Toolformer](https://arxiv.org/abs/2302.04761)
- [Gorilla](https://arxiv.org/abs/2305.15334)
- [ToolLLM](https://arxiv.org/abs/2307.16789)
- [APIGen](https://arxiv.org/abs/2406.18518)
- [BFCL](https://proceedings.mlr.press/v267/patil25a.html)

### 6.2 SFT-1: Behavior cloning on effective context-finding traces

Now train on high-quality trajectories where the teacher:

- uses the right tool,
- uses targeted queries,
- does not over-search,
- stops when enough context is found.

This stage should include explicit examples of:

- `STOP_SEARCH`,
- `NO_SEARCH_NEEDED`,
- query refinement,
- file-path narrowing,
- symbolic lookup.

### 6.3 SFT-2: Preference / contrastive distillation for minimality

This is important.

For the same task, build pairs:

- short successful trajectory vs long successful trajectory,
- successful no-search trajectory vs unnecessary-search trajectory,
- correct stop vs delayed stop,
- correct file-path trace vs wrong-subsystem trace.

Train with one of:

- DPO / IPO style trajectory preferences,
- pairwise ranking loss over action sequences,
- step-level margin loss.

This is often lower risk than jumping directly into RL.

### 6.4 Recommended supervision signals

Per step, keep:

- `valid_call`
- `arg_valid`
- `useful`
- `redundant`
- `harmful`
- `stop_now`
- `delta_progress`

The most important new labels for your objective are:

- `stop_now`
- `useful`
- `redundant`

These directly shape search efficiency.

## 7. RL Stage Design

## 7.1 Why RL is needed

SFT can teach “how to search”.

RL is needed to teach:

- “search less when possible”,
- “do not search when internal knowledge or current evidence is enough”,
- “trade off one more search step vs diminishing returns”.

The core supporting works are:

- [R1-Searcher](https://arxiv.org/abs/2503.05592)
- [R1-Searcher++](https://arxiv.org/abs/2505.17005)
- [Search-R1](https://arxiv.org/abs/2503.09516)
- [How to Train Your Deep Research Agent?](https://arxiv.org/abs/2602.19526)
- [OTC](https://arxiv.org/abs/2504.14870)
- [Search-P1](https://arxiv.org/abs/2602.22576)
- [TreePS-RAG](https://arxiv.org/abs/2601.06922)
- [ReasonRAG / Process vs. Outcome Reward](https://arxiv.org/abs/2505.14069)
- [PRIME](https://arxiv.org/abs/2502.01456)
- [Knowledgeable-R1 / Resisting Contextual Interference in RAG](https://arxiv.org/abs/2506.05154)

## 7.2 Recommended reward design

Do **not** use only a hard penalty on search count.

The most robust formulation is constrained or Lagrangian.

Recommended episode reward:

```text
R = R_quality
  + alpha * R_process
  - lambda_search * C_search
  - lambda_tool * C_tool
  - lambda_token * C_token
  - lambda_repeat * C_repeat
```

Where:

- `R_quality`
  - answer correctness, or
  - context hit / patch success / file-path recall
- `R_process`
  - useful search,
  - valid query,
  - evidence novelty,
  - correct stop
- `C_search`
  - number of search-like actions
- `C_tool`
  - total tool calls
- `C_token`
  - optional prompt/output budget cost
- `C_repeat`
  - repeated or near-duplicate searches

### 7.3 Concrete step-level reward candidates

Per step:

- `+1.0` valid tool call
- `+2.0` search result contains gold file / gold symbol
- `+1.0` candidate set entropy decreases
- `+1.0` retrieved file later used in final answer/patch
- `-1.0` repeated near-duplicate query
- `-1.0` search returns empty / obviously irrelevant output
- `-2.0` harmful detour into wrong subsystem
- `+2.0` correct early `STOP_SEARCH`
- `-2.0` premature stop

Final step:

- `+K` for task success or context oracle hit
- `0` or negative for failure

The exact weights should be tuned by validation under a quality floor.

### 7.4 Best RL formulation to start with

I would not start from pure outcome-only GRPO.

Recommended order:

1. **Constrained PPO / GRPO with outcome reward + cost penalty**
   Easy baseline.
2. **Add process shaping**
   Based on step usefulness and redundancy.
3. **Add pairwise or counterfactual advantages**
   Compare “with search” vs “without search” on the same task.

Best ideas from the literature:

- [OTC](https://arxiv.org/abs/2504.14870)
  reward correctness and tool efficiency jointly.
- [Knowledgeable-R1](https://arxiv.org/abs/2506.05154)
  compare with-retrieval vs without-retrieval trajectories.
- [PRIME](https://arxiv.org/abs/2502.01456)
  derive denser process signals from outcome.
- [Search-P1](https://arxiv.org/abs/2602.22576)
  path-centric reward shaping.
- [TreePS-RAG](https://arxiv.org/abs/2601.06922)
  estimate node utility through rollout trees.

### 7.5 Outcome-only vs step-level reward

#### Outcome-only reward

Pros:

- simple,
- reliable when final verification is strong,
- less label engineering.

Cons:

- sparse,
- poor credit assignment,
- often encourages blind search explosion early.

#### Step-level reward

Pros:

- better for stopping and redundancy reduction,
- directly targets search efficiency,
- easier to diagnose.

Cons:

- needs a good verifier or reward model,
- easier to reward-hack,
- more engineering.

Recommended conclusion:

> start with outcome + lightweight process shaping, not pure outcome and not heavy PRM from day one.

## 8. A Concrete End-to-End Experimental Plan

## Phase A. Build the offline environment

Use repo snapshots and deterministic tool wrappers.

For each repo snapshot, precompute:

- directory tree,
- file embeddings,
- symbol index,
- call/dataflow graph if available,
- patch oracle or gold file set,
- optional cached tool outputs.

This is important because [StableToolBench](https://arxiv.org/abs/2403.07714) and many tool-use papers show unstable environments can poison both SFT filtering and RL reward.

## Phase B. Build the first benchmark

I recommend a benchmark with three levels.

### B1. Context-only retrieval

Task:

- predict relevant files / symbols / spans.

Metrics:

- Hit@K
- Recall@K
- MRR
- nDCG
- latency
- tool-call count

Datasets:

- [RepoBench](https://arxiv.org/abs/2306.03091)
- [CrossCodeEval](https://arxiv.org/abs/2310.11248)
- [RepoQA](https://arxiv.org/abs/2406.06025)
- [Repository-level Code Search with Neural Retrieval Methods](https://arxiv.org/abs/2502.07067)

### B2. Context-to-answer / context-to-edit

Task:

- use found context to answer repo questions or localize edit region.

Metrics:

- answer EM / F1,
- file-path accuracy,
- patch-file coverage.

### B3. End-to-end issue resolution

Task:

- agent solves repo issue with the small model providing context policy.

Metrics:

- resolved rate,
- search calls,
- tool calls,
- latency,
- tokens,
- Pareto frontier.

Datasets:

- [SWE-bench](https://arxiv.org/abs/2310.06770)
- [SWE-bench Live](https://arxiv.org/abs/2505.23419)
- [OmniGIRL](https://arxiv.org/abs/2505.04606)

## Phase C. Train baselines

You need at least these baselines:

1. heuristic grep-first agent,
2. embedding retriever + fixed top-k,
3. big model direct search policy,
4. small model SFT-only policy,
5. small model SFT + preference,
6. small model SFT + RL.

## Phase D. Ablations

Essential ablations:

- no `STOP_SEARCH`,
- no budget token,
- no process reward,
- no redundancy penalty,
- no graph actions,
- no memory write/read,
- no contrastive minimality pairs,
- no offline simulated traces.

## 9. Recommended Observation / Memory Design

One strong insight from [R1-Searcher++](https://arxiv.org/abs/2505.17005), [AutoRefine](https://arxiv.org/abs/2505.11277), [Prometheus](https://arxiv.org/abs/2507.19942), and [MemCoder](https://arxiv.org/abs/2603.13258) is:

> repeated raw search is wasteful; agents need compressed working memory.

So the small model should maintain a compact working memory such as:

```text
Known relevant files:
- path_a.py
- utils/search.py

Known symbols:
- FooRetriever
- build_index

Open questions:
- where is ranking done?
- which file writes final output?

Rejected branches:
- docs/*
- tests/helpers/*
```

This memory can itself become a supervised target:

- after each useful step, update memory,
- at RL time, reward memory updates that reduce future search.

## 10. Main Pitfalls

### 10.1 Search cost collapse

If penalty is too strong, the model learns “do not search”.

Fix:

- constrained RL,
- automatic lambda tuning,
- quality floor on validation.

### 10.2 Benchmark leakage

SWE-style tasks can leak via issue text, old commits, or training contamination.

Fix:

- prioritize live or recent splits,
- use repo snapshots after the model cutoff when possible,
- report both static and live benchmarks.

Relevant warnings:

- [SWE-Bench+](https://arxiv.org/abs/2410.06992)
- [UTBoost](https://arxiv.org/abs/2506.09289)
- [SWE-bench Live](https://arxiv.org/abs/2505.23419)

### 10.3 Reward hacking

The model may learn to maximize cheap process rewards without truly improving context quality.

Fix:

- always keep a strong final objective,
- audit repeated-query patterns,
- evaluate on held-out repos,
- keep step reward tied to verifiable artifacts.

### 10.4 Tool/environment instability

Different repo states and tool outputs will produce non-stationary reward.

Fix:

- snapshot everything,
- cache tool outputs when possible,
- separate online real-world evaluation from offline training.

### 10.5 Overly broad action space

If the small model can emit arbitrary shell, learning becomes much harder.

Fix:

- constrain to typed actions first,
- only expand toolset after the first stable result.

## 11. Strong Initial Recommendation

If you want the most practical first paper-quality experiment, I recommend this exact sequence:

### Experiment 1

Train a **file-path retrieval policy model** only.

- Input:
  task + repo summary
- Output:
  top-k relevant files + `STOP`
- Supervision:
  patch files / oracle files / teacher traces
- Metrics:
  Hit@K, Recall@K, latency

This is the cleanest first target and is directly supported by:

- [Repository-Aware File Path Retrieval via Fine-Tuned LLMs](https://arxiv.org/abs/2510.08850)
- [RepoBench](https://arxiv.org/abs/2306.03091)
- [RepoQA](https://arxiv.org/abs/2406.06025)

### Experiment 2

Upgrade to a **multi-action context policy**.

- actions:
  search, open, symbol, stop
- train with SFT + pairwise minimality

### Experiment 3

Add **RL for search reduction** under a quality constraint.

- reward:
  quality - cost
- compare:
  outcome-only vs process-shaped vs constrained RL

### Experiment 4

Plug the small model into a strong code agent and measure:

- resolved rate,
- median search calls,
- p95 latency,
- token cost.

That experiment is the most convincing argument that the small model is useful to real CLI agents.

## 12. Recommended Paper Set to Read First

If you need a compact must-read list, start with these 15:

### Core search / agent RL

- [SimpleDeepSearcher](https://arxiv.org/abs/2505.16834)
- [R1-Searcher](https://arxiv.org/abs/2503.05592)
- [R1-Searcher++](https://arxiv.org/abs/2505.17005)
- [Search-R1](https://arxiv.org/abs/2503.09516)
- [How to Train Your Deep Research Agent?](https://arxiv.org/abs/2602.19526)
- [OTC](https://arxiv.org/abs/2504.14870)

### Core repo retrieval / selective search

- [RepoCoder](https://arxiv.org/abs/2303.12570)
- [RepoBench](https://arxiv.org/abs/2306.03091)
- [Repoformer](https://arxiv.org/abs/2403.10059)
- [RepoQA](https://arxiv.org/abs/2406.06025)
- [On The Importance of Reasoning for Context Retrieval in Repository-Level Code Editing](https://arxiv.org/abs/2406.04464)
- [RLCoder](https://arxiv.org/abs/2407.19487)

### Core tool learning / process supervision

- [Toolformer](https://arxiv.org/abs/2302.04761)
- [ToolLLM](https://arxiv.org/abs/2307.16789)
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)

## 13. Bottom Line

The most defensible framing is:

> learn a small **repo context policy** with typed tool actions, explicit stop behavior, and budget awareness; train it with filtered real trajectories plus synthetic teacher traces; then use RL to reduce search only after correctness is stable.

The main novelty opportunity is not merely “make search fewer”.

It is:

1. **turn repo context finding into a standalone trainable policy problem,**
2. **use real CLI traces as supervision,**
3. **optimize the quality-cost frontier instead of only final success,**
4. **show that a small policy model materially improves a stronger coding agent.**
