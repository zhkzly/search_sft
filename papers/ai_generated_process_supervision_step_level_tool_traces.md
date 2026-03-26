# Process Supervision / Step-Level Eval / Tool Traces (2023-2026)

This note is AI-generated on 2026-03-26. It focuses on: process supervision, step-level evaluation, tool/execution traces, and how to collect/curate trajectories for SFT from repo-level CLI agents (Codex / Claude Code style).

## 1) Papers (6-10) with Takeaways for Trajectory Collection + SFT

### 1. Let's Verify Step by Step
- Year: 2023
- Venue/source: arXiv
- URL/arXiv: `https://arxiv.org/abs/2305.20050`
- Why it matters for trajectories + SFT:
  - Clear empirical argument for **process supervision** (step-level labels) vs outcome-only, and shows **active learning** improves label efficiency.
  - Transfer to tool-using agents: treat each tool interaction (plan -> call -> result integration) as a "step" and label correctness/usefulness per step, not just final success.
  - The PRM800K release is a concrete reference for how to write labeler instructions + label schema even if your domain is code/tooling.

### 2. PRM800K (dataset accompanying "Let's Verify Step by Step")
- Year: 2023
- Venue/source: dataset / GitHub (OpenAI)
- URL: `https://github.com/openai/prm800k`
- Why it matters:
  - Shows a scalable recipe for collecting **step-level correctness** labels and packaging them (raw labels + labeling guidelines).
  - For repo agents, the analogous artifact is a "TRM" (Tool/Trace Reward Model) dataset: step labels over tool calls such as `rg`, file-open, test-run, patch-apply.

### 3. Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
- Year: 2023
- Venue/source: arXiv
- URL/arXiv: `https://arxiv.org/abs/2312.08935`
- Why it matters:
  - Demonstrates **automatically constructed process-wise supervision** to train a step reward model, reducing human labeling.
  - For repo agents: you can synthesize step labels via verifiers (e.g., tests, typecheck, lint, diff-based constraints) plus LLM judges, then train a step reward model to score tool-use traces.

### 4. Deductive Verification of Chain-of-Thought Reasoning
- Year: 2023
- Venue/source: arXiv
- URL/arXiv: `https://arxiv.org/abs/2306.03872`
- Why it matters:
  - Frames verification as decomposed subprocesses; conceptually aligns with evaluating intermediate steps rather than only final output.
  - For trajectories: encourages designing trace formats that make intermediate "claims" verifiable (e.g., cite file/line, show grep match, show test output hash).

### 5. Toolformer: Language Models Can Teach Themselves to Use Tools
- Year: 2023
- Venue/source: arXiv (also appears as NeurIPS 2023 in some indexes)
- URL/arXiv: `https://arxiv.org/abs/2302.04761`
- Why it matters:
  - Key idea: generate candidate tool calls, execute them, and keep only calls that improve a training objective. This is a **self-supervised filter** for tool traces.
  - For repo trajectories: you can over-generate candidate context-fetch actions (e.g., multiple `rg` queries, multiple files) but keep only those that measurably help downstream success, reducing "junk search" in SFT data.

### 6. Making Language Models Better Tool Learners with Execution Feedback
- Year: 2023
- Venue/source: arXiv
- URL/arXiv: `https://arxiv.org/abs/2305.13068`
- Why it matters:
  - Uses **execution feedback** to learn when tool usage is appropriate and to reduce unnecessary tool calls.
  - For repo trajectories: log tool call success/failure, exit code, stderr patterns, and feed these as supervision signals; include negative examples where "no tool needed".

### 7. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
- Year: 2023
- Venue/source: arXiv
- URL/arXiv: `https://arxiv.org/abs/2307.16789`
- Why it matters:
  - Introduces ToolBench-style instruction tuning with **annotated solution paths** and automatic evaluation components.
  - For SFT: emphasizes that trajectory corpora should include (a) tool docs / affordances, (b) multi-tool chains, (c) realistic failure modes, (d) stable evaluation.

### 8. FireAct: Toward Language Agent Fine-tuning
- Year: 2023
- Venue/source: arXiv
- URL/arXiv: `https://arxiv.org/abs/2310.05915`
- Why it matters:
  - Shows that finetuning on a *small number* of high-quality trajectories can significantly improve agent performance, and that **diversity of trajectories** matters.
  - For repo agents: you should sample tasks across bugfix/refactor/feature/test-writing, and across repo sizes/languages, not just one benchmark.

### 9. T-Eval: Evaluating the Tool Utilization Capability Step by Step
- Year: 2023
- Venue/source: arXiv
- URL/arXiv: `https://arxiv.org/abs/2312.14033`
- Why it matters:
  - Decomposes tool utilization into sub-processes; suggests a taxonomy for step-level labeling: instruction following, planning, retrieval, understanding, review.
  - For trajectories: store enough structured info to later score each sub-process separately (e.g., was the query well-formed? did the chosen file actually contain relevant symbol?).

### 10. ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities
- Year: 2024 (arXiv 2024-08-08; also listed as 2025-03 on Apple ML Research page)
- Venue/source: arXiv / Apple ML Research
- URL/arXiv: `https://arxiv.org/abs/2408.04682`
- Why it matters:
  - Focuses on **stateful execution** and **trajectory-level intermediate milestones**, not just final answer.
  - For repo agents: evaluation and data collection must model state (repo HEAD, uncommitted diff, environment deps, test cache), and you can attach milestone rewards (e.g., "identified correct file", "reproduced failing test", "patch compiles").

## 2) How to Collect High-Quality Trajectories from Codex/Claude Code Repo Interactions

### 2.1 What to log (minimum viable schema)
- Task spec: user request + constraints + success definition (tests/lint/build).
- Repo state: repo URL (optional), commit hash/branch, diff base, and a content hash for each accessed file (avoid storing full code if privacy-sensitive).
- Agent messages: system prompt, tool-availability prompt, and the assistant's emitted tool calls.
- Tool calls:
  - `tool_name` (e.g., `rg`, `sed`, `cat`, `ls`, `pytest`, `npm test`)
  - arguments, start/end timestamps, exit code
  - stdout/stderr (truncate + store hash + optionally store full output in a separate secure store)
- Edit actions:
  - patches/diffs with file paths, or "edit script" actions if using an editor.
- Final outcome:
  - did it pass? which checks? runtime? tokens? total tool steps?

### 2.2 Make trajectories "verifiable"
- Prefer steps that produce externally checkable artifacts:
  - `rg` match lines, test logs, compiler errors, static analysis results.
- For each "claim" in reasoning, encourage the agent to anchor it:
  - "Found X in `path:line`" (the trace should include the tool output slice or a hash pointer).

### 2.3 Include both positive and negative examples
- Positive: correct tool selection and good stopping behavior.
- Negative:
  - unnecessary search (tool overuse),
  - tool hallucination (calling non-existent commands),
  - premature editing (editing before locating root cause),
  - wrong file selection, wrong query formulation.
- Keep negatives if they are recoverable and end in success; they are useful to train recovery policies.

### 2.4 Capture "stop" decisions explicitly
- Add an explicit action like `DONE` / `STOP_SEARCH` / `FINAL_ANSWER` to the action space.
- Log when the agent chose to stop searching and moved to implementation/answering. This is essential if your RL stage optimizes step count.

## 3) Data Filtering and Labeling Suggestions (Pragmatic)

### 3.1 Filtering heuristics (cheap, high impact)
- Drop episodes without a deterministic outcome signal (no tests/build/lint and no user confirmation).
- Drop episodes with tool I/O failures unrelated to reasoning (network outage, permission errors), unless your model must learn those environments.
- Truncate or drop trajectories with:
  - repetitive searches (near-duplicate `rg` queries),
  - huge unstructured outputs (HTML dumps) that are never referenced later.
- De-duplicate by (repo commit hash, task signature, final patch hash).

### 3.2 Step-level labels (what to label)
Core labels per tool step:
- `valid_call`: tool exists + args parse + command runs
- `success`: exit code indicates success (tool-specific)
- `useful`: output was used later (referenced in reasoning or affected the final patch)
- `redundant`: near-duplicate of a previous step with negligible new information
- `harmful`: made the trajectory longer or pushed it toward wrong hypothesis (e.g., grep wrong subsystem)

Optional structure labels (T-Eval style):
- `planning_quality`, `query_quality`, `retrieval_choice_quality`, `integration_quality`, `review_quality`

### 3.3 How to generate step labels at scale
- Weak supervision from artifacts:
  - mark `success` from exit code; mark `useful` if subsequent messages cite file/line that came from this tool output.
- Counterfactual / ablation labeling:
  - if removing a step leaves success unchanged, label it as redundant (approximate).
- LLM-judge labeling:
  - feed the step + next 1-2 turns + final outcome to a judge model to label `useful/harmful`.
- Human spot-check:
  - sample by uncertainty (active learning), or by high cost trajectories.

### 3.4 SFT target formatting (aligns with tool-trace training)
- Only backprop on:
  - reasoning / plan tokens and the exact tool-call tokens (so the model learns tool syntax).
- Do not backprop on:
  - raw tool outputs (treat as observations / `doc_gen`), to avoid memorizing noisy environment text and to keep the model from "predicting the world".

