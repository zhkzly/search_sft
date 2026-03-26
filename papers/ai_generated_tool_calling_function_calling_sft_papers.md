# AI-Generated Notes: Tool/Function Calling SFT (2023-2026) for Repo/Context-Finding Agents

This note targets training a small "context finder" model that assists CLI coding agents (Codex, Claude Code) to locate repo context with fewer search steps.

## Key Papers (6-10)

1. Toolformer: Language Models Can Teach Themselves to Use Tools (2023, arXiv)
   URL: `https://arxiv.org/abs/2302.04761`
   Relevance:
   It operationalizes tool use as next-token prediction with explicit API-call spans inside the model output. The data generation idea (insert candidate tool calls, keep those that improve likelihood) is directly applicable to creating "minimal context" trajectories: generate multiple tool-call candidates (rg, open file, etc) and keep ones that reduce uncertainty / improve final answer correctness. The paper also makes the "when to call tools" decision a learnable behavior, not hard-coded.

2. API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs (2023, arXiv)
   URL: `https://arxiv.org/abs/2304.08244`
   Relevance:
   Provides runnable tool-use evaluation with dozens of tools plus an instruction-tuning set with tool dialogues and API calls. For your repo/context finder, the important lesson is to build an executable evaluator (did the tool call succeed; were arguments valid; did it move the task forward) and not only judge text quality. Their breakdown of failures (planning, retrieving, calling) maps well to "choose the right repo tool" vs "form the correct rg query" vs "read the right file region".

3. Gorilla: Large Language Model Connected with Massive APIs (2023, arXiv)
   URL: `https://arxiv.org/abs/2305.15334`
   Relevance:
   Shows that accurate argument generation and hallucination mitigation is the core bottleneck for API calling. For repo navigation, "argument correctness" corresponds to correct command flags, correct file paths, and correct symbol names. Gorilla also highlights retrieval augmentation over tool documentation (API specs) as a pragmatic way to keep tool-use robust when tools change; for CLI agents, this is analogous to retrieving "tool schema + usage examples" for `rg`, `sed`, `python -m`, etc.

4. ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases (2023, arXiv)
   URL: `https://arxiv.org/abs/2306.05301`
   Relevance:
   It is explicitly about training compact models to generalize tool use to unseen tools, using multi-agent simulation to generate tool-use corpora. The simulation concept transfers to repo/context finding: build a sandboxed repo environment and simulate tasks where the optimal behavior is "a small number of targeted reads/greps" rather than broad search. This is also evidence that you can get meaningful tool generalization without huge models if the data covers diverse tool patterns.

5. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs (2023, arXiv)
   URL: `https://arxiv.org/abs/2307.16789`
   Relevance:
   Introduces ToolBench (instruction-tuning dataset for tool use) and a training+evaluation pipeline. The multi-tool setting is relevant to your "MCP-like context finder" because repo context finding typically needs tool chaining (rg -> open file -> extract snippet -> decide stop). Their automated path annotation is an existence proof for generating multi-step tool traces at scale, which you can adapt using deterministic rules plus a strong teacher model.

6. Tool Learning with Foundation Models (2023, arXiv; survey-style systematization)
   URL: `https://arxiv.org/abs/2304.08354`
   Relevance:
   Useful as a taxonomy to avoid mixing problem types: tool understanding, tool selection, argument filling, tool-result integration, and planning are distinct skills and often need distinct supervision signals. For your project, it argues for explicit interfaces and stable evaluators; this aligns with building a narrow "repo context toolset" (rg/open/read_symbol) with a consistent JSON schema rather than ad-hoc CLI strings.

7. StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models (2024, arXiv)
   URL: `https://arxiv.org/abs/2403.07714`
   Relevance:
   The key transferable point is stability: real APIs change and break evaluation. Repo environments also change (branch differences, dependency upgrades), so caching and simulators can make training/eval reproducible. If your RL stage uses online tools (real filesystem, real tests), you will want a "virtualized" interface (snapshotted repo states, cached outputs) for stable reward.

8. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets (2024, NeurIPS Datasets & Benchmarks; arXiv)
   URL: `https://arxiv.org/abs/2406.18518`
   Relevance:
   Strong blueprint for generating function-calling datasets with verification, not just LLM-labeled calls. The three-stage verification idea (format, execution, semantic check) maps directly to CLI agent supervision: (a) tool-call JSON parses, (b) command executes successfully, (c) the retrieved context actually contains the needed symbol/info. This is the closest match to "SFT that teaches tool-calling + reduces wasted search".

9. NexusRaven: A Commercially-Permissive Language Model for Function Calling (2023, NeurIPS workshop / OpenReview; arXiv)
   URL: `https://arxiv.org/abs/2308.12950`
   Relevance:
   Demonstrates function calling with a data curation pipeline that avoids proprietary model distillation, plus "demonstration retrieval augmentation". For your setting, this suggests a lightweight approach: store a small bank of exemplar repo-tool trajectories and retrieve the closest examples at inference time, reducing both hallucinated tool usage and the need for many exploratory searches.

10. The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models (2025, ICML; PMLR)
   URL: `https://proceedings.mlr.press/v267/patil25a.html`
   Relevance:
   While not an SFT method, BFCL is a high-signal evaluation framework for function calling, including serial/parallel calls and AST-based checking. The direct takeaway is how to score tool calls robustly, separating "well-formed" from "executable" from "semantically correct". This can inspire your own repo-tool benchmark (did it call rg correctly; did it open the correct file; did it stop early when enough context is found).

## Practical Takeaways for "Learn Tools First, Then Use Fewer Searches"

1. Split capability into phases even inside SFT.
   First teach strict formatting and executability (JSON schema, tool name, arguments), then teach tool choice, then teach minimal-step planning.

2. Verification beats pure imitation for tool calls.
   For function calling, the tool call is only meaningful if it runs. Use execution-based filters (APIGen style) to remove hallucinated paths.

3. Generalization to unseen tools comes from interface regularity.
   ToolAlpaca and ToolLLM both benefit from consistent tool documentation and structured interfaces; for repo tools, keep a small set of composable primitives with stable schemas.

4. Demonstration retrieval is a strong low-cost lever.
   Gorilla and NexusRaven show retrieval over tool docs or exemplar traces can reduce hallucinations and reduce "trial searches".

5. Benchmarks matter: build your own "repo-function-calling benchmark".
   BFCL and StableToolBench show that evaluation design heavily shapes progress; define metrics for both success and cost (steps, tokens, wall time).

## Suggested Data Format + Supervision Signals (Repo/Context Finder)

### Minimal schema (trajectory)

Use a chat log with explicit tool calls and tool results. Keep tool results out of loss (teacher forcing only on assistant "thought/toolcall" tokens).

Recommended fields per step:

- `messages`: list of `{role, content}` for user/system/assistant
- `tool_calls`: list of `{name, arguments_json}` emitted by assistant
- `tool_result`: `{ok, stdout, stderr, metadata}` (observation)
- `labels` (optional but useful):
  - `tool_is_needed`: binary (should call tool vs answer now)
  - `tool_name`: categorical (which tool)
  - `arg_valid`: binary (arguments parse/executable)
  - `delta_progress`: scalar (did this step increase "context coverage")
  - `stop_now`: binary (enough context found)

### Core tools for repo/context finding

Keep toolset small, stable, and composable:

- `search` (rg-like): `{query, glob?, path?, max_results?}`
- `open` (file read): `{path, start_line, end_line}`
- `symbols` (optional, AST index): `{symbol_name, kind?, file_hint?}`
- `summarize` (optional): `{text, instruction}` for compressing long snippets

### Supervision signals that specifically reduce search steps

- Add a budget token in the prompt (max_tool_calls or max_search_calls) and include trajectories that succeed under tight budgets.
- Include contrastive pairs:
  - "good" minimal path vs "bad" wandering path for the same task, with the same final answer target.
- Add explicit `STOP` action:
  - teach the model to stop calling tools once it has evidence (stop_now=1).
- Prefer step-level executability checks:
  - drop or downweight steps where tool call fails to execute or returns empty/noisy output.

