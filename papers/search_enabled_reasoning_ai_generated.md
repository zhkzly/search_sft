# Search-Enabled Reasoning / Web Search Agents (2023-2026) - Notes (ai_generated)

This note collects core papers on LLM agents that *actively* decide when to search/browse/retrieve, with an eye toward transferring the same ideas to **repo-level context finding** (e.g., deciding when to `rg`, which files to open, and when to stop).

## Core Papers (6-10)

1. **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** (2025, arXiv preprint)
   - URL: `https://arxiv.org/abs/2503.09516`
   - What it is: RL framework that trains an LLM to interleave multi-turn reasoning with issuing (multiple) search queries; emphasizes stable training via retrieval-token masking and outcome-based rewards.
   - Repo-level insight: treat `search(query)` as `repo_search(pattern)`; RL can learn *query refinement loops* (broad -> narrow) and stop once evidence coverage is sufficient.

2. **R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning** (2025, arXiv preprint)
   - URL: `https://arxiv.org/abs/2503.05592`
   - What it is: two-stage, outcome-based RL to enable autonomous search invocation during reasoning; designed to work without process reward or distillation cold-start.
   - Repo-level insight: outcome-only RL can already teach "invoke search vs not" if you can define a reliable end reward (e.g., patch passes tests / correct file located); but you must manage exploration cost (search steps explode).

3. **R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning** (2025, arXiv preprint)
   - URL: `https://arxiv.org/abs/2505.17005`
   - What it is: two-stage training (SFT cold-start + RL) to adaptively use internal vs external knowledge; adds reward mechanisms and a memorization mechanism to assimilate retrieved info.
   - Repo-level insight: explicitly reward "use internal knowledge first" (e.g., infer likely file/module) and only search when uncertainty is high; add memory of discovered symbols/files to reduce repeated `rg`.

4. **Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs** (AutoRefine) (2025, arXiv preprint)
   - URL: `https://arxiv.org/abs/2505.11277`
   - What it is: RL post-training with a "search-and-refine-during-think" loop; inserts explicit refinement/distillation steps between search calls; uses retrieval-specific rewards plus answer correctness (mentions GRPO).
   - Repo-level insight: make the agent *summarize and compress evidence* between tool calls (e.g., keep a running structured "facts" scratchpad about file paths, APIs, call sites) to reduce subsequent searches and context bloat.

5. **Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty** (2025, EMNLP main; arXiv)
   - URL: `https://arxiv.org/abs/2505.17281`
   - What it is: formalizes over-search / under-search; ties inefficiency to uncertainty about knowledge boundaries; proposes a GRPO variant with a confidence threshold to reward high-certainty search decisions.
   - Repo-level insight: train a small "uncertainty head" or confidence proxy over "do we need to search more?"; use a thresholded decision to stop early when the repo evidence already supports the next action.

6. **Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging** (InForage) (2025, arXiv preprint)
   - URL: `https://arxiv.org/abs/2505.09316`
   - What it is: RL framing inspired by Information Foraging Theory; explicitly rewards intermediate retrieval quality; introduces human-guided iterative search+reason trajectories for complex web tasks.
   - Repo-level insight: define intermediate rewards for "retrieval quality" at each step (e.g., did the retrieved snippet contain the target symbol/keyword? did it reduce candidate file set entropy?), not just final answer.

7. **LeTS: Learning to Think-and-Search via Process-and-Outcome Reward Hybridization** (2025, arXiv preprint)
   - URL: `https://arxiv.org/abs/2505.17447`
   - What it is: hybridizes stepwise process reward with final outcome reward to avoid ignoring intermediate search/reasoning correctness, without extra annotation.
   - Repo-level insight: process reward can be derived from *self-consistency of intermediate states* (e.g., "the file path mentioned must exist", "the opened file contains claimed identifier") to avoid wasting steps.

8. **ZeroSearch: Incentivize the Search Capability of LLMs without Searching** (2025, arXiv preprint)
   - URL: `https://arxiv.org/abs/2505.04588`
   - What it is: trains search capability with RL without calling a real search engine by using an LLM retriever to generate (good + noisy) pseudo-documents; uses curriculum degradation of doc quality.
   - Repo-level insight: you can simulate repo search results offline to cheaply train policies: use an indexer to sample true-positive chunks + hard negatives; progressively add noise (irrelevant files) to train robust stopping/querying.

9. **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** (2023, arXiv; later published variants exist)
   - URL: `https://arxiv.org/abs/2310.11511`
   - What it is: single LM uses special "reflection tokens" to decide when to retrieve and how to critique retrieved passages and its own generations; enables controllable retrieval at inference.
   - Repo-level insight: reflection tokens map well to a small context-finder model: emit `NEED_SEARCH`, `OPEN_FILE(path)`, `ENOUGH_CONTEXT` style control tokens that gate tool calls for a bigger code model.

10. **Active Retrieval Augmented Generation** (FLARE) (2023, arXiv; also appears in EMNLP-context discussions)
   - URL: `https://arxiv.org/abs/2305.06983`
   - What it is: triggers retrieval during generation based on low-confidence spans; uses forward-looking next-sentence prediction as retrieval query; regenerates low-confidence content with retrieved evidence.
   - Repo-level insight: for repo context, confidence-triggered retrieval becomes "if the model is unsure which module/function, call `rg`"; forward-looking query is "predict the next needed identifier/API name", then search for it.

## How These Works Trigger Search

Common trigger mechanisms you can port to repo-context finding:

- **Uncertainty-triggered**: search when confidence is low (FLARE), or uncertainty about knowledge boundary is high (Search Wisely).
- **Deliberation-triggered**: search when a reasoning step requires missing facts (Search-R1 / R1-Searcher family).
- **Token/control-triggered**: dedicated tokens that explicitly choose retrieval vs generation (Self-RAG).
- **Policy-triggered via RL**: search is a discrete action in the trajectory; RL learns when it improves outcome (Search-R1, R1-Searcher, AutoRefine, InForage, LeTS).

## When They Stop Searching

Typical stopping criteria (explicit or learned):

- **Outcome sufficiency**: stop once final answer can be produced with high reward probability (outcome-only RL).
- **Confidence threshold / boundary**: stop when uncertainty is below a threshold (Search Wisely-like).
- **Evidence saturation**: stop when refinement step produces a stable, non-contradictory evidence set (AutoRefine-like).
- **Budget-awareness**: implicit via step penalties or curriculum/noise (ZeroSearch-style training can mimic constrained budgets).

## How They Use Retrieved Results

Patterns that matter for repo-context selection:

- **Interleaving**: retrieve -> reason -> retrieve (Search-R1; also aligns with multi-hop code understanding).
- **Refinement/compression between searches**: distill evidence to avoid accumulating raw context (AutoRefine).
- **Self-critique / reflection**: evaluate if retrieved info actually supports the claim (Self-RAG).
- **Intermediate retrieval-quality reward**: optimize the *usefulness* of each retrieval step, not only final correctness (InForage, LeTS).

## Transfer to Repo-Level Context Finder (Concrete Mapping)

- Replace web `search(query)` with repo tools:
  - `rg(pattern)` / symbol index query / AST query / open file / jump-to-definition / callgraph expansion.
- Define state as a compact "evidence memory":
  - candidate files/symbols, extracted signatures, constraints, and remaining unknowns.
- Use the same trigger/stop signals:
  - uncertainty head, reflection tokens, process checks (file exists, symbol exists), and budget penalties.
- Train to minimize redundant steps:
  - penalize repeated searches with similar queries; reward "hit" retrievals that reduce candidate set entropy.

