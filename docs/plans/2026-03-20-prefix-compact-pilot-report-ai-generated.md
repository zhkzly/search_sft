# Prefix Compact Pilot Report (AI-Generated)

## Files Generated

Compact pilot datasets:

- [step_level_compact_pilot_k1_state_on.jsonl](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/data/step_level_compact_pilot_k1_state_on.jsonl)
- [step_level_compact_pilot_k2_state_on.jsonl](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/data/step_level_compact_pilot_k2_state_on.jsonl)
- [step_level_compact_pilot_k3_state_on.jsonl](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/data/step_level_compact_pilot_k3_state_on.jsonl)

Example previews:

- [step_level_compact_pilot_k1_examples.json](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/data/step_level_compact_pilot_k1_examples.json)
- [step_level_compact_pilot_k2_examples.json](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/data/step_level_compact_pilot_k2_examples.json)
- [step_level_compact_pilot_k3_examples.json](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/data/step_level_compact_pilot_k3_examples.json)

Source raw pilot:

- [step_level_raw_pilot.jsonl](/home/zkl/pycodes/research/search_sft_code/SimpleDeepSearcher/sft/data/step_level_raw_pilot.jsonl)

## Summary

The compact prefix pipeline was run on the current raw pilot with `state_mode=on`.

Compared variants:

- `K=1`
- `K=2`
- `K=3`

Where `K` is the number of recent canonical `assistant(tool_call) + tool(result)` pairs preserved in the recent window.

## Quantitative Results

### `K=1`

- kept samples: `250`
- repos: `10`
- action steps: `231`
- stop steps: `15`
- average recent window messages: `1.86`
- average older unmapped turns: `1.00`
- average state evidence items: `1.99`
- average candidate files in state: `4.65`

### `K=2`

- kept samples: `228`
- repos: `10`
- action steps: `210`
- stop steps: `14`
- average recent window messages: `3.57`
- average older unmapped turns: `0.80`
- average state evidence items: `1.75`
- average candidate files in state: `4.11`

### `K=3`

- kept samples: `214`
- repos: `10`
- action steps: `198`
- stop steps: `12`
- average recent window messages: `5.10`
- average older unmapped turns: `0.69`
- average state evidence items: `1.50`
- average candidate files in state: `3.67`

## Interpretation

The expected trend holds:

- smaller recent windows preserve more samples
- larger recent windows retain more local raw trace but reduce retention

Current trade-off:

- `K=1` maximizes retention
- `K=2` is the balanced default
- `K=3` is the higher-fidelity but lower-retention option

## Current Recommendation

The most reasonable `v0` default remains:

- `K=2`
- `state_mode=on`
- deterministic clipping

Reason:

- retention is substantially better than strict full-prefix canonicalization
- recent context is not reduced too aggressively
- state still carries older searchable evidence

However, `K=1` should remain a real baseline for training ablation because it has the highest sample retention.
