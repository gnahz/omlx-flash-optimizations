# Frozen DeepSeek requests

`case0-8k.json`, `case1-8k.json`, and `case2-8k.json` are byte-exact historical requests. Each uses the model alias `deepseek-v41-lsq`, an exactly 8192-token formatted prompt, temperature 0, disabled thinking, streaming with usage, and 512 generated tokens. `manifest.json` pins the request bytes and raw prompt hashes; `reference.json` preserves the accepted complete outputs, finish reason, token usage and DSpark counters. Timing values inside that historical reference are not used as a speed baseline.

Case 0 asks for SQLite code text, case 1 asks for campaign drafts, and case 2 asks for a video/computer-use plan. These are synthetic throughput fixtures. Their length-limited outputs do not represent completed engineering projects or performed computer operations.

At 16K and 32K, `scripts/benchmark_deepseek.py` preserves case 0's exact task suffix, rebuilds the synthetic archive deterministically using the pinned CPU formatter/tokenizer, and requires exact prompt geometry. It records generated payload hashes and compares complete outputs against that context's fresh A warmup. The 8K historical golden does not apply to a changed input.

The separate cache smoke uses the same deterministic builder at exactly 8242 prompt tokens and generates 64 tokens, permitting an 8192-token reusable boundary with a real uncached suffix. Its short outputs are compared with its serial reference and are not included in the 512-token throughput summaries.
