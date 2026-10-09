# Archived local evidence

These sanitized summaries preserve the author's already published measurements, source filenames and original report SHA256 values. `deepseek.json` and `deepseek-prefill.csv` derive from the curated 2026-10-07 technical report; `qwen.json` retains only valid `full512` and `validated_task` groups from the local v0.7.0 report, including stage, sample counts and actual prompt/output lengths. `deepseek-cache.json` records the later bounded concurrency/restart acceptance.

Historical native binary/proof hashes identify those archived inputs. Freshly compiled binaries and the path-sanitized published metadata proof can have different file hashes; the release receipts and `SHA256SUMS` identify the current build/download. Archive timing values are never overwritten with portable package runs. New release verification belongs in a separate result directory and is summarized in `docs/VALIDATION.md`.

See the bilingual result documents for aggregation, warmup rules, prefix reuse, dense-tail fallback, sampled-memory limits and the distinction between text generation and completed agent workflows.
