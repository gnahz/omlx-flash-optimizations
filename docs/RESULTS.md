# Published results and scope

[中文](RESULTS.zh-CN.md) · [Reproduce](REPRODUCE.md)

These are the author's archived local measurements from 2026-10-07 and the later qualified cache smoke, on Apple M5 Ultra (30 CPU / 64 GPU cores, 256 GiB, macOS 27.0.1). The portable release's new validation is recorded separately in [VALIDATION](VALIDATION.md). The archived numbers are not overwritten by a new release run.

## Exact metadata compression

The original mixed LSQ3-G128 checkpoint is already lossy. The additional compression described here is **lossless relative to that fixed checkpoint**: each original BF16 scale/bias pair is packed as a raw U32 word; a per-projection dictionary deduplicates those words, and fixed U16 indices select them. Across all 120 routed-expert projections, dictionaries contain 17,885–64,321 entries. The proof checks all raw-word round trips and full encoded readback. Integer quantization codes are unchanged; no model is requantized during packaging.

| Scope | Before | After | Saved |
|---|---:|---:|---:|
| Routed-expert metadata | 16,986,931,200 bytes | 8,503,863,760 bytes | 8,483,067,440 bytes |
| Resident model tensors, including metadata | 223.481091 GiB | 215.580619 GiB | **7.900472 GiB** |
| Routed-expert storage including metadata | 3.25 bits/weight | 3.125153 bits/weight | — |

Resident tensor totals exclude KV caches and computation workspaces, and retain DSpark and vision components. The model's underlying routed-expert quantizer stays at 3 bits and group size 128; “3.125153” describes storage overhead, not a new quantization bitwidth. SSD Engram offload is distinct from expert-weight offload: routed experts remain resident.

The direct prefill kernel performs dictionary lookup while dequantizing qualified BF16 inputs for the native tiled GEMM, with FP32 accumulation. The qualified dispatch is batch 1, at least 512 prefill tokens, top-k 6 and aligned affine 3-bit/G128 shapes. Unsupported geometry and tail/single-token work retain the checked dense path and its **405 MiB** reserve. Thus zero dense *bulk* calls does not imply zero metadata expansion throughout an entire request.

## Same-process direct-prefill A/B

A and B use the same compressed resident weights, decode kernels and async DSpark settings. A expands metadata for existing dense bulk prefill; B directly consumes dictionaries. Gate/up pairing stays off, prefill chunk is 8192, prefix reuse is disabled, temperature is zero and thinking is off. Each complete request emits 512 tokens. Prefill, decode and client TTFT are geometric means of measured requests.

| Context | A prefill t/s | B prefill t/s | Prefill change | A decode t/s | B decode t/s | A client TTFT | B client TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8K, three-case geometric mean | 2,107.57 | 2,196.22 | **+4.21%** | 76.10 | **76.07** | 3.931 s | 3.774 s |
| 16K, case 0 | 2,122.89 | 2,203.40 | **+3.79%** | 94.40 | **94.40** | 7.777 s | 7.495 s |
| 32K, case 0 | 2,114.76 | 2,194.11 | **+3.75%** | 97.22 | **97.23** | 15.581 s | 15.022 s |

The 8K decode geometric mean is **76.10 → 76.07 t/s (-0.045%)**. B's three case rates are 91.22 (code text), 71.59 (marketing text), and 67.40 (video/computer-use *plan* text) t/s. The 16K/32K decode rates, 94.40/97.23 t/s, cover only case 0 and should not be compared with the three-case 8K mean as a context-scaling result.

The protocol includes 42 full requests: 32 measured and ten discarded warmups; measured output totals 16,384 tokens. At 8K each case receives A/B warmups and two ABBA cycles (four measurements per arm); 16K/32K each use case 0, two warmups and one ABBA cycle. Complete output/finish/usage and every non-timing DSpark counter match. B records 20/40/80 direct bulk calls at 8K/16K/32K, zero dense bulk calls, and 20/40/80 dense tail calls. Earlier separate optimization gains are not added together.

The API prefill rate and prompt-evaluation duration include the server's first-token boundary; they are not isolated GPU-kernel timers. Client TTFT includes loopback overhead. The model is resident before the comparison, both arms are warmed, and “cold” refers to prefix reuse being zero. OS filesystem/Engram caches are not purged.

Archived physical footprints were sampled every 0.5 seconds: peaks **225.56 / 225.73 / 225.62 GiB** at 8K/16K/32K, with normal reported pressure. Between-sample peaks are unverified. The 240 GiB process budget is not a live 16 GiB available-memory guarantee. The portable HTTP driver records that it does not independently sample physical footprint.

## Qwen local replay

The fixed upstream checkpoint is `mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp@e171af86f499f1855b0fb71d781105e8dd609610`; the source base is oMLX v0.7.0. All rows below are medians of five valid requests with zero prefix reuse, thinking disabled and greedy fixed-depth-3 MTP. The two modes differ by prompt lookup. Core requests have exactly 512 generated tokens; task rows are checked through complete generated output and task contracts.

| Input / task | Actual output tokens | Fixed3 decode t/s | Fixed3 + lookup t/s |
|---|---:|---:|---:|
| 2K | 512 | 151.37 | 143.45 |
| 8K | 512 | 136.29 | 134.54 |
| 32K | 512 | 126.32 | 124.63 |
| `edit_file`, 1344 input tokens | 1278 | 176.38 | 245.63 |
| JSON change, 880 input tokens | 822 | 184.10 | 214.51 |

Lookup can help repeated/edit-heavy output and reduce throughput on these plain text prompts. The author does not claim a universal lookup uplift or take credit for the community lookup algorithm. The new work preserves safe policy handling in the scheduler and improves prefix retention. Differences from earlier oMLX releases were historical sequential comparisons, not a same-process causal A/B estimate. Community model-card or recipe speeds are not substituted for these local results.

The archive's warmed, prefix-reusing Qwen 8K/32K fixed3 medians were **0.111 / 0.229 seconds client TTFT**; lookup medians were **0.113 / 0.229 seconds**. These warm requests belong to the `warm` groups in the evidence and are excluded from the uncached table. The public-prefix tail optimization is reproduced in the separate baseline/full-patch prefix suite.

## DeepSeek cached profile evidence

The archived concurrency-8 acceptance used ten requests at 8242 input / 64 generated tokens: a serial reference, eight simultaneous cached streams, and a replay after orderly restart. All eight streams overlapped for 5.637 seconds, reached eight observed running requests, reused 8192 tokens, and matched generated token IDs and API outputs. Restart restored one checkpoint and reused the same boundary; inference-time cache disk-read counters did not increase. The sampled physical peak was 222.45 GiB. The store cap is 3 GiB, startup reads only once and background saves occur at idle 300-second intervals.

This is a bounded short-output smoke. It did not saturate the 3 GiB store, certify eight simultaneous maximum-length contexts, or measure sustained aggregate throughput. The release probe reports warm/parallel output equality and prefix reuse; the restart receipt/snapshot check is explicit in the reproduction guide.

## Evidence files

| Released evidence | Provenance |
|---|---|
| [`deepseek.json`](../results/published/deepseek.json) | Curated resident/dictionary proof and final prefill summaries; original source filenames and SHA256 retained |
| [`deepseek-prefill.csv`](../results/published/deepseek-prefill.csv) | Per-case final A/B figures and each report's SHA256 |
| [`deepseek-cache.json`](../results/published/deepseek-cache.json) | Sanitized archived concurrency-8 acceptance summary, original report SHA256 |
| [`qwen.json`](../results/published/qwen.json) | Valid full512 / validated-task groups from the local v0.7.0 report; stage and denominator preserved |
| [`fixtures/deepseek`](../fixtures/deepseek/manifest.json) | Three exact historical 8K requests and complete accepted outputs, with checksums |
| [`fixtures/qwen`](../fixtures/qwen/manifest.json) | Frozen core, task and cache payloads, with checksums |

The synthetic DeepSeek text requests ask for code, campaign drafts and a video/computer-use plan. The 512-token outputs are throughput fixtures, often truncated at the requested limit. They do not show a runnable engineering project completed, real browser/computer operations, full-logit identity, or an accuracy evaluation of the original quantization.
