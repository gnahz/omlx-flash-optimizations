# oMLX Flash optimizations

[中文](README.zh-CN.md) · [Full reproduction](docs/REPRODUCE.md) · [Methods and evidence](docs/RESULTS.md)

Pinned, ordered kernel and engine patches for **DeepSeek V4.1 Flash LSQ3-G128** and **Qwen3.8 Flash Next oQ6e-MTP** on Apple silicon. The repository contains my additions, their dependencies, frozen request fixtures, and independent benchmark drivers. Each model uses its own tested oMLX base: DeepSeek `0.7.0.dev4`, Qwen `0.7.0`.

## My changes

- **Exact DeepSeek quantization-metadata compression.** The 120 routed-expert projections store U16 dictionary indices into raw U32 words containing BF16 scale/bias pairs. Every original metadata bit round-trips; integer weight codes stay unchanged. Resident model tensors decrease **223.481 → 215.581 GiB**, saving **7.900 GiB**. This is exact compression of an existing lossy LSQ3 checkpoint, not a new quantizer.
- **Direct dictionary prefill.** A native tiled GEMM reads that representation directly during qualified bulk prefill, removing full metadata expansion from that path. The engine retains checked dense fallback for tails and unsupported shapes. Prefill improves **4.21% / 3.79% / 3.75%** at 8K / 16K / 32K in the same-process A/B test; decode is unchanged within that comparison.
- **Engine integration.** DeepSeek adds bounded exact-prefix retention and restart persistence. Qwen adds safe lookup-policy handling and public-prefix tail retention, with the qualified startup hooks moved into engine/scheduler patches. Benchmark and cached serving use separate profiles.

## Published results

My machine: **Apple M5 Ultra, 30 CPU / 64 GPU cores, 256 GiB, macOS 27.0.1**. Greedy, thinking disabled, concurrency 1, zero reused prefix tokens, 512 actual output tokens. DeepSeek uses geometric means: three scenarios at 8K, case 0 only at 16K/32K. Qwen is the median of five full-length requests with fixed MTP depth 3 and lookup disabled.

| Model / input | Prefill tokens/s | Decode tokens/s | Client TTFT |
|---|---:|---:|---:|
| DeepSeek, 8K | 2,196.22 | **76.07** | 3.774 s |
| DeepSeek, 16K | 2,203.40 | **94.40** | 7.495 s |
| DeepSeek, 32K | 2,194.11 | **97.23** | 15.022 s |
| Qwen, 2K | 2,819.29 | **151.37** | 0.735 s |
| Qwen, 8K | 3,885.47 | **136.29** | 2.128 s |
| Qwen, 32K | 3,912.49 | **126.32** | 8.440 s |

Qwen lookup is workload-dependent: validated `edit_file` / JSON-change replay medians were **176.38 → 245.63** / **184.10 → 214.51 tokens/s**, with complete outputs equal. Cold 8K text decode was **136.29 → 134.54 tokens/s** with lookup enabled. These are local measurements, not the community recipe author's measurements. The archived text fixtures measure generation throughput; they are not completed agent workflows. See [precise A/B results, cache results, and scope](docs/RESULTS.md).

## Reproduce

Use an arm64 Mac with the above memory capacity, Xcode/SDK 27.2, Git, `uv`, and `hf` on `PATH`. DeepSeek uses Python 3.13; Qwen uses Python 3.12. The preparation scripts install pinned dependencies and build kernels from source. Before running DeepSeek, set the wired-memory capacity explicitly: `sudo sysctl iogpu.wired_limit_mb=245760` (240 GiB). DeepSeek weights require **420.53 GiB** of download space plus build/cache space; the download retains original tensor shards, SSD Engram tables, and the exact-metadata overlay.

```bash
git clone https://github.com/gnahz/omlx-flash-optimizations.git
cd omlx-flash-optimizations
export DS_WORK="$PWD/work/deepseek" DS_MODEL="$PWD/models/deepseek"
python3 scripts/repro.py prepare --model deepseek --work-dir "$DS_WORK"
python3 scripts/repro.py download --model deepseek --model-dir "$DS_MODEL"
python3 scripts/repro.py verify --model deepseek --model-dir "$DS_MODEL"
"$DS_WORK/venv/bin/python" scripts/repro.py serve --model deepseek --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --profile benchmark
```

Keep that isolated localhost server running, then in another terminal use the same `DS_WORK` / `DS_MODEL` values:

```bash
"$DS_WORK/venv/bin/python" scripts/repro.py benchmark --model deepseek --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --profile benchmark --output-dir "$DS_WORK/results/cold-1"
```

The driver executes all 42 requests: 8K three scenarios, plus 16K / 32K case 0; ten full warmups are discarded. It retains request bytes, SSE, full outputs, real token usage, timings, DSpark counters and prefill-path counters. Every output must match its baseline warmup, and exact 8K outputs must also match the archived reference. Stop this release server with Ctrl-C before switching profiles or models.

```bash
export Q_WORK="$PWD/work/qwen" Q_MODEL="$PWD/models/qwen"
python3 scripts/repro.py prepare --model qwen --work-dir "$Q_WORK"
python3 scripts/repro.py download --model qwen --model-dir "$Q_MODEL"
python3 scripts/repro.py verify --model qwen --model-dir "$Q_MODEL"
python3 scripts/repro.py benchmark --model qwen --work-dir "$Q_WORK" --model-dir "$Q_MODEL" --suite all --output-dir "$Q_WORK/results/replay-1"
```

The Qwen driver owns isolated temporary servers for its cold core and cache suites. For everyday serving use `serve --model qwen ... --profile cached`. For DeepSeek cache acceptance use `serve --model deepseek ... --profile cached`, then `benchmark --model deepseek ... --profile cached` in another terminal. These profiles allow eight requests and keep their caches under the chosen work directory. [The full guide](docs/REPRODUCE.md) gives patch order, exact build flags, profiles, authentication, restart checks and failure criteria.

## Weights and attribution

- DeepSeek: [junwzhang/DeepSeek-V4.1-Flash-LSQ3-G128-MLX](https://huggingface.co/junwzhang/DeepSeek-V4.1-Flash-LSQ3-G128-MLX), pinned by [release metadata](profiles/release.json); [model card](docs/HF-MODEL-CARD.md). Quantization is described there; regeneration code is not required for reproduction.
- Qwen: [mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp/tree/e171af86f499f1855b0fb71d781105e8dd609610), revision `e171af86f499f1855b0fb71d781105e8dd609610`. These are upstream weights.
- DeepSeek baseline: [tacos8me/m5-ultra](https://github.com/tacos8me/m5-ultra/tree/4e4d5a81f8135249815a3f284c0ed5bbd1faee58), revision `4e4d5a81f8135249815a3f284c0ed5bbd1faee58`.
- Qwen baseline: [community recipe](https://github.com/sethforprivacy/Qwen3.8-Flash-Next-M5-Ultra-oMLX/tree/050233efcd0a765de9963ab21a01df0723a50530), revision `050233efcd0a765de9963ab21a01df0723a50530`, and [oMLX PR #4030](https://github.com/jundot/omlx/pull/4030).

[Apache-2.0](LICENSE) for repository additions; original notices and model licenses are preserved. See [third-party notices](THIRD_PARTY_NOTICES.md) and [release validation](docs/VALIDATION.md).
