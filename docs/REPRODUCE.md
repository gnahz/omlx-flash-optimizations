# Reproduction guide

[中文](REPRODUCE.zh-CN.md) · [Results](RESULTS.md) · [Validation](VALIDATION.md)

## 1. Requirements and fixed sources

The measured machine is an Apple M5 Ultra with 30 CPU cores, 64 GPU cores, 256 GiB unified memory and macOS 27.0.1. Use arm64 macOS and the full Xcode toolchain; the release builds were checked with SDK 27.2. The NAX Qwen kernels require SDK 26.2 or later; the DeepSeek JIT build sets deployment target 27.0. A smaller Mac is outside the published full-resident profile.

Install Git, `uv`, and the Hugging Face `hf` CLI. The locked environments use `huggingface_hub==1.33.0`. `hf download --help` and `hf cache verify --help` should be available. Public downloads need no token. DeepSeek downloads **420.53 GiB**, including the retained original shards, Engram and metadata overlay; reserve additional disk space for source builds, the download client's metadata and prefix caches.

```bash
git clone https://github.com/gnahz/omlx-flash-optimizations.git
cd omlx-flash-optimizations
export RELEASE_ROOT="$PWD"
export DS_WORK="$RELEASE_ROOT/work/deepseek" DS_MODEL="$RELEASE_ROOT/models/deepseek"
export Q_WORK="$RELEASE_ROOT/work/qwen" Q_MODEL="$RELEASE_ROOT/models/qwen"
uv python install 3.13.15 3.12.14
uv tool install --python 3.13.15 'huggingface_hub==1.33.0'
export PATH="$(uv tool dir --bin):$PATH"
xcrun --show-sdk-version
sudo sysctl iogpu.wired_limit_mb=245760
```

The last command sets a 240 GiB wired allocation capacity. The DeepSeek launcher checks this setting and physical memory before loading; it does not change host settings itself. A 240 GiB process budget leaves 16 GiB outside that budget, but does not promise a live 16 GiB free-memory floor. Run one full model at a time.

| Component | Fixed revision / environment |
|---|---|
| DeepSeek oMLX | `d4298ad1811bdd1b565660263380b52fc4ff71f0` |
| DeepSeek MLX | `59d600b5e64c238427d0f8d897ab7c682ef4d3d2` plus the ordered MLX patch |
| DeepSeek build Python | 3.13.15 (CPython 3.13 ABI) |
| Qwen oMLX | v0.7.0, `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40` |
| Qwen MLX / Metal | 0.32.2, Python 3.12.14 |
| DeepSeek weights | Published revision recorded in `profiles/release.json` |
| Qwen weights | `mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp@e171af86f499f1855b0fb71d781105e8dd609610` |

The source manifests and full dependency locks are in `profiles/deepseek/` and `profiles/qwen/`. Do not apply either series to an arbitrary newer oMLX checkout or share their Python environments.

## 2. Build the kernel and engine patches

```bash
python3 scripts/repro.py prepare --model deepseek --work-dir "$DS_WORK"
python3 scripts/repro.py prepare --model qwen --work-dir "$Q_WORK"
```

Preparation runs no model inference. It checks patch checksums, creates independent source directories, checks out the fixed commits, checks/applies each patch, installs the dependency lock, and builds the native extensions. Existing source directories are preserved instead of silently replaced. The generic command uses the required pinned interpreter through `uv` when necessary.

DeepSeek layout is `work/source/{mlx,omlx}`, `work/venv`, `work/build/glm`, and `work/assets`. Patch order:

1. `patches/deepseek/mlx/0001-tacos8me-mlx-and-portable-jit.patch`.
2. oMLX `0001-tacos8me-qualified-baseline.patch`.
3. oMLX `0002-exact-metadata-and-direct-prefill.patch`.
4. oMLX `0003-exact-prefix-cache-persistence.patch`.
5. oMLX `0004-portable-release-identities.patch`.

The direct prefill patch contains both the native dictionary GEMM and its engine dispatch. The cache patch contains retention, identity checks and persistence; the portable identities patch allows the published relative-path metadata sidecar. The first dependency patch is attributed by [source and revision](https://github.com/tacos8me/m5-ultra/tree/4e4d5a81f8135249815a3f284c0ed5bbd1faee58).

For a separately reviewable application/build sequence:

```bash
python3 scripts/prepare_deepseek.py --work-dir "$DS_WORK" --apply-only
python3 scripts/prepare_deepseek.py --work-dir "$DS_WORK" --check
python3 scripts/prepare_deepseek.py --work-dir "$DS_WORK" --build-only --python "$(uv python find 3.13.15)"
```

Use this sequence instead of the earlier generic `prepare` command on that work directory. The optional `--mlx-source` / `--omlx-source` arguments use a local clone as a read-only object source. Patch application is always to the new checkout. `source-receipt.json` pins all patched source files; `build-receipt.json` records the compiled binaries and build inputs.

The DeepSeek build uses CMake 4.1.0, setuptools 84.0.0, wheel 0.48.0, C++20, `MLX_METAL_JIT=ON`, `CMAKE_OSX_DEPLOYMENT_TARGET=27.0`, and `BASELINE_RUNTIME_METAL=ON`. The extension uses the same fetched nanobind **3.0.1** CMake package as the patched MLX build. Python's separately installed nanobind package is not substituted for it. Metal source is expanded with the chosen build interpreter and loaded through the qualified JIT path with safe math. Build paths are derived from the work directory.

Preparation downloads the [official `mlx-metal` 0.32.2 wheel](https://files.pythonhosted.org/packages/dd/cd/4e50bf325100e7165e13d025f264362bf0009196269f9eaf87f2c6e738a2/mlx_metal-0.32.2-py3-none-macosx_26_0_arm64.whl) only to extract its stock auxiliary `mlx.metallib`; no machine-specific binary is in Git. The wheel SHA256 is `e6abeac9ac5265830c9c1541b6f96e9be37a85c2446763a46ad466c63a3837ab`; the extracted 182,351,120-byte asset SHA256 is `dc59d1cceb1a5c7e578232e6e41e28e2c73c9463ac6dbc3886c3ee17ffc270ed`. Both are checked before use. That auxiliary library does not replace the patched MLX runtime.

Qwen prepares `work/{baseline,optimized}/source` and a separate `.venv` in each source checkout. `baseline` applies the first four patches; `optimized` applies all six, in the order recorded in `profiles/qwen/manifest.json`: isolated server runtime (`00`), community QSA capacity, community prompt lookup, lookup-policy safety, public-prefix tail retention, and durable scheduler integration. Both native builds use the pinned dependency lock. The common isolation patch places startup shim/reaper state under the explicit config directory. For selective preparation use `scripts/prepare_qwen.py --workdir ... --variant baseline|optimized|both --python ...`; `--patch-only` creates the patched sources without compiling. Attribution is [here](../THIRD_PARTY_NOTICES.md).

## 3. Download and verify the fixed weights

```bash
python3 scripts/repro.py download --model deepseek --model-dir "$DS_MODEL"
python3 scripts/repro.py verify --model deepseek --model-dir "$DS_MODEL"
python3 scripts/repro.py download --model qwen --model-dir "$Q_MODEL"
python3 scripts/repro.py verify --model qwen --model-dir "$Q_MODEL"
```

DeepSeek selects the verified Hub revision from `profiles/release.json`; it is not a moving `main`. Verification reads every file listed in `SHA256SUMS`. An explicit `--revision` is available for inspecting a different release, but that is no longer this fixed reproduction. The complete downloaded directory must include `config.json`, tokenizer/template, the original index and tensor shards, SSD Engram files, `LICENSE`, `ds41-conversion.json`, and `exact-metadata/` with the 120 dictionary projections, sidecar, and `metadata-proof.json`. Keep that layout intact. The model directory is not used to store service state.

Qwen downloads the existing upstream revision and uses `hf cache verify ... --fail-on-missing-files`; the launcher also checks the pinned config/index checksums and presence of all indexed shards. No Qwen weights are redistributed by this repository. Quantization provenance and the distinction between download size and resident size are documented in the [DeepSeek model card](HF-MODEL-CARD.md).

## 4. Reproduce uncached DeepSeek results

Terminal 1:

```bash
"$DS_WORK/venv/bin/python" scripts/repro.py serve --model deepseek --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --profile benchmark --port 8011
```

Terminal 2, from the same repository with the same exported directory values:

```bash
"$DS_WORK/venv/bin/python" scripts/repro.py benchmark --model deepseek --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --profile benchmark --output-dir "$DS_WORK/results/cold-1" --port 8011
```

The server binds only `127.0.0.1`; the model alias is `deepseek-v41-lsq`. A fresh API/control key is written to `work/config/benchmark/api-key.txt` with mode 0600. The driver reads that file and sends it only to the loopback server. It acquires `benchmark.lock`, attaches to the already running release server, and never starts or stops another service. The server itself holds `service.lock`, uses a separate model registry and config directory, and puts logs/cache/shim state under the work directory.

The benchmark profile has prefix cache disabled, concurrency 1, prefill chunk 8192, 240 GiB process ceiling, full routed-expert/DSpark/vision residency, Engram SSD offload, native decode on, native verify off, sparse/MHC/gather on, gate/up pairing **off**, native metadata prefill **on**, and async DSpark **on**. MTP drafts four tokens; requests are greedy with thinking disabled. A temporarily disables only direct metadata prefill, expanding metadata for the existing dense bulk GEMM; B enables it. Both retain the same compressed checkpoint and decode implementation. Admission controls refuse mutation while an inference request, model load, engine lease, or scheduler work is active.

All output directories must be new. Default core protocol:

| Context | Cases | Discarded warmups | Measured order | Measured requests |
|---|---|---:|---|---:|
| 8192 | 0, 1, 2 | A/B per case: 6 | two ABBA cycles per case | 24 |
| 16384 | 0 | A/B: 2 | one ABBA cycle | 4 |
| 32768 | 0 | A/B: 2 | one ABBA cycle | 4 |

Each request actually generates 512 tokens. Total: 42 requests, ten warmups excluded, 32 measurements, 16,384 measured output tokens. Frozen 8K request bytes are shipped verbatim. The 16K/32K prompts retain case 0's task suffix and deterministically extend the synthetic archive with the fixed CPU encoder/tokenizer. Counts must be exact; near-size inputs are rejected. Long-context output identity is checked against that context's A warmup, not the 8K golden.

For a CPU-only payload check or an 8K-only GPU replay:

```bash
"$DS_WORK/venv/bin/python" scripts/benchmark_deepseek.py --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --output-dir "$DS_WORK/results/payload-check-1" --prepare-only
"$DS_WORK/venv/bin/python" scripts/benchmark_deepseek.py --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --output-dir "$DS_WORK/results/8k-1" --context 8192
```

Success requires complete SSE termination, exact actual input/output counts, explicit `cached_tokens=0`, complete text/reasoning/finish identity, and equality of every DSpark non-timing counter with the baseline warmup (and archived golden at 8K). A must show dense bulk calls and zero direct calls; B must show direct bulk calls and zero dense bulk calls. Dense tail calls are permitted and reported. The driver checks gate/up off, async DSpark on, stable process ID and unchanged source/proof hashes, and restores the initial prefill setting in `finally`. Raw SSE, event timings, full outputs, log slices and reports survive failure. Native-symbol failures abort instead of measuring a fallback runtime.

The first request and each arm's JIT warmup are excluded. Here “uncached” means no prefix reuse; it does not mean purged OS/Engram filesystem caches or process-cold model loading. These drivers do not certify physical-memory peaks; the archived sampled footprints and their scope are in [RESULTS](RESULTS.md).

## 5. Reproduce Qwen core and cache comparisons

Stop the DeepSeek release server with Ctrl-C and wait for it to exit before loading Qwen. The Qwen driver starts and stops only its own isolated child servers:

```bash
python3 scripts/repro.py benchmark --model qwen --work-dir "$Q_WORK" --model-dir "$Q_MODEL" --suite all --output-dir "$Q_WORK/results/replay-1" --port 8012
```

`--suite core` runs frozen 2K/8K/32K throughput requests; `--suite tasks` replays the archived `edit_file` and JSON-change measurements with their full output contracts; `--suite prefix` runs cold/warm prefix comparisons and a separate semantic-correctness replay. `--suite all` includes all three. Core/tasks use greedy fixed MTP depth 3, first with lookup off, then with lookup on. Lookup parameters are n-gram 8, maximum draft 7, long-match threshold 32, long draft 14, verified-row exact maximum 4. Core servers use concurrency 1, explicitly clear their isolated cache before each request, and confirm zero cached tokens from actual usage. The fixed-depth lookup-policy guard remains present in both core variants. The prefix comparison uses the first-four-patch baseline versus the full six-patch build and records cache reuse separately. The output directory contains raw requests/SSE, full outputs, checks, timings, environment/source identities, and the summarized report. Runtime directories and generated tokens belong only to this run.

Core uses 36 requests / 30 measurements; tasks use 24 / 20 (one warmup plus five measured requests for each task in each lookup mode); prefix uses 51 / 36. `all` therefore uses **111 requests / 86 measurements**. Task acceptance checks the archived complete output SHA256, exact actual token lengths and stop reason, absence of reasoning/tool calls, unchanged Python AST except the requested function rename, and complete JSON equality except the requested field change. Generated Python is not executed by this release driver. For task results alone:

```bash
python3 scripts/repro.py benchmark --model qwen --work-dir "$Q_WORK" --model-dir "$Q_MODEL" --suite tasks --output-dir "$Q_WORK/results/tasks-1" --port 8012
```

Qwen server choices for manual use:

```bash
python3 scripts/repro.py serve --model qwen --work-dir "$Q_WORK" --model-dir "$Q_MODEL" --profile benchmark --port 8012
# Stop this release server with Ctrl-C before the next command.
python3 scripts/repro.py serve --model qwen --work-dir "$Q_WORK" --model-dir "$Q_MODEL" --profile cached --port 8012
```

`benchmark` chooses the baseline build, fixed3, concurrency 1; its normal upstream prefix cache remains available, so use the cache-clearing driver above to reproduce zero-reuse measurements. `cached` chooses the optimized build, fixed3lookup, concurrency 8, 8192-token blocks, public-prefix retention and `work/runs/cached`. Qwen uses a 232 GiB process guard, fully resident PLE, expert/A8/TurboQuant off, a 50 GB isolated SSD cache, and zero hot-cache capacity. Cached runtime reuse validates the full source package plus model config/index identities. A fresh API key is generated at each launch and printed to the terminal; an explicit `OMLX_API_KEY` environment value may be supplied for a chosen persistent key. Both profiles bind loopback and use an explicit config base directory.

## 6. DeepSeek cached serving and restart checks

This is a separate acceptance profile, not the published cold-performance comparison. After the benchmark server exits:

```bash
"$DS_WORK/venv/bin/python" scripts/repro.py serve --model deepseek --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --profile cached --port 8011
```

In another terminal:

```bash
"$DS_WORK/venv/bin/python" scripts/repro.py benchmark --model deepseek --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --profile cached --output-dir "$DS_WORK/results/prefix-1" --port 8011
```

The cached profile keeps native metadata prefill, gate/up off, async DSpark on and chunk 8192, but allows eight requests and a 3 GiB exact-prefix cap. It stores persistence in `work/cache/exact-prefix-v1`, reads SSD state only at startup, and saves when idle at 300-second intervals. Startup allows 900 seconds; orderly shutdown allows 120 seconds. Save generations are atomic and preserve the prior valid generation until the new one is complete. Restore fingerprints bind the model settings, tokenizer/config/index, patched runtime, metadata proof and draft signature.

The probe builds an exact 8242-token case-0 prompt, generates 64 tokens, primes then repeats it, and issues eight parallel copies. All complete outputs must equal the serial output; warm and parallel requests must report exactly 8192 reused tokens. The report includes observed scheduler concurrency. This checks bounded short-response prefix behavior, not eight simultaneous 32K/long-output capacity. The first request may use an already restored cache and is not labeled cold.

To check persistence, Ctrl-C this release server, wait for full exit, inspect `work/logs/cache-shutdown.json` for `saved: true`, then restart with the same cached command. Run the probe again into a new `prefix-after-restart-1` output directory. Its initial `cache.models` snapshot contains restore/persistence counters; the first request should reuse the restored boundary. The probe itself records `restart_assessed: false`: comparing the shutdown receipt, restored initial snapshot and request reuse is the explicit restart check. Never force-kill the service while expecting an orderly save.

## Troubleshooting and verification

- Missing or incompatible native symbols: retain the build logs/receipts, confirm the fixed base and SDK, then use a new work directory. Do not bypass the native check.
- Failed historical output or DSpark identity: retain the failed report. Confirm the pinned weights, source receipts, exact flags, greedy requests and isolated profile before interpreting any timing.
- HTTP/control errors: the driver retries connection failures while the owned server starts, within a 900-second readiness deadline. Authentication and HTTP status errors fail immediately. Inspect the service-owned log and ensure the matching profile/work directory/port and fresh key file.
- Reusing a result directory is rejected so evidence is not overwritten. Qwen cached source/model identity changes require a fresh runtime directory.

CPU checks:

```bash
python3 -m unittest discover -s tests -p 'test_benchmark_deepseek.py'
python3 scripts/serve_deepseek.py --help
python3 scripts/benchmark_deepseek.py --help
```

Read [VALIDATION](VALIDATION.md) for what this release actually checked, and retain your new reports instead of treating the archived timings as a performance guarantee.
