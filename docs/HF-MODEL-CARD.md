---
license: mit
library_name: mlx
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
tags:
  - mlx
  - omlx
  - deepseek
  - lsq
  - 3-bit
  - apple-silicon
---

# DeepSeek V4.1 Flash LSQ3-G128 MLX with exact metadata dictionaries

MLX/oMLX release of the author's existing mixed LSQ3-G128 checkpoint, with a lossless routed-expert metadata overlay. The supported reproduction is the pinned kernel and engine build in [gnahz/omlx-flash-optimizations](https://github.com/gnahz/omlx-flash-optimizations). Use the fixed published Hub revision recorded in that repository's `profiles/release.json`.

## Source and quantization

The original checkpoint source is [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), revision **`2cba9e42aa026125f3ed06c6d98c1db82f7ca027`**. The conversion manifest identifies `lsq` and the mixed policy `q3g128-native8-embedhead8-engramfp8`: routed experts use 3-bit weights with group size 128, selected native/embedding/head components use their preserved 8-bit formats, and Engram keeps the sourced FP8 storage. Original native MTP/DSpark and vision components are retained. Quantization method/source: [tacos8me/m5-ultra](https://github.com/tacos8me/m5-ultra/tree/4e4d5a81f8135249815a3f284c0ed5bbd1faee58), revision `4e4d5a81f8135249815a3f284c0ed5bbd1faee58`.

The original LSQ3 conversion is lossy. The additional dictionary compression is exact relative to that converted checkpoint, with unchanged integer weight codes and unchanged raw BF16 metadata values. Downloading this package needs no quantization or metadata-generation scripts.

## Parameter count

The complete retained checkpoint contains **763,205,315,794 logical parameters (763.205B)**, including Engram, MTP/DSpark and vision. The [pinned upstream model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/README.md) describes the text backbone as **552B**, with an additional **196B Engram** memory; active parameters are **8B/token for prefill and 16B/token for decode**.

| Component | Logical parameters |
|---|---:|
| Text backbone, excluding Engram tables | 551,880,835,184 |
| Engram embedding tables | 196,613,849,600 |
| MTP / DSpark | 14,225,362,530 |
| Vision, projector and image markers | 485,268,480 |

The former **66B** sidebar value counted packed storage elements in resident shards, including quantization metadata, and omitted the SSD Engram tables. It was not the logical model size. The release now exposes the existing per-module MLX quantization descriptors and indexes the existing Engram weight/scale tensors, with an exact total that excludes quantization metadata and duplicate auxiliary projections. No tensor payload or quantization policy changes. See the [header-based count and verification](https://github.com/gnahz/omlx-flash-optimizations/blob/main/results/qualification/deepseek/parameter-count.json).

## Package layout and size

The package downloads **420.53 GiB**. This is larger than the runtime resident tensor footprint because it preserves all original tensor files and SSD Engram tables alongside the compact metadata overlay and checked fallback originals.

```text
config.json
model.safetensors.index.json
batch-*.safetensors             # original converted tensor shards
engram/                        # original Engram weight/scale files named by config
tokenizer.json
tokenizer_config.json
chat_template.jinja
ds41-conversion.json
LICENSE                        # preserved DeepSeek MIT license
exact-metadata/
  *.safetensors                # 120 encoded routed-expert projection dictionaries
  DeepSeek-V4.1-LSQ3-G128-exact-metadata.json
  metadata-proof.json
allowlist.json
SHA256SUMS
```

Original tensor shards and Engram files are not requantized by packaging. The portable sidecar uses relative paths and pins the release config/index identities, including the parameter-count metadata above. The portable proof preserves the original proof hash as provenance, along with the numeric proof and raw-word/readback evidence. Its current file hash differs after path sanitization and is covered by the published `SHA256SUMS`. Preserve the directory structure and verify all checksums before loading.

## What the metadata overlay changes

Each of 120 routed-expert projections deduplicates raw U32 words containing BF16 scale/bias pairs. U16 indices select each exact pair. Metadata storage falls from 16,986,931,200 to 8,503,863,760 bytes, saving **8,483,067,440 bytes / 7.900472 GiB**. All words round-trip and all encoded files pass readback checks. Full resident model tensors decrease **223.481091 → 215.580619 GiB**, excluding KV/cache/workspaces and retaining DSpark/vision. The routed-expert storage overhead decreases from 3.25 to 3.125153 bits/weight; the quantizer remains 3-bit/G128.

The patched runtime consumes dictionaries directly for native decode and qualified bulk prefill. Unsupported/tail work retains a checked dense fallback. Use the patched runtime; ordinary unpatched oMLX will not reproduce the compact resident profile.

## Run

Use an Apple silicon Mac with **256 GiB** unified memory, the qualified arm64 kernel build, and a 240 GiB wired allocation capacity. Only the large Engram tables are SSD-offloaded; routed experts remain resident. The published measurements used M5 Ultra, 30 CPU / 64 GPU cores and macOS 27.0.1.

Install the full Xcode/SDK 27.2 toolchain, Git, `uv` and the `hf` CLI on `PATH` before these commands. The [environment setup](https://github.com/gnahz/omlx-flash-optimizations/blob/main/docs/REPRODUCE.md#1-requirements-and-fixed-sources) installs the pinned Python 3.13.15 and Hub CLI and gives the complete requirements.

```bash
git clone https://github.com/gnahz/omlx-flash-optimizations.git
cd omlx-flash-optimizations
export DS_WORK="$PWD/work/deepseek" DS_MODEL="$PWD/models/deepseek"
python3 scripts/repro.py prepare --model deepseek --work-dir "$DS_WORK"
python3 scripts/repro.py download --model deepseek --model-dir "$DS_MODEL"
python3 scripts/repro.py verify --model deepseek --model-dir "$DS_MODEL"
sudo sysctl iogpu.wired_limit_mb=245760
"$DS_WORK/venv/bin/python" scripts/repro.py serve --model deepseek --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --profile benchmark
```

Use `--profile cached` for the separately qualified eight-request exact-prefix profile. Full build requirements, pinned revisions, authentication, benchmark commands and restart checks are in the [reproduction guide](https://github.com/gnahz/omlx-flash-optimizations/blob/main/docs/REPRODUCE.md).

## Measurements and limits

Archived optimized direct-prefill results (B arm), using greedy generation, thinking disabled, concurrency 1, zero reused prefix tokens and **512 actual output tokens per request**:

| Input / scope | Prefill tokens/s | Decode tokens/s | Client TTFT |
|---|---:|---:|---:|
| 8K, three scenarios | 2,196.22 | **76.07** | 3.774 s |
| 16K, case 0 | 2,203.40 | **94.40** | 7.495 s |
| 32K, case 0 | 2,194.11 | **97.23** | 15.022 s |

All entries are geometric means of measured requests, excluding warmups. The 8K row includes four measurements per scenario across three scenarios; 16K/32K each include two measurements of code-text case 0 only. These different scenario scopes limit comparisons across context lengths.

A same-process dense-versus-direct prefill comparison gives **+4.21% / +3.79% / +3.75%** at 8K / 16K / 32K, with no measured decode uplift in that comparison. These are generation-throughput tests with complete output/counter identity checks, not a full model-quality or agent-workflow evaluation. The 8K cases are synthetic code text, marketing drafts and a video/computer-use plan; the latter does not execute computer operations. The metadata proof establishes exactness of the additional representation, not losslessness of the original LSQ quantizer. See [methods and evidence](https://github.com/gnahz/omlx-flash-optimizations/blob/main/docs/RESULTS.md).

## License

The original DeepSeek MIT license is preserved as `LICENSE`. Kernel/engine additions and third-party notices are in the linked reproduction repository. Qwen weights are not included in this model repository.

## 中文摘要

本仓库发布已有的混合 LSQ3-G128 MLX 权重，并附加 120 个路由专家投影的**精确元数据字典**。原始量化有损；本次压缩对既有权重的 BF16 scale/bias 位模式逐位无损，整数权重编码不变。常驻模型张量从 223.481 降至 215.581 GiB，节省 7.900 GiB。发布包保留原始分片、SSD Engram 表及回退原件，因此下载总量为 420.53 GiB，不等于常驻内存大小。直接下载并校验后，按上述仓库的固定版本编译内核与引擎补丁即可复现，无需重新量化。模型原始 MIT 许可证保留。

上表为优化后 B 组的存档结果，速度与 TTFT 均取剔除预热后的实测几何均值；8K 汇总三个场景，16K/32K 仅场景 0，每请求实际生成 512 token。

完整 checkpoint 的逻辑参数为 **763.205B**，包括官方标称的 **552B 主干、196B Engram**，以及保留的 MTP/DSpark 和视觉组件。此前侧栏 66B 是量化打包后的存储元素数，并遗漏 SSD Engram 表；本次通过标准 MLX 量化描述、真实 Engram 索引及精确总数更正，权重内容与量化策略不变。
