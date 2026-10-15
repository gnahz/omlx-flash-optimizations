# 完整复现说明

[English](REPRODUCE.md) · [测试方法与证据](RESULTS.zh-CN.md) · [发布验收](VALIDATION.zh-CN.md)

## 1. 环境与固定版本

公布测试的设备为 Apple M5 Ultra，30 CPU 核、64 GPU 核、256 GiB 统一内存，macOS 27.0.1。使用 arm64 macOS 与完整 Xcode 工具链；发布编译检查使用 SDK 27.2。Qwen NAX 内核至少需要 SDK 26.2；DeepSeek JIT 编译部署目标为 27.0。较小内存设备不属于这组全部专家常驻配置。

准备 Git、`uv` 与 Hugging Face `hf` CLI。固定依赖环境使用 `huggingface_hub==1.33.0`。确认 `hf download --help` 与 `hf cache verify --help` 可用。公开权重下载无需 token。DeepSeek 下载总量为 **420.53 GiB**，包含保留的原始分片、Engram 与元数据文件；另预留编译、下载元信息和前缀缓存空间。

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

最后一项明确设置 240 GiB wired 内存分配容量。DeepSeek 启动器检查该值与物理内存，但不会自行更改系统设置。240 GiB 进程预算把 16 GiB 留在预算之外，并不保证任何时刻都有 16 GiB 实时空闲内存。一次只加载一个完整模型。

| 组件 | 固定版本 |
|---|---|
| DeepSeek oMLX | `0.7.0.dev4`，`d4298ad1811bdd1b565660263380b52fc4ff71f0` |
| DeepSeek MLX | `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`，另加有序 MLX 补丁 |
| DeepSeek Python | 3.13.15，CPython 3.13 ABI |
| Qwen oMLX | v0.7.0，`4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40` |
| Qwen MLX / Metal / Python | 0.32.2 / 0.32.2 / 3.12.14 |
| DeepSeek 权重 | `profiles/release.json` 记录的已校验 Hub revision |
| Qwen 权重 | `mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp@e171af86f499f1855b0fb71d781105e8dd609610` |

完整源代码与依赖固定清单位于 `profiles/deepseek/`、`profiles/qwen/`。两个模型分别使用独立源码与 Python 环境，不对任意新版 oMLX 直接套用。

## 2. 内核补丁与引擎补丁

```bash
python3 scripts/repro.py prepare --model deepseek --work-dir "$DS_WORK"
python3 scripts/repro.py prepare --model qwen --work-dir "$Q_WORK"
```

准备阶段不运行模型。脚本检查补丁 SHA256，在新目录拉取源码并 checkout 固定提交，依次执行补丁检查和应用，安装完整依赖锁，再编译原生扩展。已有源码目录不会被静默替换。通用命令按需通过 `uv` 使用固定 Python。

DeepSeek 目录结构为 `work/source/{mlx,omlx}`、`work/venv`、`work/build/glm`、`work/assets`。补丁顺序：

1. MLX：`0001-tacos8me-mlx-and-portable-jit.patch`。
2. oMLX：`0001-tacos8me-qualified-baseline.patch`。
3. oMLX：`0002-exact-metadata-and-direct-prefill.patch`。
4. oMLX：`0003-exact-prefix-cache-persistence.patch`。
5. oMLX：`0004-portable-release-identities.patch`。
6. oMLX：`0005-long-context-decode-and-prefill-fairness.patch`。

第二组新增补丁同时包含字典 GEMM 内核与引擎 dispatch；缓存补丁包含保留策略、身份检查与持久化；portable 补丁支持公开权重的相对路径 sidecar。`0005` 增加 prefill 公平调度、终包屏障及其 CPU 回归测试；启动器为 cached 配置启用已有 packed-cache growth 实现。包括 MLX 补丁在内的六份补丁 SHA256 均由 `profiles/deepseek/sources.json` 固定。依赖来源仅引用 [tacos8me 固定版本](https://github.com/tacos8me/m5-ultra/tree/4e4d5a81f8135249815a3f284c0ed5bbd1faee58)。

如需先审查补丁再编译，可用以下流程**代替**该工作目录的通用 `prepare`：

```bash
python3 scripts/prepare_deepseek.py --work-dir "$DS_WORK" --apply-only
python3 scripts/prepare_deepseek.py --work-dir "$DS_WORK" --check
python3 scripts/prepare_deepseek.py --work-dir "$DS_WORK" --build-only --python "$(uv python find 3.13.15)"
```

`--mlx-source` / `--omlx-source` 可把本地 clone 用作只读 Git 对象来源，补丁仍应用到新的 checkout。`source-receipt.json` 固定全部补丁后源码；`build-receipt.json` 记录二进制与构建输入。

**从 `0004` 升级时，必须选择新工作目录，重新准备完整 MLX + oMLX `0001`–`0005` 序列。** 仅替换 `serve_deepseek.py` 并复用旧 prepared 源码不会安装这些 runtime 修复。GET 会如实报告缺失/未启用的功能，但不作为启动拒绝门。更新后的 cached 模型就绪后，核对 `decode_runtime.packed_cache_growth=true`、`decode_runtime.packed_cache_growth_pingpong=true`，以及该 scheduler 的 `preserve_prefill_chunks=true`、`preserved_prefill_max_defer_seconds=8.0`、`terminal_response_barrier=true`；仅有目标 `profile` 字段不足以证明已加载源码实现了这些功能。

DeepSeek 使用 CMake 4.1.0、setuptools 84.0.0、wheel 0.48.0、C++20，以及 `MLX_METAL_JIT=ON`、`CMAKE_OSX_DEPLOYMENT_TARGET=27.0`、`BASELINE_RUNTIME_METAL=ON`。扩展使用与该 MLX 构建相同的 fetched **nanobind 3.0.1** CMake 包，不用另一版本的 Python nanobind 替换。Metal 源码由指定 Python 展开，通过经过验证的 safe-math JIT 路径加载。所有编译路径从工作目录派生。

脚本从[官方 mlx-metal 0.32.2 wheel](https://files.pythonhosted.org/packages/dd/cd/4e50bf325100e7165e13d025f264362bf0009196269f9eaf87f2c6e738a2/mlx_metal-0.32.2-py3-none-macosx_26_0_arm64.whl)提取 stock 辅助 `mlx.metallib`，不在 Git 中发布本机二进制。wheel SHA256 为 `e6abeac9ac5265830c9c1541b6f96e9be37a85c2446763a46ad466c63a3837ab`；提取的 182,351,120 字节文件 SHA256 为 `dc59d1cceb1a5c7e578232e6e41e28e2c73c9463ac6dbc3886c3ee17ffc270ed`。两者都必须校验通过；辅助库不能替代带补丁的 MLX runtime。

Qwen 分别创建 `work/{baseline,optimized}/source` 与各源码目录中的 `.venv`。baseline 应用前四个补丁；optimized 按 `profiles/qwen/manifest.json` 应用全部六个：`00` 隔离服务 runtime、社区 QSA 容量、社区 prompt lookup、lookup 策略安全、公共前缀尾块保留、持久的 scheduler 集成。共同的隔离补丁把启动 shim/reaper 状态放进指定配置目录。两套原生扩展分别编译。需要指定变体时使用 `scripts/prepare_qwen.py --workdir ... --variant baseline|optimized|both --python ...`；`--patch-only` 只准备补丁后源码。来源说明见 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)。

## 3. 下载并完整校验权重

```bash
python3 scripts/repro.py download --model deepseek --model-dir "$DS_MODEL"
python3 scripts/repro.py verify --model deepseek --model-dir "$DS_MODEL"
python3 scripts/repro.py download --model qwen --model-dir "$Q_MODEL"
python3 scripts/repro.py verify --model qwen --model-dir "$Q_MODEL"
```

DeepSeek 下载固定为 `profiles/release.json` 的已验证 revision，不跟随移动的 main；校验读取 `SHA256SUMS` 中的每个文件。`--revision` 可用于审查其他版本，但不属于当前固定复现。保留完整模型目录，包括发布版 config/index、tokenizer/template、原始张量分片、SSD Engram、LICENSE、`ds41-conversion.json` 与 `exact-metadata/` 中的 120 个字典投影、sidecar、proof。发布版 JSON 增加标准 MLX 量化描述及 Engram 索引，使 Hub 正确统计参数；sidecar 绑定这两个 JSON 的准确哈希。服务状态写入工作目录，不写进模型目录。

Qwen 直接下载固定上游权重，使用 `hf cache verify ... --fail-on-missing-files` 完整校验；启动器另检查固定 config/index checksum 和全部索引分片。该 Git 仓库不重新分发 Qwen 权重。量化来源、发布结构及下载量/常驻内存的区别见 [DeepSeek 模型卡](HF-MODEL-CARD.md)。

## 4. 复现 DeepSeek 无前缀复用 A/B

终端 1：

```bash
"$DS_WORK/venv/bin/python" scripts/repro.py serve --model deepseek --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --profile benchmark --port 8011
```

终端 2，从相同仓库路径运行，并设置相同目录变量：

```bash
"$DS_WORK/venv/bin/python" scripts/repro.py benchmark --model deepseek --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --profile benchmark --output-dir "$DS_WORK/results/cold-1" --port 8011
```

服务仅绑定 `127.0.0.1`，模型别名为 `deepseek-v41-lsq`。每次启动生成新 key，写入 `work/config/benchmark/api-key.txt`，权限 0600；驱动读取该文件并仅发送给 localhost。驱动持有 `benchmark.lock`，只连接已运行的发布服务，不启停别的服务。服务持有 `service.lock`，独立 model registry、配置、日志、缓存和 cluster shim/reaper 状态均在工作目录内；继承的 OMLX/DS41 等引擎配置和 Python 路径环境变量在导入前清理。

benchmark 配置：关闭 prefix cache，并发 1，prefill chunk 8192，进程预算 240 GiB；路由专家、DSpark、vision 常驻，仅 Engram SSD offload。native decode 开，native verify 关，sparse/MHC/gather 开，gate/up pairing **关**，原生 metadata prefill **开**，async DSpark **开**。MTP 草拟 4 token；请求温度 0、思考关闭。A 仅关闭 direct metadata prefill，使用展开元数据的 dense bulk GEMM；B 开启 direct。同一压缩权重、同一进程、相同 decode。请求尚未结束、模型加载、engine lease 或 scheduler 工作期间拒绝切换控制项。

历史 prefill 协议保持 packed-cache growth **关闭**、`preserved_prefill_max_defer_seconds` 为 **0**；ping-pong 开关保持开启，但在 growth 关闭时不生效。终包屏障在两个发布配置中均默认开启。GET `/api/reproduction` 从实际 runtime 的 `decode_runtime` 返回 growth 状态，并逐 scheduler 返回 defer、capability 与终包屏障状态。POST 仍只有 benchmark 配置下原有的鉴权、loopback、idle-only metadata prefill/计数重置控制，不新增 growth/调度切换，也不向 cached 配置开放。

| Context | 场景 | 排除的完整预热 | 实测顺序 | 实测请求 |
|---|---|---:|---|---:|
| 8192 | 0、1、2 | 每场景 A/B，共 6 | 每场景两次 ABBA | 24 |
| 16384 | 0 | A/B，共 2 | 一次 ABBA | 4 |
| 32768 | 0 | A/B，共 2 | 一次 ABBA | 4 |

每次实际生成 512 token；共 42 请求，十次预热不计入汇总，32 次实测生成 16,384 token。8K 请求使用存档原始字节；16K/32K 保留场景 0 的完整任务尾部，用固定 CPU formatter/tokenizer 确定性扩展合成历史文本，必须达到精确 token 数。长上下文输出对照该长度的 A 预热，不套用 8K 输出参考。

只检查 CPU 请求构建、或只跑 8K：

```bash
"$DS_WORK/venv/bin/python" scripts/benchmark_deepseek.py --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --output-dir "$DS_WORK/results/payload-check-1" --prepare-only
"$DS_WORK/venv/bin/python" scripts/benchmark_deepseek.py --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --output-dir "$DS_WORK/results/8k-1" --context 8192
```

验收要求：完整 SSE 结束；精确 input/output 用量；明确 `cached_tokens=0`；完整文本/思考/结束原因一致；所有非计时 DSpark 计数与 A 预热一致，8K 还与存档参考一致。A 必须 dense bulk 调用大于零且 direct 为零；B 必须 direct bulk 大于零且 dense bulk 为零。dense 尾部回退允许且独立计数。检查真实 gate/up off、async on、进程 ID 与源码/proof 哈希不变；finally 恢复初始 prefill 开关。失败仍保存请求、原始 SSE、事件、完整输出、日志片段和报告。必需的 native symbol 缺失则直接中止。

首请求和各 arm 的 JIT 预热不计入性能。这里的“冷”仅指没有前缀复用，不表示清除 OS/Engram 文件缓存，也不计模型冷启动加载。发布 HTTP 驱动不独立采样物理内存峰值，历史采样范围见[测试说明](RESULTS.zh-CN.md)。结果目录必须是新目录。

### 2026-10-15 更新的复现范围

独立的[decode 验收存档](../results/qualification/deepseek/decode-20261015/README.md)记录原生产部署真实模型的 1M growth 单项、热 A/冷 B 并发 ABBA 等测量。上方命令保留历史 metadata-prefill 协议，不复刻该诊断控制 harness。本次不新增 cached 配置的修改端点，也不声称已在重定位后的公开 checkout 再次运行这些完整模型测试。请分别查阅[证据清单](../results/qualification/deepseek/decode-20261015/manifest.json)、[测量汇总](../results/qualification/deepseek/decode-20261015/summary.json)与[发布验收](VALIDATION.zh-CN.md)。源码审查和补丁应用检查无需下载模型或推理；可在新工作目录使用第 2 节的 `--apply-only` 与 `--check`。

## 5. Qwen 核心吞吐与前缀回放

先用 Ctrl-C 结束 DeepSeek 发布服务并等待完全退出，再加载 Qwen。Qwen 驱动自行管理所属的临时独立子服务：

```bash
python3 scripts/repro.py benchmark --model qwen --work-dir "$Q_WORK" --model-dir "$Q_MODEL" --suite all --output-dir "$Q_WORK/results/replay-1" --port 8012
```

`--suite core` 回放固定 2K/8K/32K 吞吐请求；`--suite tasks` 回放存档 `edit_file` 与 JSON 修改的实测请求并校验完整输出约束；`--suite prefix` 比较冷/热前缀，并包含独立语义正确性检查；`--suite all` 包含三组。核心/任务回放使用贪心、MTP 深度 3，分别关闭/开启 lookup。lookup 参数为 n-gram 8、最大草稿 7、长匹配阈值 32、长草稿 14、exact row 最大 4。核心服务并发 1，每请求前明确清 cache，并检查实际 `cached_tokens=0`。prefix 对比前四补丁 baseline 与全部六补丁 optimized，缓存收益单独记录。报告保留请求/SSE、完整输出、验收、实际 token、计时和环境/源码身份。

core 为 36 请求 / 30 实测；tasks 为 24 / 20，每任务每模式一次预热加五次实测；prefix 为 51 / 36；all 共 **111 请求 / 86 实测**。任务验收检查存档完整输出 SHA256、实际 token 长度、stop 结束、没有思考/tool call、Python AST 仅改变指定函数名，以及 JSON 除指定字段外完全一致。该发布驱动不执行生成的 Python。

每个新 job 启动前检查五组 native kernel、NAX 硬件可用性与已编译 NAX QMM。bulk 路径门槛为 2048 prompt token，存档 1344/880-token 任务正常走短输入路径；达到门槛的 core/prefix job 仍必须观察到 bulk NAX dispatch。冷测清 cache 前等待 save/persisted 计数稳定、GDN staging 为零且没有临时文件，避免删除仍在异步写入的文件。native/model 就绪、cache 排空/持久化及源码/二进制身份检查均在请求计时外。

只复现任务结果：

```bash
python3 scripts/repro.py benchmark --model qwen --work-dir "$Q_WORK" --model-dir "$Q_MODEL" --suite tasks --output-dir "$Q_WORK/results/tasks-1" --port 8012
```

手动服务命令：

```bash
python3 scripts/repro.py serve --model qwen --work-dir "$Q_WORK" --model-dir "$Q_MODEL" --profile benchmark --port 8012
# 用 Ctrl-C 结束该发布服务后再运行下一行。
python3 scripts/repro.py serve --model qwen --work-dir "$Q_WORK" --model-dir "$Q_MODEL" --profile cached --port 8012
```

benchmark 选择 baseline/fixed3/并发 1，常规上游前缀 cache 仍可用；要复现“零复用”吞吐需用上面的明确清 cache 测试程序。cached 选择 optimized/fixed3lookup/并发 8、8192-token block、公共前缀保留与稳定 `work/runs/cached`。Qwen 进程 guard 为 232 GiB，PLE 全常驻，expert/A8/TurboQuant 关闭，独立 SSD cache 50 GB、hot cache 为零。复用 cached runtime 时校验完整 source package 与模型 config/index 身份。每次启动默认打印临时 API key；也可明确设置 `OMLX_API_KEY` 作为自选持久 key。服务配置目录独立且绑定 localhost。

## 6. DeepSeek 缓存与重启验收

这是单独的验收配置，不混入前面的无前缀复用性能。结束 benchmark 服务后运行：

```bash
"$DS_WORK/venv/bin/python" scripts/repro.py serve --model deepseek --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --profile cached --port 8011
```

另一终端：

```bash
"$DS_WORK/venv/bin/python" scripts/repro.py benchmark --model deepseek --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --profile cached --output-dir "$DS_WORK/results/prefix-1" --port 8011
```

cached 保持 metadata prefill 开、gate/up 关、async DSpark 开和 chunk 8192，允许八请求、精确前缀容量 3 GiB。导入模型前显式设定 `DS41_GROWTH=1`、`DS41_GROWTH_PINGPONG=1`，启用已有 growth 和双缓冲；`preserved_prefill_max_defer_seconds=8.0`，终包屏障开启。启动器清理继承的 DS41 环境，因此外部 export 不替代这些 profile 设置。

持久化使用新 namespace `work/cache/exact-prefix-v2-decode-fixes`，仅启动时从 SSD 读取，空闲状态每 300 秒保存。启动允许 900 秒、有序关闭允许 120 秒。保存使用原子世代切换，保留上一个有效世代直到新世代完成；restore 身份绑定模型配置、tokenizer/config/index、补丁后 runtime、metadata proof、draft signature，以及存储和调度策略。新 namespace 隔离此前策略的快照，不绕过原有持久化身份校验。

8 秒是保留 decode 状态的冷 prefill 前新增的 decode 优先窗口，不是绝对 TTFT 上限；完整 8192-token prefill forward 无法抢占，等待的冷请求可能增加 TTFT。GET 反映实际已加载功能，cached 的 POST 控制仍不开放。具体效果和边界见[独立结果章节](RESULTS.zh-CN.md#2026-10-15长上下文-decode-与并发-prefill)。

探针构建精确 8242-token 的场景 0，生成 64 token：先 prime、再 warm，然后发八份并行相同请求。所有完整输出必须与串行一致，warm/并行必须复用精确 8192 token；报告记录观察到的 scheduler 并发。它只验证有限短输出的前缀行为，不代表八个同时 32K 或长输出的容量。首次请求可能使用已恢复 cache，不标为冷请求。

验证持久化：Ctrl-C 该发布服务并等退出，在 `work/logs/cache-shutdown.json` 确認 `saved: true`，然后用相同 cached 命令重启。把探针结果写入新的 `prefix-after-restart-1` 目录，检查初始 `cache.models` 中的恢复/持久化计数，以及首请求的复用边界。探针自身记录 `restart_assessed: false`；关闭 receipt、重启初始快照和请求复用三者的对照才是重启验收。

## 故障与 CPU 检查

native symbol 缺失时保留日志/receipt 并检查固定版本、SDK，使用新工作目录重做；输出或 DSpark identity 失败时保留失败报告，先核对权重、flags、贪心请求及配置，不能只引用失败请求的速度。服务启动中的连接失败在 900 秒就绪期限内重试；auth/HTTP 状态错误立即失败，需检查 preload、所选 profile/port/work 目录与新 key。已有结果目录拒绝覆盖；Qwen cached 源码/模型身份改变时须使用新 runtime 目录。

```bash
python3 -m unittest discover -s tests -p 'test_benchmark_deepseek.py'
python3 scripts/serve_deepseek.py --help
python3 scripts/benchmark_deepseek.py --help
```

实际发布验收见 [VALIDATION](VALIDATION.md)，保存本机新报告，不把存档数字视为任何设备的性能保证。
