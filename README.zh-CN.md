# oMLX Flash 优化补丁

[English](README.md) · [完整复现说明](docs/REPRODUCE.zh-CN.md) · [测试方法与证据](docs/RESULTS.zh-CN.md)

用于 Apple silicon 的 **DeepSeek V4.1 Flash LSQ3-G128** 与 **Qwen3.8 Flash Next oQ6e-MTP** 内核、引擎补丁。仓库包含我的新增优化、所需依赖补丁、固定请求样本及独立测试程序。两个模型分别从已测试的固定 oMLX 版本打补丁：DeepSeek `0.7.0.dev4`，Qwen `0.7.0`。

## 我的新增工作

- **DeepSeek 量化元数据无损压缩。** 120 个路由专家投影改用 U16 字典索引，字典保存 BF16 scale/bias 对的原始 U32 位模式。所有元数据逐位还原，整数权重编码不变。常驻模型张量从 **223.481 降至 215.581 GiB**，节省 **7.900 GiB**。这是对已有有损 LSQ3 权重的精确元数据压缩，不是新量化算法。
- **直接读取字典的 prefill 内核。** 原生分块 GEMM 在满足条件的大批量 prefill 中直接读压缩格式，避免该路径整体展开元数据；尾部及不支持的形状保留经过检查的 dense 回退。同进程 A/B 测试中，8K / 16K / 32K prefill 提升 **4.21% / 3.79% / 3.75%**，这组比较的 decode 基本不变。
- **引擎集成。** DeepSeek 增加有容量限制的精确前缀保留和重启持久化；Qwen 增加 lookup 策略安全处理、公共前缀尾块保留，并把已验证的启动挂钩落实为引擎与调度器补丁。测试与缓存服务分别使用独立配置。

## 已公布测试结果

本机：**Apple M5 Ultra，30 CPU / 64 GPU 核，256 GiB，macOS 27.0.1**。贪心采样、关闭思考、并发 1、前缀复用为零、实际生成 512 token。DeepSeek 使用几何均值：8K 汇总三个场景，16K/32K 仅场景 0。Qwen 为五次完整请求的中位数，MTP 深度固定为 3、关闭 lookup。

| 模型 / 输入长度 | Prefill token/s | Decode token/s | 客户端 TTFT |
|---|---:|---:|---:|
| DeepSeek，8K | 2,196.22 | **76.07** | 3.774 s |
| DeepSeek，16K | 2,203.40 | **94.40** | 7.495 s |
| DeepSeek，32K | 2,194.11 | **97.23** | 15.022 s |
| Qwen，2K | 2,819.29 | **151.37** | 0.735 s |
| Qwen，8K | 3,885.47 | **136.29** | 2.128 s |
| Qwen，32K | 3,912.49 | **126.32** | 8.440 s |

Qwen lookup 的收益取决于任务：通过完整输出一致性检查的 `edit_file` / JSON 修改回放分别从 **176.38 → 245.63** / **184.10 → 214.51 token/s**；8K 无前缀复用的文本生成从 **136.29 → 134.54 token/s**。这些数值来自我的本机测试。固定文本样本用于测吞吐，不能视为已完成真实代理工作流。详细 A/B、缓存结果与适用范围见[测试说明](docs/RESULTS.zh-CN.md)。

## 复现

准备上述内存容量的 arm64 Mac、Xcode/SDK 27.2，以及 PATH 中的 Git、`uv`、`hf`。DeepSeek 使用 Python 3.13，Qwen 使用 Python 3.12；准备脚本安装固定依赖并从源码编译内核。运行 DeepSeek 前明确设置 wired 内存容量：`sudo sysctl iogpu.wired_limit_mb=245760`（240 GiB）。DeepSeek 下载需要 **420.53 GiB** 磁盘空间，另需编译与缓存空间；发布包同时保留原始张量分片、SSD Engram 表和精确元数据文件。

```bash
git clone https://github.com/gnahz/omlx-flash-optimizations.git
cd omlx-flash-optimizations
export DS_WORK="$PWD/work/deepseek" DS_MODEL="$PWD/models/deepseek"
python3 scripts/repro.py prepare --model deepseek --work-dir "$DS_WORK"
python3 scripts/repro.py download --model deepseek --model-dir "$DS_MODEL"
python3 scripts/repro.py verify --model deepseek --model-dir "$DS_MODEL"
"$DS_WORK/venv/bin/python" scripts/repro.py serve --model deepseek --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --profile benchmark
```

保留这个隔离的 localhost 服务，在另一终端设置相同的 `DS_WORK` / `DS_MODEL`：

```bash
"$DS_WORK/venv/bin/python" scripts/repro.py benchmark --model deepseek --work-dir "$DS_WORK" --model-dir "$DS_MODEL" --profile benchmark --output-dir "$DS_WORK/results/cold-1"
```

测试共 42 次完整请求：8K 的三个场景，以及 16K / 32K 的场景 0；其中十次完整预热不计入性能汇总。保存请求原始字节、SSE、完整输出、实际 token 用量、计时、DSpark 计数与 prefill 路径计数。所有输出必须与基线预热一致，8K 输出还必须与存档参考一致。切换配置或模型前，用 Ctrl-C 结束本仓库启动的服务。

```bash
export Q_WORK="$PWD/work/qwen" Q_MODEL="$PWD/models/qwen"
python3 scripts/repro.py prepare --model qwen --work-dir "$Q_WORK"
python3 scripts/repro.py download --model qwen --model-dir "$Q_MODEL"
python3 scripts/repro.py verify --model qwen --model-dir "$Q_MODEL"
python3 scripts/repro.py benchmark --model qwen --work-dir "$Q_WORK" --model-dir "$Q_MODEL" --suite all --output-dir "$Q_WORK/results/replay-1"
```

Qwen 测试程序自行管理临时的独立服务，回放无前缀复用的核心测试和缓存测试。日常 Qwen 服务使用 `serve --model qwen ... --profile cached`。DeepSeek 缓存验收则先用 `serve --model deepseek ... --profile cached`，再在另一终端运行 `benchmark --model deepseek ... --profile cached`。缓存服务允许八个请求，缓存位于指定工作目录。补丁顺序、编译选项、完整配置、鉴权、重启检查和失败标准见[完整复现说明](docs/REPRODUCE.zh-CN.md)。

## 权重与来源

- DeepSeek：[junwzhang/DeepSeek-V4.1-Flash-LSQ3-G128-MLX](https://huggingface.co/junwzhang/DeepSeek-V4.1-Flash-LSQ3-G128-MLX)，固定版本见[发布清单](profiles/release.json)；[模型说明](docs/HF-MODEL-CARD.md)描述量化过程。复现直接下载权重，无需重新量化。
- Qwen：[mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp/tree/e171af86f499f1855b0fb71d781105e8dd609610)，固定版本 `e171af86f499f1855b0fb71d781105e8dd609610`，使用上游权重。
- DeepSeek 基线：[tacos8me/m5-ultra](https://github.com/tacos8me/m5-ultra/tree/4e4d5a81f8135249815a3f284c0ed5bbd1faee58)，版本 `4e4d5a81f8135249815a3f284c0ed5bbd1faee58`。
- Qwen 基线：[社区方案](https://github.com/sethforprivacy/Qwen3.8-Flash-Next-M5-Ultra-oMLX/tree/050233efcd0a765de9963ab21a01df0723a50530)，版本 `050233efcd0a765de9963ab21a01df0723a50530`，及 [oMLX PR #4030](https://github.com/jundot/omlx/pull/4030)。

仓库新增代码使用 [Apache-2.0](LICENSE)，保留原始版权声明和模型许可证。另见[第三方说明](THIRD_PARTY_NOTICES.md)与[发布验收](docs/VALIDATION.zh-CN.md)。
