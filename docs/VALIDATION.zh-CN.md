# 发布验收

[English](VALIDATION.md)

历史性能数据位于 `results/published/`；本页只记录公开复现包已实际完成的检查，不把未运行的工作流标为已验证。

- DeepSeek 驱动/启动器 **17 项 CPU 测试通过**：完整 SSE 与失败证据、固定样本 checksum、精确输入构建、ABBA/预热计数、bulk/tail 路径区分、完整输出与草稿计数身份、localhost 鉴权、活跃请求控制保护、独立缓存配置、每次新 key、清理继承的引擎/auth 环境、cluster shim/reaper 状态限定于工作目录，以及有期限的启动连接重试且 auth/HTTP 错误立即失败。两个 `--help` 均不导入 MLX。
- 使用真实固定 CPU tokenizer/formatter，三个存档请求均得到精确 8192 token；场景 0 确定性构建的 16384、32768、8242 token 输入也全部验证，无模型或 HTTP 请求。
- DeepSeek 实际权重切片 GPU 组件检查：**1,218 case 与 40 control 全通过，BF16 mismatch 为零**，覆盖第 0/19/39 层的选定专家、元数据解码与字典 prefill，并对照原始 NAX oracle。峰值 MLX 分配为 301,542,446 bytes；仅加载选定切片，不加载完整模型、不测模型吞吐。证据为 [`actual-metadata-prefill.json`](../results/qualification/deepseek/actual-metadata-prefill.json)、[`validator-provenance.json`](../results/qualification/deepseek/validator-provenance.json)。
- DeepSeek 完整模型回放：**42 次请求全部通过**，覆盖精确 8K 三场景与 16K/32K 场景 0。十次完整预热不计入性能，32 次实测生成 16,384 token；零前缀复用，完整输出及非计时 DSpark 计数一致，A dense/B direct bulk 路径正确，进程/源码/proof 身份不变且恢复 prefill 开关。完整请求/SSE/输出和原始 report 哈希位于 [`full-core`](../results/qualification/deepseek/full-core/manifest.json)。本次有后台 Hub 上传与其他主机负载，新计时仅作复验记录，不替代存档 headline 速度；HTTP 驱动未独立采样 footprint。
- Qwen 的 baseline/optimized 独立原生编译、pip check、原始 **62 项 cache** 与 **14 项 policy** 测试，以及 **40 项发布 CPU 测试 / 24 subtest** 通过。六补丁完整版本与四补丁 baseline 均可从固定源版本应用。共同的启动隔离补丁在已编译 editable source 上追加，二进制/依赖未改变；检查了 native kernel availability、policy 迁移、源码/二进制身份及 24 个存档任务输出约束。详见 `profiles/qwen/validation.json`。

其他模型套件、缓存/重启验收与 Hub revision 验证在实际完成后由发布负责人另行记录。历史速度始终指原有已验收 runtime，新复验记录单独保存。下载命令要求 `profiles/release.json` 包含验证后的固定版本，未发布/未验证时拒绝默认下载。

这些验收不证明 full-logit 等价、原始量化精度、采样间内存峰值或合成工程请求已由代理完成。
