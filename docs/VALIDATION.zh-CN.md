# 发布验收

[English](VALIDATION.md)

历史性能数据位于 `results/published/`；本页只记录公开复现包已实际完成的检查，不把未运行的工作流标为已验证。

- DeepSeek 驱动/启动器 **17 项 CPU 测试通过**：完整 SSE 与失败证据、固定样本 checksum、精确输入构建、ABBA/预热计数、bulk/tail 路径区分、完整输出与草稿计数身份、localhost 鉴权、活跃请求控制保护、独立缓存配置、每次新 key、清理继承的引擎/auth 环境、cluster shim/reaper 状态限定于工作目录，以及有期限的启动连接重试且 auth/HTTP 错误立即失败。两个 `--help` 均不导入 MLX。
- 使用真实固定 CPU tokenizer/formatter，三个存档请求均得到精确 8192 token；场景 0 确定性构建的 16384、32768、8242 token 输入也全部验证，无模型或 HTTP 请求。
- DeepSeek 实际权重切片 GPU 组件检查：**1,218 case 与 40 control 全通过，BF16 mismatch 为零**，覆盖第 0/19/39 层的选定专家、元数据解码与字典 prefill，并对照原始 NAX oracle。峰值 MLX 分配为 301,542,446 bytes；仅加载选定切片，不加载完整模型、不测模型吞吐。证据为 [`actual-metadata-prefill.json`](../results/qualification/deepseek/actual-metadata-prefill.json)、[`validator-provenance.json`](../results/qualification/deepseek/validator-provenance.json)。
- DeepSeek 完整模型回放：**42 次请求全部通过**，覆盖精确 8K 三场景与 16K/32K 场景 0。十次完整预热不计入性能，32 次实测生成 16,384 token；零前缀复用，完整输出及非计时 DSpark 计数一致，A dense/B direct bulk 路径正确，进程/源码/proof 身份不变且恢复 prefill 开关。完整请求/SSE/输出和原始 report 哈希位于 [`full-core`](../results/qualification/deepseek/full-core/manifest.json)。本次有后台 Hub 上传与其他主机负载，新计时仅作复验记录，不替代存档 headline 速度；HTTP 驱动未独立采样 footprint。
- DeepSeek 缓存/重启验收：**两组各十次请求全部通过**，8242 输入 / 64 输出、完整输出一致、warm/并行精确复用 8192 token，两次均观察到八个 running 请求。有序关闭保存一个 checkpoint；不同进程恢复相同世代，restore failure 为零，首请求复用 8192 token；启动时 cache SSD 读取两次。脱敏证据为 [`cache manifest`](../results/qualification/deepseek/cache/manifest.json)、[`重启验收`](../results/qualification/deepseek/cache/restart-acceptance.json)、[`关闭 receipt`](../results/qualification/deepseek/cache/cache-shutdown-first.json)。不证明满缓存、八个最大长度请求容量、持续吞吐或采样 footprint。
- Qwen 的 baseline/optimized 独立原生编译、pip check、原始 **62 项 cache** 与 **14 项 policy** 测试，以及 **49 项发布 CPU 测试 / 24 subtest** 通过。六补丁完整版本与四补丁 baseline 均可从固定源版本应用。共同的启动隔离补丁在已编译 editable source 上追加，二进制/依赖未改变；检查了 native kernel availability、policy 迁移、源码/二进制身份及 24 个存档任务输出约束。详见 `profiles/qwen/validation.json`。最后全发布包 CPU 检查为 **79 项测试 / 24 subtest 通过**。
- Qwen 完整模型协议：三次运行覆盖 **111 次通过验收的请求 / 86 次实测**，分别为已完成 core r2 36 次、通过 tasks r3 24 次和通过 prefix r3 51 次。core 精确 2K/8K/32K 输入、512 输出且零复用；tasks 完整输出与存档一致，并通过 AST/JSON 约束。缓存对比及七个完整 JSON 语义回放均通过；optimized 公共前缀 C/D/E 复用 2048 token，最后两个语义 warm 请求复用 2126 token。完整请求/输出/SSE、实际指标、身份及原始哈希见 [`protocol manifest`](../results/qualification/qwen/protocol-manifest.json)。r2 的**整次运行仍为 failed**：core 已完成后，短任务触发不适用的 bulk NAX 日志断言；旧版清 cache 还有 31 条 SSD ENOENT，故不声称 r2 SSD 验收无错误。四个新 r3 job 使用计时外 native preflight 与稳定 writer 排空，全部通过且日志无 ERROR/ENOENT；源码/二进制身份检查也在计时外。1344/880-token 任务低于 2048-token bulk 门槛，正常走短输入路径；合格长度的 core/prefix 仍要求观察到 NAX dispatch。详见[证据说明](../results/qualification/qwen/README.md)。
- Qwen 上游权重：固定 revision `e171af86f499f1855b0fb71d781105e8dd609610`，`hf cache verify --fail-on-missing-files` **42 文件验证通过**，无额外模型 payload。详见 [`upstream-weight-verification.json`](../results/qualification/qwen/upstream-weight-verification.json)。

- DeepSeek Hub 参数元数据：只读文件头得到完整 checkpoint **763,205,315,794 个逻辑参数**，包括 Engram、MTP/DSpark 和视觉组件。固定官方 Hub 计数器、CPU 加载形状检查及真实 native 元数据 header admission 均通过。标准量化描述和四条真实 Engram 索引保留全部原始映射与自定义加载设置，sidecar 绑定发布版 JSON。实际 Hub API 与公开页面均确认 **763B**；250 个张量文件哈希全部不变。九项打包测试覆盖原件保持、packing/元数据排除、辅助副本去重、再次打包与重定位，已计入上述全包 CPU 检查。本次元数据更新未重跑模型推理；既有 config/index fingerprint 会使旧前缀快照失效。详见[参数计数证据](../results/qualification/deepseek/parameter-count.json)。

- DeepSeek Hub 发布：公开版本 `9162be9dcd8763395ee1e19977e30abd28b18951` 的 **262 个发布文件全部验证通过**，总计 451,545,062,577 字节。250 个大文件的远端 LFS SHA256 和尺寸与完整本地内容哈希一致；12 个普通文件下载后逐一核验 SHA256。远端文件集合与发布清单加 Hub 管理的 `.gitattributes` 完全一致；这不代表另行完整下载了 420.53 GiB。详见[逐文件校验记录](../results/qualification/deepseek/hub-release-verification.json)和[固定发布版本](../profiles/release.json)。

新模型回放期间有后台 Hub 上传和其他主机负载，计时单独保存，不替代历史速度。DeepSeek 下载命令要求 `profiles/release.json` 包含验证后的固定版本，未发布/未验证时拒绝默认下载。

这些验收不证明 full-logit 等价、原始量化精度、采样间内存峰值或合成工程请求已由代理完成。
