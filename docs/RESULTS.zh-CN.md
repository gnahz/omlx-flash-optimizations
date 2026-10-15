# 测试结果与适用范围

[English](RESULTS.md) · [复现](REPRODUCE.zh-CN.md)

这里汇总本人的本机测试，硬件为 Apple M5 Ultra，30 CPU / 64 GPU 核、256 GiB，macOS 27.0.1。2026-10-15 验收在原生产实机上、使用相同模型和 runtime 于隔离端口完成，随后恢复 8011 并回放；较早章节保留 2026-10-07 存档与后续缓存验收。公开包的移植检查单独记在[发布验收](VALIDATION.zh-CN.md)，不替代存档计时。

版本基线：DeepSeek 使用 oMLX `0.7.0.dev4`，Qwen 使用 oMLX `0.7.0`。

## 2026-10-15：长上下文 decode 与并发 prefill

本次验收在原生产实机的隔离服务上运行，使用相同模型和 runtime，固定权重、原生二进制与量化不变，之后恢复生产 8011 并通过回放。`0005-long-context-decode-and-prefill-fairness.patch` 在 `0004` 后移植这些 runtime 改动；发布不表示已从公开 checkout 再跑完整模型。发布准备期间，已恢复的生产 8011 服务继续运行。

| 负载 / 指标 | 基线均值 | 修复后均值 | 变化 |
|---|---:|---:|---:|
| 1M 上下文，API generation token/s | 47.485 | **50.840** | **+7.0654%（7.1%）** |
| 同一 1M 负载，API generation duration | 5.455 s | **5.095 s** | — |
| 64K 热缓存 A 与 262K 冷 B 并发，A API generation token/s | 16.27 | **54.89** | **3.3737×** |
| 同一并发负载，A API generation duration | 23.355 s | **6.920 s** | — |
| 同一并发负载，A 客户端 SSE DONE 时间 | 24.030475 s | **7.560659 s** | **−68.54%** |
| 同一并发负载，B 客户端首个输出用时 | 142.212814 s | 141.839588 s | 描述性数据；本次未观察到增加 |

两组负载分别采用四轮 ABBA、每个 arm 两轮。同一负载的完整发送输入、完整生成输出、生成 token SHA256、结束原因及非计时 usage 在两种设置下完全一致。1M 请求虽设置更高上限，实际均为 **259 token、EOS 结束**；并发 A/B 实际输出分别为 **380/128 token**。相同实际长度与完整身份使对照可比，不混作 512-token 测试。每个 arm 两次观察只支持本机这组比较，不用于宣称总体统计置信度。表格为显示用舍入值，收益按汇总中的原始实测值计算。

**growth 单项对照**的实际 prompt 为 **1,040,040 token，复用 1,032,192 token，记录的剩余 prefill chunk 为 7,847 token**；它测前缀复用后的 decode，不是 1M-token 冷 prefill。仅切换 growth，两种设置均固定 `preserved_prefill_max_defer_seconds=0`、诊断终包屏障关闭；开启 growth 时使用已有 ping-pong 双缓冲，没有引入新 attention 数学或重写 growth 实现。

**并发对照**中 A 为 **65,580 prompt token / 65,536 reused token**，B 为 **262,190 prompt token / 零复用**；仅在 A 第一个真实非空输出到达后才提交 B。该对照同时切换 growth、新增 decode 优先窗口和终包屏障，结果只能归因于三项联合修复，不能拆算各项收益。这些诊断控制设置与公开启动器不同，后者两种 profile 均默认启用终包修复。

调度改动是在下一次保留 decode 状态的冷 prefill 开始前，为已运行的 decode 增加 **8 秒优先窗口**。它不是绝对延迟 SLA，也不是逐 token 的 GPU 抢占：完整 **8192-token forward 仍不可抢占**，此前的调度与内存条件仍然适用。引擎还在终包边界结束 step burst，让已完成请求的 SSE DONE 先到客户端，再开始下一次冷 prefill。推迟 B 可能增加其首 token 排队，但本次 B 的客户端首个输出平均用时未增加；这点小差异只作描述，不另作因果收益。B 的 reported prefill duration 与 TTFT 均含排队，不能当成纯 GPU prefill 时间。[逐轮汇总](../results/qualification/deepseek/decode-20261015/summary.json)保留请求时间。API generation 吞吐和客户端 SSE DONE 用时覆盖不同区间，都不是独立 GPU kernel 计时。

另一次 **实际输出 2048 token** 的运行验证固定策略下等待中的 prefill 最终能够执行；它使用不同 prompt，只作功能性防饥饿覆盖，不纳入同输入性能 A/B。热缓存验收观察到 **八个 running 请求、共同重叠 6.0989 秒**，完整输出/token 一致，并完成有序重启持久化与原生产 8011 恢复后的回放。这组有限验收不证明八个最大上下文请求的容量或持续聚合吞吐。

原部署 **791 项 CPU 测试通过：scheduler 663 + 终包交付 52 + growth 生命周期 44 + launcher/gate 32**。终包 52 项由 14 项新增与 38 项既有同步测试组成。因缺少 `pytest-asyncio`，**47 项既有异步测试未运行**，不声称完整 engine 套件全过。CPU 字节一致性和逻辑 capacity 检查不测 GPU buffer donation 或真实物理访存；端到端速度来自上方独立的真实模型运行。本次修复未改原生二进制、权重张量或量化。

[脱敏测量汇总](../results/qualification/deepseek/decode-20261015/summary.json)与[清单](../results/qualification/deepseek/decode-20261015/manifest.json)记录逐轮指标、输入/输出/token 身份和原证据 SHA。公开包的源码/CPU 移植检查另见[发布验收](VALIDATION.zh-CN.md)，不把原生产验收改称新的公开包 GPU 回放。

## 精确元数据压缩

原始混合 LSQ3-G128 量化有损。本次新增压缩**相对于该固定量化结果无损**：把原始 BF16 scale/bias 对保存为 raw U32 字典词，按投影去重，U16 索引选择精确值。120 个路由专家投影的字典各有 17,885–64,321 个条目。proof 验证所有位模式还原与完整 encoded readback，整数权重编码不变，发布打包不重新量化。

| 范围 | 压缩前 | 压缩后 | 节省 |
|---|---:|---:|---:|
| 路由专家元数据 | 16,986,931,200 bytes | 8,503,863,760 bytes | 8,483,067,440 bytes |
| 含元数据的常驻模型张量 | 223.481091 GiB | 215.580619 GiB | **7.900472 GiB** |
| 含元数据的路由专家存储 | 3.25 bit/weight | 3.125153 bit/weight | — |

常驻统计不含 KV cache 与计算 workspace，保留 DSpark、vision。底层量化仍为 3-bit/G128，3.125153 描述存储开销而非新量化位宽。SSD Engram offload 与专家 offload 不同，路由专家仍常驻。

direct prefill 内核在合格 BF16 输入上，边查字典边解量化，以 FP32 累加进行原生分块 GEMM。验证 dispatch 为 batch 1、至少 512 个 prefill token、top-k 6、对齐的 affine 3-bit/G128 形状。其余形状、尾部与单 token 保留 checked dense 回退及 **405 MiB** 预留。整批 dense 调用为零不等于整次请求完全没有元数据展开。

## 同进程 direct-prefill A/B

A/B 压缩权重、decode 和 async DSpark 相同，gate/up 关闭、chunk 8192、前缀复用关闭、温度 0、思考关闭，每次实际生成 512 token。A 使用展开元数据的 dense bulk GEMM，B 直接消费字典。Prefill、decode 和客户端 TTFT 均为实测请求的几何均值。

| 输入 / 范围 | A prefill t/s | B prefill t/s | Prefill 变化 | A decode t/s | B decode t/s | A 客户端 TTFT | B 客户端 TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8K，三场景几何均值 | 2,107.57 | 2,196.22 | **+4.21%** | 76.10 | **76.07** | 3.931 s | 3.774 s |
| 16K，场景 0 | 2,122.89 | 2,203.40 | **+3.79%** | 94.40 | **94.40** | 7.777 s | 7.495 s |
| 32K，场景 0 | 2,114.76 | 2,194.11 | **+3.75%** | 97.22 | **97.23** | 15.581 s | 15.022 s |

8K decode 几何均值为 **76.10 → 76.07 t/s（-0.045%）**。B 的代码文本、营销文本、视频/电脑操作**规划文本**分别为 91.22、71.59、67.40 t/s。16K/32K 的 94.40/97.23 t/s 只覆盖场景 0，不能直接与三场景 8K 几何均值比较并推断上下文长度收益。

共 42 次完整请求，其中 32 次实测、十次预热，实测生成 16,384 token。8K 每场景先 A/B 预热再两轮 ABBA，每 arm 四次；16K/32K 各仅场景 0、两次预热、一轮 ABBA。完整输出、结束原因、用量及所有非计时 DSpark 计数一致。8K/16K/32K 的 B 分别记录 direct bulk 20/40/80 次、dense bulk 零次、dense tail 20/40/80 次。早期单独优化的收益不相加。

API prefill rate / prompt-eval duration 含服务首 token 边界，不是孤立 GPU 内核计时；客户端 TTFT 含 localhost 请求开销。模型提前常驻、各 arm 完整预热。“冷”只表示零前缀复用，不清 OS/Engram 文件缓存。物理 footprint 每 0.5 秒采样，8K/16K/32K 峰值为 **225.56 / 225.73 / 225.62 GiB**、报告压力正常；采样间峰值未知。240 GiB 预算不保证实时 16 GiB 空闲；发布 HTTP 驱动明确不独立采样 footprint。

## Qwen 本机回放

使用固定上游 `mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp@e171af86f499f1855b0fb71d781105e8dd609610` 与 oMLX v0.7.0。以下每行是五次有效请求的中位数，均零前缀复用、关闭思考、贪心且 MTP 深度 3；两种模式区别仅 lookup。核心输出实际 512 token；任务输出校验完整内容与任务约束。

| 输入 / 任务 | 实际输出 token | fixed3 decode t/s | fixed3 + lookup t/s |
|---|---:|---:|---:|
| 2K | 512 | 151.37 | 143.45 |
| 8K | 512 | 136.29 | 134.54 |
| 32K | 512 | 126.32 | 124.63 |
| `edit_file`，1344 输入 | 1278 | 176.38 | 245.63 |
| JSON 修改，880 输入 | 822 | 184.10 | 214.51 |

lookup 对重复/修改密集输出可能有利，对这组普通文本吞吐则较低，不宣称普遍收益，也不把社区 lookup 算法归为本人创新。我的新增工作为安全策略处理、调度器集成和前缀保留。与早期 oMLX 的顺序历史对比不能解释为同进程因果 A/B；上游模型卡或社区作者的速度不计为本机结果。

存档 Qwen 热前缀请求的 8K/32K fixed3 TTFT 中位数为 **0.111 / 0.229 秒**，lookup 为 **0.113 / 0.229 秒**。它们属于证据中的 `warm` 分组，不混入无前缀复用表。公共前缀尾块优化在独立 baseline/full-patch prefix suite 中复现。

## DeepSeek 缓存验收

历史八并发验收共有十次 8242 输入 / 64 输出请求：串行参考、八个并行缓存流、一次有序重启后回放。八流同时生成重叠 5.637 秒，观察到八个 running 请求，均复用 8192 token，API 输出和生成 token ID 一致。重启恢复一个 checkpoint 并复用同一边界，请求中 cache SSD 读取计数不增加。采样 footprint 峰值 222.45 GiB。缓存上限 3 GiB，仅启动读 SSD，空闲每 300 秒保存。

这只验证有限短输出，没有填满 3 GiB 缓存，不证明八个同时最大上下文或持续吞吐。发布探针报告 warm/parallel 的完整输出一致与复用边界；重启 receipt/初始 restore 快照/首请求复用必须按指南额外对照。

## 发布的证据

[`deepseek.json`](../results/published/deepseek.json)保存元数据 proof、常驻与 final-prefill 汇总；[`deepseek-prefill.csv`](../results/published/deepseek-prefill.csv)包含逐场景 A/B、原始文件名和 SHA256；[`deepseek-cache.json`](../results/published/deepseek-cache.json)为脱敏的历史八并发验收；[`qwen.json`](../results/published/qwen.json)仅保留有效 full512 / validated-task 分组并保留 stage、样本数及实际输出长度。两个模型的固定输入与 checksum 分别在 [`fixtures/deepseek`](../fixtures/deepseek/manifest.json)、[`fixtures/qwen`](../fixtures/qwen/manifest.json)。公开源码迁移后的新验证另存，不能用它重写历史速度。

DeepSeek 合成样本要求生成代码、营销草稿及视频/电脑操作方案。实际 512-token 输出常在长度上限截断，只用于吞吐与一致性检查，不代表可运行工程已完成、已执行真实浏览器或电脑操作、全 logits 一致，亦不代表原始量化精度评估。
