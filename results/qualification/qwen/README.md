# Portable Qwen model qualification

The selected protocol requests passed across three invocations: **111 requests / 86 measurements**. These are new portable-package checks on the published M5 Ultra machine with background Hub upload/host workload; they do not replace the archived headline timings in `results/published/`.

| Export | Original invocation status | Selected accepted requests | Measurements |
| --- | --- | ---: | ---: |
| `core-r2` | **failed overall**, after the completed core jobs | 36 | 30 |
| `tasks-r3` | passed | 24 | 20 |
| `prefix-r3` | passed | 51 | 36 |

The [protocol manifest](protocol-manifest.json) records each original report/file SHA256 alongside its released SHA256. Request, output, result and readiness JSON files retain their original bytes. `output.json` includes complete text, reasoning/tool-call fields, usage, finish state, normalized SSE lines with arrival times, and parsed frames. `sse.txt` is an inspection copy derived from those recorded lines with LF separators; it is not an independently captured network byte stream. The source/native digest map and non-auth launch flags are retained with the model path replaced by `<MODEL_DIR>`. Service configs, API keys, server logs and the incomplete r2 task job are excluded.

All selected request payload and complete output hashes were independently recomputed during export. Every request passed its recorded acceptance checks and reached SSE `[DONE]`. Core requests used exact 2048/8192/32768 inputs, 512 actual output tokens and zero cached tokens. Both task modes reproduced the archived full outputs: `edit_file` at 1344/1278 input/output tokens and JSON change at 880/822, with normal stop, no reasoning/tool calls and their AST/JSON contracts. The prefix suite includes baseline/optimized cache comparisons and seven complete JSON semantic replays; optimized public-prefix variants C/D/E reused 2048 tokens, and the final two semantic warm replays reused 2126 tokens.

The r2 invocation remains failed: its first task job hit an inapplicable assertion requiring a lazy bulk-NAX log message below that path's 2048-token dispatch floor. Both core jobs had already passed request, source/native identity and eligible NAX dispatch checks. Their old asynchronous cold-clear path also logged **31 SSD ENOENT errors**; zero actual prefix reuse and output geometry passed, but this is not a clean SSD persistence acceptance. Fresh r3 task/prefix runs use native availability preflight and wait for stable save counters, GDN staging and temporary files before deleting cache state. Those four jobs passed with **zero ERROR/ENOENT log lines**. Preflight, model readiness, writer drain and source/native identity checks run outside request timers. Bulk NAX logging remains required for eligible core/prefix jobs; 1344/880-token tasks use the normal short routes.

The upstream weight download was independently verified at its fixed revision; see [upstream-weight-verification.json](upstream-weight-verification.json). These request checks do not establish full-logit equivalence, sampled physical-memory peaks or sustained service capacity.

中文：三次运行合计覆盖 **111 请求 / 86 实测**。`core-r2` 的 36 次核心请求全部通过，但原始整次运行保持 failed：随后短任务触发了不适用的 NAX bulk 日志断言，旧版异步清缓存还产生 31 条 SSD ENOENT 错误，不能据此声称 SSD 持久化无错误。`tasks-r3` 的 24 次任务请求与 `prefix-r3` 的 51 次缓存/语义请求全部通过，四个新服务日志无 ERROR/ENOENT。新版本在计时外检查 native/NAX 可用性及 writer 排空；1344/880-token 任务低于 2048-token bulk 门槛，正常走短输入路径。公开内容保留完整请求、输出、SSE 事件、实际 token 与指标，排除私有配置、key 和服务日志；原始与公开文件哈希在 manifest 中分别记录。新计时只作复验，不改写历史 headline。
