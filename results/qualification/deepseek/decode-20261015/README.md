# DeepSeek decode qualification — 2026-10-15

These are sanitized extracts of the original production-machine qualification, not a full-model rerun of the portable GitHub release. Historical cold 512-token benchmark results remain separate.

| Controlled experiment | Baseline mean | Fixed mean | Result |
| --- | ---: | ---: | ---: |
| 1,040,040-token prompt, 1,032,192 reused, 259 output tokens | 47.485 tok/s | 50.84 tok/s | +7.1% |
| Concurrent A decode (380 output tokens), while B prefills 262,190 tokens | 16.27 tok/s | 54.89 tok/s | 3.37× |
| Concurrent A complete client response, through SSE DONE | 24.030475 s | 7.560659 s | 68.5% shorter |

Both comparisons used ABBA ordering. The 1M experiment changed growth only. The concurrent comparison changed growth, the 8-second decode priority window and the terminal delivery barrier together; it does not isolate their individual contributions. A's maximum output budget was 512, but both fixtures reached EOS earlier. B generated its full 128-token budget. B's reported prefill duration includes queueing.

- [Summary](summary.json), [growth rounds](growth-rounds.json) and [concurrent rounds](concurrency-rounds.json) preserve per-round measurements, control settings, full-output parity and token-ID hashes.
- [Input/output summary](input-output-summary.json) records exact captured input hashes, input sizes, generation parameters and links to three full output artifacts. The large captured source-code inputs and raw token-ID arrays are not released. Full outputs use canonical JSON SHA256; exact file-byte hashes are separately recorded in the manifest.
- [Long decode fairness](long-decode-fairness.json) records B completing an 8192-token chunk while a different 2048-token A was still generating. This is a functionality check, not an additional performance comparison.
- [Original production qualification](production-qualification.json) records the six qualified file hashes, eight-way/restart checks and **791 CPU tests passed**, with **47 existing asynchronous tests not executed**. Those 47 require the unavailable pytest-asyncio plugin and are not counted as passed.
- [Portable release validation](release-validation.json) separately records this publication's clean patch-chain application and CPU checks. No full-model GPU rerun is claimed for the portable release.
- [Manifest](manifest.json) distinguishes original source artifact SHA256 values from the SHA256 values of the sanitized released artifacts. Hashes establish artifact identity, not independent replayability of unreleased inputs.

The concurrent runner retains B's `diagnostic_valid=false` from a single-request probe that does not handle overlapping requests. Its separate concurrent validation passed with one matching completion record, identical input/full output/token-ID hashes and no runner errors; this limitation is explicit in the released rounds.

A single preserved **8192-token forward remains non-preemptible**. The 8-second priority window adds to existing fairness, memory and cross-engine gates; it is not an absolute latency guarantee. Sampled memory stayed under 240 GiB during the original eight-way checks; peaks between samples were not verified. No model weights, cache contents, keys, machine paths, full server logs or aborted runs are included.
