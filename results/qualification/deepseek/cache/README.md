# Cached serving and orderly restart acceptance

Both ten-request probes passed at 8242 prompt / 64 generated tokens: prime, warm, and eight parallel copies. All complete outputs matched their serial reference, warm/parallel requests reused exactly 8192 tokens, and both probes reached eight observed running requests.

The first service started with an empty cache. It saved one checkpoint on orderly shutdown. The second service used a different process ID, restored that same saved generation with zero restore failures, read cache SSD state twice at startup, and reused 8192 tokens on its first request. The acceptance record links the shutdown receipt and both probes. Each individual probe retains `restart_assessed: false`; the separate `restart-acceptance.json` records the cross-process verification.

The cache contained one 10,420,408-byte checkpoint under the configured 3 GiB cap. This checks bounded short-output concurrency and persistence, not full-cache saturation, eight simultaneous maximum-length contexts, sustained throughput or physical-memory peaks. Background Hub upload and host workload were active; new timings are not substituted for the published headline results.

Raw request/SSE/event files retain exact bytes. Reports and complete output JSON use portable paths; original and released hashes are recorded in `manifest.json`. Full server logs and auth keys are excluded.
