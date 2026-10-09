# Full-model portable release replay

All **42 requests passed** at exact 8192, 16384 and 32768 prompt tokens, with 512 actual generated tokens each and zero prefix reuse. Ten complete warmups are excluded from the 32 measured requests / 16,384 measured generated tokens. Complete output and every non-timing DSpark counter matched the baseline warmup; the three exact 8K cases also matched the archived golden. The driver restored its initial prefill control and checked unchanged process/source/proof identity.

This run used the freshly prepared portable runtime and published model-stage layout. A background Hub upload and other host workload were active. Its timings are retained as qualification evidence and do not replace the published historical headline numbers.

Each context directory includes its sanitized report, exact request payloads, raw SSE, parsed event timings and complete output JSON for every request. Full server logs are omitted; their parsed counters and module-path counters remain in the reports. Local paths in reports/output JSON are replaced by portable placeholders or local artifact filenames. No auth keys are included.

`manifest.json` records the original private report hashes and the source/released SHA256 of every evidence file. Request/SSE/event files retain exact bytes; report/output JSON hashes differ after path sanitization. The runtime/benchmark source hashes recorded by the run are preserved. The later startup-connection retry correction changes the current driver file hash but does not alter those completed request protocols or results.
