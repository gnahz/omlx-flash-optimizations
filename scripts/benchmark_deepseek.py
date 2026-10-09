#!/usr/bin/env python3
"""Replay frozen DeepSeek prefill ABBA or a separate exact-prefix smoke test.

This CPU/HTTP driver attaches only to the isolated release launcher. It never
imports MLX, loads a model, or starts/stops a service. Output directories must
be new; request bytes, SSE frames, output, counters and log slices are retained.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import time
from urllib.parse import urlparse
from serve_deepseek import CONTROL_PATH, MODEL_ID, loopback, sha

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures/deepseek"
ARCHIVE_SUFFIX = "\nEnd of historical archive. The archive is background only. Complete the following new task in full; provide detailed deliverables:\n"

def validate_url(value):
    parsed = urlparse(value)
    if parsed.scheme != "http" or not loopback(parsed.hostname or "") or parsed.path not in ("", "/"):
        raise ValueError("Use a loopback http server URL without a path")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("URL must not contain credentials, query or fragment")
    return value.rstrip("/")

def content_identity(row):
    return dict(content=row.get("content", ""), reasoning=row.get("reasoning", ""),
                finish_reason=row["finish_reason"], prompt_tokens=row["usage"]["prompt_tokens"],
                completion_tokens=row["usage"]["completion_tokens"])

def parse_mtp(log_text):
    matches = re.findall(r"MTP\[[^\]]+\] finish=(\w+) tokens=(\d+) cycles=(\d+) tok/cycle=([\d.]+) accept=(\d+)/(\d+)([^\n]*)", log_text)
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one complete MTP request in the isolated log slice")
    m = matches[0]
    result = dict(finish=m[0], tokens=int(m[1]), cycles=int(m[2]), tokens_per_cycle=float(m[3]),
                  accepted=int(m[4]), drafted=int(m[5]),
                  acceptance_rate=int(m[4]) / int(m[5]) if int(m[5]) else None)
    timing = re.search(r"timing\[backbone=([\d.]+)ms mtp=([\d.]+)ms sample=([\d.]+)ms cache=([\d.]+)ms\]", m[6])
    if timing:
        result["timing_ms"] = dict(zip(("backbone", "mtp_head", "sampling", "cache"), map(float, timing.groups())))
    copied = re.search(r"copy\[cycles=(\d+) accept=(\d+)/(\d+)\]", m[6])
    if copied:
        result["prompt_copy"] = dict(zip(("cycles", "accepted", "drafted"), map(int, copied.groups())))
    return result

def plan(repeat: int, cases: list[int], order: str) -> list[dict]:
    trials = []
    for case in cases:
        for arm in "AB":
            trials.append(dict(scenario=case, arm=arm, warmup=True, round=None, position=None))
        for iteration in range(repeat):
            for position, arm in enumerate(order):
                trials.append(dict(scenario=case, arm=arm, warmup=False, round=iteration, position=position))
    return trials

def encoder(path: Path):
    # Directly load this self-contained CPU formatter; importing its package
    # would initialize MLX and steal GPU ownership from the supervised server.
    spec = importlib.util.spec_from_file_location("_metadata_prefill_cpu_encoding", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the checkpoint's CPU chat formatter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_messages

def long_payload(raw: bytes, index: int, context: int, tokenizer, encode_messages) -> tuple[bytes, dict]:
    payload = json.loads(raw)
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("The protected archive payload must contain one user message")
    original = messages[0]["content"]
    if original.count(ARCHIVE_SUFFIX) != 1:
        raise ValueError("Cannot identify the protected archive/task boundary")
    archive, task = original.split(ARCHIVE_SUFFIX)
    prefix = archive.split("\n", 1)[0] + "\n"
    suffix = ARCHIVE_SUFFIX + task
    corpus = "".join(
        f"Archive {index}-{i:06}: workspace W{i % 31}; budget {1234 + i*19}; owner team {i%13}; historical status closed.\n"
        for i in range(context // 8 + 100)
    )

    def count(candidate):
        value = encode_messages([dict(role="user", content=candidate)], thinking_mode="chat")
        return len(tokenizer.encode(value, add_special_tokens=False).ids)

    if count(prefix + corpus + suffix) < context:
        raise ValueError("Archive source is shorter than the requested context")
    low, high = 0, len(corpus)
    while low < high:
        middle = (low + high + 1) // 2
        if count(prefix + corpus[:middle] + suffix) <= context:
            low = middle
        else:
            high = middle - 1
    padding = ""
    prompt = prefix + corpus[:low] + suffix
    tokens = count(prompt)
    if tokens != context:
        for candidate_padding in (" ", "\n", " x", " .", " 0", "\nX", "\n.", "\n0", "\n\n"):
            candidate = prefix + corpus[:low] + candidate_padding + suffix
            if count(candidate) == context:
                padding, prompt, tokens = candidate_padding, candidate, context
                break
    if tokens != context:
        # Token merges can make character-prefix counts locally non-monotonic.
        # Search a bounded neighborhood; never silently label a near-size prompt
        # as an exact 16K/32K workload.
        for offset in range(1, 257):
            found = False
            for position in (low - offset, low + offset):
                if 0 <= position <= len(corpus):
                    candidate = prefix + corpus[:position] + suffix
                    if count(candidate) == context:
                        low, padding, prompt, tokens, found = position, "", candidate, context, True
                        break
            if found:
                break
    if tokens != context:
        raise ValueError(f"Cannot construct an exact {context}-token workload; got {tokens}")
    payload["messages"][0]["content"] = prompt
    result = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return result, dict(context_target=context, cpu_prompt_tokens=tokens,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        archive_character_count=low, exact_token_padding=padding,
                        task_suffix_sha256=hashlib.sha256(suffix.encode()).hexdigest(),
                        historical_payload_sha256=hashlib.sha256(raw).hexdigest(),
                        original_task_preserved=True, historical_output_golden_applicable=False)

def positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"Missing or invalid {label}")
    return value

def speculative_identity(row):
    counters = row.get("speculative")
    required = {"finish", "tokens", "cycles", "tokens_per_cycle", "accepted", "drafted", "acceptance_rate"}
    if not isinstance(counters, dict) or not required.issubset(counters):
        raise RuntimeError("Missing complete DSpark request counters")
    for key in ("tokens", "cycles", "accepted", "drafted"):
        if type(counters[key]) is not int or counters[key] < 0:
            raise RuntimeError("Invalid DSpark request counter")
    if (counters["cycles"] <= 0 or counters["drafted"] <= 0 or counters["accepted"] > counters["drafted"]
            or counters["tokens"] != row["usage"]["completion_tokens"]
            or counters["finish"] != row["finish_reason"]):
        raise RuntimeError("Inconsistent DSpark request counters")
    # All non-timing evidence is exact, including prompt-copy counters and
    # derived acceptance/tokens-per-cycle values. Only wall timings may differ.
    return {key: value for key, value in counters.items() if key != "timing_ms"}

def validate_result(row, context: int, golden: dict | None, warmup: dict | None):
    if row["usage"].get("prompt_tokens") != context or row["usage"].get("completion_tokens") != 512:
        raise RuntimeError("Actual prompt/output token counts differ from the protected geometry")
    if row.get("reported_cached_tokens") != 0:
        raise RuntimeError("Fresh prefill test requires zero reused/prefix tokens")
    if golden is not None and content_identity(row) != content_identity(golden):
        raise RuntimeError("Complete output/finish/token counts differ from the historical golden")
    if warmup is not None and content_identity(row) != content_identity(warmup):
        raise RuntimeError("Complete output/finish/token counts differ from this baseline warmup")
    counters = speculative_identity(row)
    if golden is not None and counters != speculative_identity(golden):
        raise RuntimeError("Complete DSpark non-timing counters differ from the historical golden")
    if warmup is not None and counters != speculative_identity(warmup):
        raise RuntimeError("Complete DSpark non-timing counters differ from this baseline warmup")
    row["speculative_counter_identity"] = counters
    for name in ("server_decode_tps", "server_prefill_tps", "ttft_seconds"):
        positive(row.get(name), name)
    duration = positive(row["usage"].get("prompt_eval_duration"), "server prompt_eval_duration")
    # Duration is rounded to hundredths by the existing SSE API. Keep both API
    # values and full-resolution client TTFT; neither is labeled pure GPU time.
    row["server_prompt_eval_seconds"] = duration
    row["server_model_ttft_seconds"] = positive(row["usage"].get("time_to_first_token"), "server model TTFT")
    row["server_model_load_seconds"] = row["usage"].get("model_load_duration")

def summarize(rows: list[dict]):
    measured = [row for row in rows if not row["warmup"]]
    scenarios = {}
    for case in sorted({row["case"] for row in measured}):
        arms = {}
        for arm in "AB":
            chosen = [row for row in measured if row["case"] == case and row["arm"] == arm]
            if not chosen:
                raise RuntimeError("Incomplete measured arm")
            arms[arm] = dict(requests=len(chosen), actual_generated_tokens=sum(row["usage"]["completion_tokens"] for row in chosen),
                prefill_geomean_tps=statistics.geometric_mean(row["server_prefill_tps"] for row in chosen),
                prompt_eval_geomean_seconds=statistics.geometric_mean(row["server_prompt_eval_seconds"] for row in chosen),
                client_ttft_geomean_seconds=statistics.geometric_mean(row["ttft_seconds"] for row in chosen),
                server_ttft_geomean_seconds=statistics.geometric_mean(row["server_model_ttft_seconds"] for row in chosen),
                decode_geomean_tps=statistics.geometric_mean(row["server_decode_tps"] for row in chosen),
                end_to_end_geomean_seconds=statistics.geometric_mean(row["elapsed_seconds"] for row in chosen),
                accepted=sum(row["speculative"]["accepted"] for row in chosen),
                drafted=sum(row["speculative"]["drafted"] for row in chosen),
                whole_output_hashes=sorted({row["whole_output_sha256"] for row in chosen}))
        arms["prefill_change_percent"] = 100 * (arms["B"]["prefill_geomean_tps"] / arms["A"]["prefill_geomean_tps"] - 1)
        arms["ttft_change_percent"] = 100 * (arms["B"]["client_ttft_geomean_seconds"] / arms["A"]["client_ttft_geomean_seconds"] - 1)
        arms["decode_change_percent"] = 100 * (arms["B"]["decode_geomean_tps"] / arms["A"]["decode_geomean_tps"] - 1)
        scenarios[case] = arms
    return dict(scenarios=scenarios,
        geometric_prefill_change_percent=100 * (statistics.geometric_mean(v["B"]["prefill_geomean_tps"] / v["A"]["prefill_geomean_tps"] for v in scenarios.values()) - 1),
        geometric_ttft_change_percent=100 * (statistics.geometric_mean(v["B"]["client_ttft_geomean_seconds"] / v["A"]["client_ttft_geomean_seconds"] for v in scenarios.values()) - 1),
        geometric_decode_change_percent=100 * (statistics.geometric_mean(v["B"]["decode_geomean_tps"] / v["A"]["decode_geomean_tps"] for v in scenarios.values()) - 1),
        measured_requests=len(measured), measured_actual_generated_tokens=sum(row["usage"]["completion_tokens"] for row in measured),
        complete_speculative_counters_equal=True,
        warmup_requests_discarded=sum(row["warmup"] for row in rows),
        first_driver_request_separate=next((dict(arm=row["arm"], client_ttft_seconds=row["ttft_seconds"],
            server_model_load_seconds=row["server_model_load_seconds"], decode_tps=row["server_decode_tps"],
            elapsed_seconds=row["elapsed_seconds"]) for row in rows), None),
        timing_note="SSE prompt_eval_duration and prompt_tokens_per_second include the server first-token boundary; this is not a pure GPU prefill kernel timer. Model load is reported separately. Client TTFT includes loopback request overhead.",
        cold_start_note="The first driver request is a discarded warmup. Only claim process-cold loading if initial loaded_weight_inventory is empty and model_load_duration is reported; candidate warmup JIT costs are also retained separately in rows.",
        full_logits_assessed=False, concurrency_assessed=False, agent_task_completion_assessed=False,
        comparison_note="Same compressed checkpoint, same process, same fresh payloads, same decode and DSpark settings; no historical speed is added to the measured gain.")


def load_payloads(context, cases, model, source):
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for item in manifest["files"]:
        if sha(FIXTURES / item["path"]) != item["sha256"]:
            raise RuntimeError("Frozen fixture checksum mismatch: " + item["path"])
    reference = json.loads((FIXTURES / "reference.json").read_text())
    if reference["context_target"] != 8192 or reference["max_output_tokens"] != 512:
        raise RuntimeError("Invalid frozen reference geometry")
    from tokenizers import Tokenizer
    tokenizer_path = model / "tokenizer.json"
    encoder_path = source / "omlx/patches/deepseek_v41/encoding.py"
    tokenizer, encode_messages = Tokenizer.from_file(str(tokenizer_path)), encoder(encoder_path)
    payloads, geometry = {}, {}
    for index in cases:
        raw = (FIXTURES / f"case{index}-8k.json").read_bytes()
        parsed = json.loads(raw)
        if (parsed.get("model") != MODEL_ID or parsed.get("temperature") != 0
                or parsed.get("max_tokens") != 512 or parsed.get("stream") is not True
                or parsed.get("enable_thinking") is not False):
            raise RuntimeError("Frozen request controls changed")
        if context == 8192:
            count = len(tokenizer.encode(encode_messages(parsed["messages"], thinking_mode="chat"),
                                        add_special_tokens=False).ids)
            if count != 8192:
                raise RuntimeError("Pinned formatter/tokenizer does not reproduce exact 8K input")
            prompt_sha = hashlib.sha256(parsed["messages"][0]["content"].encode()).hexdigest()
            if prompt_sha != manifest["files"][index]["prompt_sha256"]:
                raise RuntimeError("Frozen prompt checksum mismatch")
            payloads[index] = raw
            geometry[index] = dict(context_target=context, cpu_prompt_tokens=count,
                prompt_sha256=prompt_sha, historical_payload_sha256=hashlib.sha256(raw).hexdigest(),
                historical_output_golden_applicable=True)
        else:
            payloads[index], geometry[index] = long_payload(raw, index, context, tokenizer, encode_messages)
    return reference, payloads, geometry, {"tokenizer": sha(tokenizer_path), "encoder": sha(encoder_path),
        "fixture_manifest": sha(FIXTURES / "manifest.json"), "script": sha(Path(__file__))}


class SSEAccumulator:
    def __init__(self):
        self.data, self.events = [], []
        self.content, self.reasoning, self.usage = "", "", {}
        self.first, self.finish, self.done = None, None, False
        self.data_at = None

    def line(self, line, elapsed):
        if not line:
            self.flush(elapsed)
        elif line.startswith("data:"):
            if not self.data:
                self.data_at = elapsed
            self.data.append(line[5:].lstrip(" "))

    def flush(self, elapsed):
        if not self.data:
            return
        raw, arrival = "\n".join(self.data), self.data_at
        self.data, self.data_at = [], None
        if raw.strip() == "[DONE]":
            self.done = True
            return
        event = json.loads(raw)
        if not isinstance(event, dict) or event.get("error"):
            raise RuntimeError("Invalid or failed SSE frame")
        self.events.append(dict(elapsed_seconds=arrival, event=event))
        if isinstance(event.get("usage"), dict):
            self.usage = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if not isinstance(content, str) or not isinstance(reasoning, str):
                raise RuntimeError("Invalid SSE text delta")
            if (content or reasoning) and self.first is None:
                self.first = arrival if arrival is not None else elapsed
            self.content += content
            self.reasoning += reasoning
            self.finish = choice.get("finish_reason") or self.finish


def request(client, url, raw, key, stem, server_log):
    raw_path, event_path = Path(str(stem) + "-sse.txt"), Path(str(stem) + "-events.json")
    output_path, log_path = Path(str(stem) + "-output.json"), Path(str(stem) + "-server.log")
    start, accumulator, error = time.perf_counter(), SSEAccumulator(), None
    offset = server_log.stat().st_size
    try:
        with raw_path.open("x") as stream, client.stream("POST", url + "/v1/chat/completions",
                content=raw, headers={"Authorization": "Bearer " + key,
                                     "Content-Type": "application/json"}) as response:
            if response.status_code != 200:
                raise RuntimeError("Inference HTTP " + str(response.status_code))
            for line in response.iter_lines():
                stream.write(line + "\n")
                stream.flush()
                accumulator.line(line, time.perf_counter() - start)
            accumulator.flush(time.perf_counter() - start)
        if not accumulator.done or accumulator.finish is None:
            raise RuntimeError("Incomplete SSE stream")
    except BaseException as exc:
        error = exc
    finally:
        elapsed = time.perf_counter() - start
        usage = accumulator.usage
        row = dict(content=accumulator.content, reasoning=accumulator.reasoning, usage=usage,
            finish_reason=accumulator.finish, done=accumulator.done, elapsed_seconds=elapsed,
            ttft_seconds=accumulator.first, server_decode_tps=usage.get("generation_tokens_per_second"),
            server_prefill_tps=usage.get("prompt_tokens_per_second"),
            reported_cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            log_start_offset=offset, artifacts=dict(raw_sse=str(raw_path), events=str(event_path),
                output=str(output_path), server_log=str(log_path)))
        row["whole_output_sha256"] = hashlib.sha256(json.dumps(dict(content=row["content"],
            reasoning=row["reasoning"]), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if error:
            row["error"] = f"{type(error).__name__}: {error}"
        event_path.write_text(json.dumps(accumulator.events, ensure_ascii=False, indent=2) + "\n")
        output_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    if error:
        raise error
    return row


def control(client, url, key, value=None, check_sources=False):
    headers = {"Authorization": "Bearer " + key}
    path = CONTROL_PATH + ("?verify_sources=1" if check_sources else "")
    response = (client.get(url + path, headers=headers) if value is None else
                client.post(url + CONTROL_PATH, headers=headers, json=value))
    if response.status_code != 200:
        raise RuntimeError(f"Control HTTP {response.status_code}: {response.text[:400]}")
    return response.json()


def wait_idle(client, url, key, timeout=900):
    import httpx
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            state = control(client, url, key)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            # The owned launcher creates its key before Uvicorn binds the port.
            # Retry only startup connection failures within the same deadline.
            time.sleep(.2)
            continue
        if state.get("error"):
            raise RuntimeError("Service initialization failed: " + state["error"])
        if state.get("ready") and not any(state["idle"].values()):
            return state
        time.sleep(.2)
    raise TimeoutError("Owned service did not become ready and idle")


def validate_server(state, profile):
    expected = 1 if profile == "benchmark" else 8
    settings = state.get("profile", {})
    if (settings.get("profile") != profile or settings.get("max_concurrent_requests") != expected
            or settings.get("prefix_cache_enabled") != (profile == "cached")
            or settings.get("prefill_step_size") != 8192 or settings.get("gate_up") is not False
            or settings.get("dspark_async") is not True):
        raise RuntimeError("Service does not match the isolated published profile")
    if not state.get("fingerprint", {}).get("metadata_proof"):
        raise RuntimeError("Compressed metadata proof missing")
    if (state.get("runtime_flags") != {"gate_up": False, "dspark_async": True}
            or state.get("source_unchanged") is not True):
        raise RuntimeError("Native flags or source identity differ from the published profile")


def validate_arm_stats(stats, arm):
    direct, dense = stats.get("metadata_prefill_direct_calls", 0), stats.get("metadata_prefill_dense_calls", 0)
    if arm == "A" and (direct != 0 or dense <= 0):
        raise RuntimeError("Baseline did not exercise dense metadata prefill")
    if arm == "B" and (direct <= 0 or dense != 0):
        raise RuntimeError("Candidate did not exercise direct dictionary bulk prefill")
    # Dense single-token/tail calls remain valid and are recorded separately.
    if stats.get("dspark_single_calls", 0) <= 0 or stats.get("dspark_batch_calls", 0) != 0:
        raise RuntimeError("DSpark async scheduling changed")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def core_context(args, context, client, key, url, log, initial):
    output = args.output_dir / str(context)
    output.mkdir()
    cases, repeat = ([0, 1, 2], 2) if context == 8192 else ([0], 1)
    reference, payloads, geometry, sources = load_payloads(context, cases, args.model_dir,
                                                        args.work_dir / "source/omlx")
    record = dict(status="prepared" if args.prepare_only else "running", context_target=context,
        max_output_tokens=512, repeat=repeat, order="ABBA", plan=plan(repeat, cases, "ABBA"),
        payload_manifest=geometry, source_sha256=sources, rows=[], prefix_cache_enabled=False,
        compressed_resident_weights_identical=True, sampled_memory_assessed=False,
        model_already_loaded_by_service=True)
    for index, raw in payloads.items():
        path = output / f"case{index}-payload.json"
        path.write_bytes(raw)
        geometry[index]["prepared_payload_sha256"] = sha(path)
    report = output / "report.json"
    save(report, record)
    if args.prepare_only:
        return record
    original_control, warms = initial["controls"].copy(), {}
    try:
        for serial, trial in enumerate(record["plan"]):
            before = wait_idle(client, url, key)
            if before["pid"] != initial["pid"] or before["fingerprint"] != initial["fingerprint"]:
                raise RuntimeError("Service process or deployment fingerprint changed")
            value = dict(metadata_prefill=trial["arm"] == "B", reset_stats=True)
            changed = control(client, url, key, value)
            if changed["controls"]["metadata_prefill"] != value["metadata_prefill"]:
                raise RuntimeError("Metadata prefill switch failed")
            stem = output / f"{serial:03d}-{trial['arm']}-{'warmup' if trial['warmup'] else 'measured'}"
            row = request(client, url, payloads[trial["scenario"]], key, stem, log)
            row.update(trial, serial=serial, case=reference["rows"][trial["scenario"]]["case"])
            record["rows"].append(row)
            save(report, record)
            after = wait_idle(client, url, key)
            row["module_stats"] = after["stats"]
            with log.open("rb") as stream:
                stream.seek(row["log_start_offset"])
                text = stream.read().decode(errors="replace")
            Path(row["artifacts"]["server_log"]).write_text(text)
            row["speculative"] = parse_mtp(text)
            golden = reference["rows"][trial["scenario"]] if context == 8192 else None
            validate_result(row, context, golden, warms.get(trial["scenario"]))
            # Missing cached-token reporting cannot certify a cold request.
            if row["reported_cached_tokens"] != 0:
                raise RuntimeError("Expected explicit zero cached tokens")
            validate_arm_stats(row["module_stats"], trial["arm"])
            if trial["warmup"] and trial["arm"] == "A":
                warms[trial["scenario"]] = row
            save(Path(row["artifacts"]["output"]), row)
            save(report, record)
            print(json.dumps(dict(context=context, serial=serial, arm=trial["arm"],
                warmup=trial["warmup"], prefill_tps=row["server_prefill_tps"],
                decode_tps=row["server_decode_tps"], output_matches=True)), flush=True)
        _, _, _, final_sources = load_payloads(context, cases, args.model_dir, args.work_dir / "source/omlx")
        wait_idle(client, url, key)
        final = control(client, url, key, check_sources=True)
        validate_server(final, "benchmark")
        if final_sources != sources or final["pid"] != initial["pid"] or final["fingerprint"] != initial["fingerprint"]:
            raise RuntimeError("Input, driver or deployment changed during measurement")
        record.update(status="passed", server_initial=initial, server_final=final,
                      summary=summarize(record["rows"]))
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            wait_idle(client, url, key, timeout=10)
            restored = control(client, url, key, dict(original_control, reset_stats=False))
            record["controls_restored_to_initial"] = restored["controls"] == original_control
        except Exception as error:
            record["restore_error"] = f"{type(error).__name__}: {error}"
        save(report, record)
    return record


def prefix_probe(args, client, key, url, log, initial):
    output = args.output_dir / "prefix"
    output.mkdir()
    _, payloads, geometry, sources = load_payloads(8242, [0], args.model_dir, args.work_dir / "source/omlx")
    payload = json.loads(payloads[0])
    payload["max_tokens"] = 64
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    (output / "payload.json").write_bytes(raw)
    record = dict(status="prepared" if args.prepare_only else "running", context_target=8242,
        max_output_tokens=64, geometry=geometry[0], source_sha256=sources, rows=[],
        profile="cached", concurrency=8, scope="Warm exact-prefix bounded concurrency smoke; not a cold throughput benchmark",
        sampled_memory_assessed=False, initial_cache_may_be_restored=True, server_initial=initial)
    report = output / "report.json"
    save(report, record)
    if args.prepare_only:
        return record
    try:
        for index, name in enumerate(("prime", "warm")):
            row = request(client, url, raw, key, output / name, log)
            record["rows"].append(row)
            save(report, record)
            if row["usage"].get("prompt_tokens") != 8242 or row["usage"].get("completion_tokens") != 64:
                raise RuntimeError("Cached smoke geometry changed")
            if index and (content_identity(row) != content_identity(record["rows"][0])
                          or row["reported_cached_tokens"] != 8192):
                raise RuntimeError("Warm exact-prefix output or reused boundary differs")
        reference = record["rows"][1]
        observed = []
        import httpx
        def one(index):
            with httpx.Client(timeout=1800, trust_env=False) as parallel_client:
                return request(parallel_client, url, raw, key, output / f"parallel-{index}", log)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(one, index) for index in range(8)]
            while not all(future.done() for future in futures):
                observed.append(control(client, url, key))
                time.sleep(.2)
            rows = [future.result() for future in futures]
        record["parallel_rows"] = rows
        save(report, record)
        for row in rows:
            if content_identity(row) != content_identity(reference) or row["reported_cached_tokens"] != 8192:
                raise RuntimeError("Concurrent cached output or prefix reuse differs")
        wait_idle(client, url, key)
        final = control(client, url, key, check_sources=True)
        validate_server(final, "cached")
        if final["pid"] != initial["pid"] or final["fingerprint"] != initial["fingerprint"]:
            raise RuntimeError("Service identity changed")
        running = [m.get("scheduler", {}).get("max_observed_running", 0)
            for state in observed + [final] for m in state.get("cache", {}).get("models", {}).values()]
        record.update(status="passed", server_final=final, observed_max_running=max(running, default=0),
            warm_ttft_seconds=reference["ttft_seconds"], all_eight_outputs_equal=True,
            long_context_concurrency_capacity_assessed=False, restart_assessed=False)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(report, record)
    return record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True, help="New directory; existing evidence is preserved")
    p.add_argument("--profile", choices=("benchmark", "cached"), default="benchmark")
    p.add_argument("--suite", choices=("core", "prefix", "all"), default="all")
    p.add_argument("--context", type=int, choices=(8192, 16384, 32768))
    p.add_argument("--port", type=int, default=8011)
    p.add_argument("--prepare-only", action="store_true", help="Verify exact payload geometry on CPU; no HTTP or MLX")
    return p


def main():
    args = parser().parse_args()
    args.work_dir, args.model_dir, args.output_dir = (value.expanduser().resolve() for value in
        (args.work_dir, args.model_dir, args.output_dir))
    if (args.profile == "benchmark" and args.suite == "prefix") or (args.profile == "cached" and args.suite == "core"):
        raise ValueError("Core requires benchmark profile; prefix requires cached profile")
    if args.context and args.profile == "cached":
        raise ValueError("Cached smoke uses the fixed 8242-token prompt")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    url = validate_url(f"http://127.0.0.1:{args.port}")
    config = args.work_dir / "config" / args.profile
    log = args.work_dir / "logs" / f"deepseek-{args.profile}.log"
    reports = []
    if args.prepare_only:
        if args.profile == "cached":
            reports.append(prefix_probe(args, None, None, url, log, {}))
        else:
            for context in [args.context] if args.context else (8192, 16384, 32768):
                reports.append(core_context(args, context, None, None, url, log, {}))
    else:
        key = (config / "api-key.txt").read_text().strip()
        if not re.fullmatch("[0-9a-f]{64}", key):
            raise RuntimeError("Invalid service control key")
        import httpx
        with (args.work_dir / "benchmark.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with httpx.Client(timeout=1800, trust_env=False) as client:
                wait_idle(client, url, key)
                initial = control(client, url, key, check_sources=True)
                validate_server(initial, args.profile)
                if args.profile == "cached":
                    reports.append(prefix_probe(args, client, key, url, log, initial))
                else:
                    for context in [args.context] if args.context else (8192, 16384, 32768):
                        reports.append(core_context(args, context, client, key, url, log, initial))
    result = dict(status="prepared" if args.prepare_only else "passed", profile=args.profile,
        contexts=[record["context_target"] for record in reports], gpu_imported=False,
        http_used=not args.prepare_only, model_service_owned_by_driver=False,
        reports=[str(args.output_dir / ("prefix" if args.profile == "cached" else str(record["context_target"])) / "report.json")
                 for record in reports])
    save(args.output_dir / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
