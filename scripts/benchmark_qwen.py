#!/usr/bin/env python3
"""Replay frozen Qwen core, task, and prefix protocols against owned servers."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import re
import secrets
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from serve_qwen import MODEL, ROOT, environment, launch_plan, prepare_runtime, runtime_identity, verify_model

FIXTURES = ROOT / 'fixtures/qwen'

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def load_fixtures():
    manifest = json.loads((FIXTURES / 'manifest.json').read_text())
    fixtures = {}
    for name, expected in manifest['files_sha256'].items():
        p = FIXTURES / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen fixture identity changed: ' + name)
        fixtures[p.stem] = json.loads(p.read_text())
    return fixtures

class SSEAccumulator:
    """Parse SSE frames without mistaking frames, roles, or usage for tokens."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.frames: list[Any] = []
        self.data: list[str] = []
        self.data_arrival_ns: int | None = None
        self.text = ""
        self.reasoning = ""
        self.tool_calls: list[Any] = []
        self.usage: dict[str, Any] | None = None
        self.finish_reason: str | None = None
        self.first_token_ns: int | None = None
        self.done = False
        self.errors: list[str] = []

    def line(self, line: str, relative_ns: int) -> None:
        self.events.append({"relative_ns": relative_ns, "line": line})
        if line == "":
            self.flush(relative_ns)
        elif line.startswith("data:"):
            if not self.data:
                self.data_arrival_ns = relative_ns
            self.data.append(line[5:].lstrip(" "))

    def flush(self, relative_ns: int) -> None:
        if not self.data:
            return
        payload = "\n".join(self.data)
        arrival_ns = self.data_arrival_ns if self.data_arrival_ns is not None else relative_ns
        self.data.clear()
        self.data_arrival_ns = None
        if payload.strip() == "[DONE]":
            self.done = True
            return
        try:
            frame = json.loads(payload)
        except (ValueError, TypeError):
            self.errors.append("malformed_sse_json")
            return
        self.frames.append(frame)
        if not isinstance(frame, dict):
            self.errors.append("non_object_sse_frame")
            return
        if frame.get("error"):
            self.errors.append("server_stream_error")
        if isinstance(frame.get("usage"), dict):
            # Some compatible servers repeat usage; it is a cumulative final total.
            self.usage = frame["usage"]
        for choice in frame.get("choices", []):
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            content = choice.get("text") or delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""
            tools = delta.get("tool_calls") or []
            if not isinstance(content, str) or not isinstance(reasoning, str):
                self.errors.append("non_text_delta")
                continue
            generated = bool(content or reasoning or any(
                t.get("function", {}).get("arguments") for t in tools if isinstance(t, dict)))
            if generated and self.first_token_ns is None:
                self.first_token_ns = arrival_ns
            self.text += content
            self.reasoning += reasoning
            self.tool_calls.extend(tools)

    def output(self) -> dict[str, Any]:
        return {"text": self.text, "reasoning": self.reasoning,
                "tool_calls": self.tool_calls, "usage": self.usage,
                "finish_reason": self.finish_reason, "done": self.done,
                "stream_errors": self.errors, "sse_events": self.events,
                "sse_frames": self.frames}

def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if number == number and abs(number) != float("inf"):
            return number
    return None

def measure(acc: SSEAccumulator, wall_ns: int, status: int | None,
            transport_error: str | None, max_tokens: int,
            cache_expectation: str, expected_prompt_tokens: int | None) -> dict[str, Any]:
    usage = acc.usage or {}
    prompt = numeric(usage.get("prompt_tokens"))
    completion = numeric(usage.get("completion_tokens"))
    cached = numeric((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
    problems = list(acc.errors)
    if transport_error:
        problems.append(transport_error)
    if status != 200:
        problems.append(f"http_status_{status}")
    if not acc.done:
        problems.append("missing_sse_done")
    if acc.finish_reason is None:
        problems.append("missing_finish_reason")
    if prompt is None or completion is None or prompt < 0 or completion < 0:
        problems.append("missing_or_invalid_usage")
    if cache_expectation == "cold" and cached != 0:
        problems.append("cold_cache_not_confirmed_zero")
    if cache_expectation == "warm" and (cached is None or cached <= 0):
        problems.append("warm_cache_not_confirmed")
    success = not problems
    wall = wall_ns / 1e9
    ttft = acc.first_token_ns / 1e9 if acc.first_token_ns is not None else None
    decode_wall = wall - ttft if ttft is not None else None
    server_tg = numeric(usage.get("generation_tokens_per_second"))
    # The first SSE payload may contain multiple MTP tokens. This is explicitly
    # N/(wall-TTFT), not a per-token interarrival estimate.
    client_tg = completion / decode_wall if (
        success and completion and decode_wall and decode_wall > 0) else None
    if not success or server_tg is None or server_tg <= 0:
        server_tg = None
    fixed_length = bool(success and completion == max_tokens and acc.finish_reason == "length")
    return {
        "success": success, "problems": problems, "http_status": status,
        "prompt_tokens": int(prompt) if prompt is not None else None,
        "completion_tokens": int(completion) if completion is not None else None,
        "cached_tokens": int(cached) if cached is not None else None,
        "expected_local_prompt_tokens": expected_prompt_tokens,
        "prompt_length_matches_local": prompt == expected_prompt_tokens
        if expected_prompt_tokens is not None and prompt is not None else None,
        "max_tokens": max_tokens, "finish_reason": acc.finish_reason,
        "fixed_length_complete": fixed_length,
        "early_eos": bool(success and acc.finish_reason == "stop" and completion < max_tokens),
        "client_wall_s": wall, "client_ttft_s": ttft,
        "client_decode_wall_s": decode_wall,
        "client_n_over_wall_minus_ttft_tps": client_tg,
        "server_generation_tps": server_tg,
        "server_ttft_s": numeric(usage.get("time_to_first_token")),
        "server_total_s": numeric(usage.get("total_time")),
        "server_prompt_eval_s": numeric(usage.get("prompt_eval_duration")),
        "server_generation_s": numeric(usage.get("generation_duration")),
        "server_prompt_tps": numeric(usage.get("prompt_tokens_per_second")),
    }

def validate_json(text, expected):
    def pairs(values):
        result = {}
        for k, v in values:
            if k in result: raise ValueError('duplicate JSON key')
            result[k] = v
        return result
    try:
        actual = json.loads(text, object_pairs_hook=pairs,
            parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    except (ValueError, TypeError):
        return False
    return digest(actual) == digest(expected)

def validate_task(text, contract, completion_tokens, finish_reason):
    """Require the archived complete output and the original task's contract.

    Python output is compared structurally with the supplied module after only
    the requested rename. No generated Python is executed by this driver.
    """
    checks = {
        'historical_whole_output': hashlib.sha256(text.encode()).hexdigest() == contract['historical_output_sha256'],
        'historical_completion_tokens': completion_tokens == contract['historical_completion_tokens'],
        'normal_stop': finish_reason == 'stop',
    }
    if contract['kind'] == 'json-change':
        checks['exact_json_change'] = validate_json(text, contract['expected_json'])
    elif contract['kind'] == 'python-rename':
        old, new = contract['old_name'], contract['new_name']
        expected = re.sub(r'\b' + re.escape(old) + r'\b', new, contract['original_source'])
        checks['old_name_removed'] = re.search(r'\b' + re.escape(old) + r'\b', text) is None
        try:
            checks['requested_ast_change_only'] = ast.dump(ast.parse(text)) == ast.dump(ast.parse(expected))
        except (SyntaxError, ValueError, TypeError):
            checks['requested_ast_change_only'] = False
    else:
        raise ValueError('Unknown frozen task contract')
    return checks

def occupancy(cache):
    values = {'main_files': 0, 'main_bytes': 0, 'sidecar_files': 0, 'sidecar_bytes': 0, 'transient_bytes': 0}
    for p in cache.rglob('*'):
        try:
            if not p.is_file(): continue
            size = p.stat().st_size
        except FileNotFoundError:
            continue
        kind = 'sidecar' if '_gdn_sidecars' in p.parts else ('main' if p.suffix == '.safetensors' and not any(x.startswith('_') for x in p.relative_to(cache).parts) else 'transient')
        values[kind + '_bytes'] += size
        if kind != 'transient': values[kind + '_files'] += 1
    return values

class Client:
    def __init__(self, port, key, output, cache):
        self.origin = f'http://127.0.0.1:{port}'
        self.key, self.output, self.cache = key, output, cache
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.logged_in = False

    def json(self, path, data=None):
        req = urllib.request.Request(self.origin + path, data=json.dumps(data).encode() if data is not None else None,
            headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        with self.opener.open(req, timeout=10) as reply:
            return json.load(reply)

    def stats(self):
        if not self.logged_in:
            self.json('/admin/api/login', {'api_key': self.key, 'remember': False})
            self.logged_in = True
        return self.json('/admin/api/stats')

    def clear(self):
        self.stats()
        for name in ('hot-cache', 'ssd-cache'):
            if self.json('/admin/api/' + name + '/clear', {}).get('status') != 'ok':
                raise RuntimeError('Cache clear did not succeed')
        zeros = 0
        for _ in range(20):
            runtime = self.stats().get('runtime_cache', {})
            zero = all(runtime.get(k) == 0 for k in ('total_num_files', 'total_size_bytes', 'hot_cache_entries', 'hot_cache_size_bytes'))
            zero &= all(m.get('gdn_sidecar_count', 0) == 0 for m in runtime.get('models', []))
            zeros = zeros + 1 if zero else 0
            if zeros >= 2: return
            if not zero: self.json('/admin/api/ssd-cache/clear', {})
            time.sleep(.1)
        raise RuntimeError('Prefix state did not become stably empty')

    def persisted(self, target):
        samples, previous, stable = [], None, 0
        deadline = time.monotonic() + 30
        time.sleep(.3)
        while time.monotonic() < deadline:
            runtime = self.stats().get('runtime_cache', {})
            model = next((m for m in runtime.get('models', []) if m.get('id') == MODEL), {})
            row = {k: model.get(k) for k in ('num_files', 'saves_persisted', 'errors')}
            row.update(pending_bytes=model.get('gdn_staging', {}).get('pending_bytes'), disk=occupancy(self.cache))
            samples.append(row)
            stable = stable + 1 if row == previous and row['pending_bytes'] == 0 else 0
            if stable >= 3 and row['disk']['main_files'] and row['disk']['sidecar_files']:
                save(target, {'ready': True, 'snapshots': samples})
                return row['disk']
            previous = row
            time.sleep(.25)
        save(target, {'ready': False, 'snapshots': samples})
        raise RuntimeError('Prefix writer did not become durably quiescent')

    def request(self, stage, label, fixture, measured=True, cold=False, minimum=0, persist=False, expected=None, fixed512=False, contract=None):
        path = self.output / stage / label
        payload = copy.deepcopy(fixture['payload'])
        metadata = fixture.get('metadata', {})
        save(path / 'request.json', {'payload': payload, 'metadata': metadata, 'measured': measured, 'cache_expectation': 'cold' if cold else 'observe'})
        if cold: self.clear()
        acc = SSEAccumulator(); status = None; error = None
        t0 = time.perf_counter_ns()
        try:
            req = urllib.request.Request(self.origin + '/v1/chat/completions', data=json.dumps(payload).encode(),
                headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
            with self.opener.open(req, timeout=900) as reply:
                status = reply.status
                for raw in reply:
                    acc.line(raw.decode().rstrip('\r\n').replace(self.key, '[REDACTED]'), time.perf_counter_ns() - t0)
                acc.flush(time.perf_counter_ns() - t0)
        except Exception as exc:
            error = type(exc).__name__
        wall = time.perf_counter_ns() - t0
        result = measure(acc, wall, status, error, payload['max_tokens'], 'cold' if cold else 'observe', metadata.get('local_prompt_tokens'))
        result.update(measured=measured, stage=stage, label=label, payload_sha256=digest(payload), output_sha256=hashlib.sha256(acc.text.encode()).hexdigest())
        checks = {'transport': result['success'], 'no_tool_calls': not acc.tool_calls,
            'prompt_length': result['prompt_length_matches_local'] is not False,
            'cache_minimum': isinstance(result['cached_tokens'], int) and result['cached_tokens'] >= minimum,
            'minimum_completion': (result['completion_tokens'] or 0) >= metadata.get('required_min_completion_tokens', 0)}
        if fixed512: checks['fixed512'] = result['fixed_length_complete'] and result['completion_tokens'] == 512
        if expected is not None:
            checks.update(json_correct=validate_json(acc.text, expected), normal_stop=result['finish_reason'] == 'stop', no_reasoning=not acc.reasoning)
        if contract is not None:
            checks.update(validate_task(acc.text, contract, result['completion_tokens'], result['finish_reason']))
            checks['no_reasoning'] = not acc.reasoning
        result['checks'] = checks
        save(path / 'output.json', acc.output()); save(path / 'result.json', result)
        if not all(checks.values()):
            raise RuntimeError('Request acceptance failed: ' + str(path))
        if persist:
            result['occupancy'] = self.persisted(path / 'cache-ready.json')
            save(path / 'result.json', result)
        print(json.dumps({'stage': stage, 'label': label, 'measured': measured, 'ttft_s': result['client_ttft_s'], 'cached': result['cached_tokens']}), flush=True)
        return result, acc.text

def wait_ready(child, client):
    deadline = time.monotonic() + 900
    while child.poll() is None:
        try:
            value = client.json('/v1/models')
            if MODEL in {r['id'] for r in value.get('data', [])}: return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        if time.monotonic() >= deadline: raise RuntimeError('Server readiness timeout')
        time.sleep(.5)
    raise RuntimeError('Owned server exited before readiness')

def token_count(source, model, payload):
    code = 'import json,sys; from transformers import AutoTokenizer; p=json.load(sys.stdin); t=AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=False); s=t.apply_chat_template(p["messages"],tokenize=False,add_generation_prompt=True,enable_thinking=False); print(len(t.encode(s)))'
    result = subprocess.run([str(source / '.venv/bin/python'), '-c', code, str(model)], input=json.dumps(payload), text=True, capture_output=True, check=True, env=environment('', 'fixed3lookup', 'baseline'))
    return int(result.stdout.strip())

def core(client, fixtures):
    rows = []
    for n in (2048, 8192, 32768):
        for r in range(6):
            row, _ = client.request('core', f'p{n}-' + ('warmup' if r == 0 else f'r{r:02d}'), fixtures[f'core{n}'], measured=r > 0, cold=True, fixed512=True)
            rows.append(row)
    return rows

def tasks(client, fixtures):
    rows = []
    for fixture in fixtures['tasks']['fixtures']:
        name = fixture['metadata']['id']
        for repetition in range(6):
            label = name + '-' + ('warmup' if repetition == 0 else f'r{repetition:02d}')
            row, _ = client.request('tasks', label, fixture, measured=repetition > 0,
                cold=True, contract=fixture['contract'])
            rows.append(row)
    return rows

def prefix(client, fixtures, optimized, source, model):
    rows = []
    functional = fixtures['functional']
    text = functional['payload']['messages'][-1]['content']
    expected = json.loads(text[text.index('{'):]); expected['service']['request_timeout_seconds'] = 45
    for r in range(2):
        row, _ = client.request('functional', 'cold' if r == 0 else 'warm', functional, measured=False, cold=r == 0, minimum=1 if r else 0, persist=True, expected=expected); rows.append(row)
    original = None
    for r in range(6):
        row, body = client.request('same-prompt', 'cold-00' if r == 0 else f'warm-{r:02d}', fixtures['core7792'], cold=r == 0, minimum=7785 if optimized and r else 0, persist=True); rows.append(row)
        if r == 0: original = body
    continuation = copy.deepcopy(fixtures['core7792'])
    continuation['payload']['messages'].extend([{'role': 'assistant', 'content': original}, {'role': 'user', 'content': 'Continue the unfinished module with more distinct tests. Output code only.'}])
    continuation['payload']['max_tokens'] = 256
    continuation['metadata'] = {'local_prompt_tokens': token_count(source, model, continuation['payload'])}
    for r in range(1, 4):
        row, _ = client.request('continuation', f'warm-{r:02d}', continuation, minimum=1, persist=True); rows.append(row)
    for i, fixture in enumerate(fixtures['public']):
        row, _ = client.request('public-prefix', f'variant-{i+1:02d}', fixture, measured=i >= 2, cold=i == 0, minimum=2048 if optimized and i >= 2 else 0, persist=True); rows.append(row)
    for mode in ('cold', 'warm'):
        for r in range(1, 4):
            row, _ = client.request('context32k', f'{mode}-{r:02d}', fixtures['core32k'], cold=mode == 'cold', minimum=32761 if optimized and mode == 'warm' else (1 if mode == 'warm' else 0), persist=True); rows.append(row)
    if optimized:
        semantic = fixtures['semantic']['fixtures']
        for i, fixture in enumerate(semantic + [semantic[-1], semantic[-1]]):
            minimum = fixture['metadata']['local_prompt_tokens'] - 7 if i >= 5 else (2048 if i >= 2 else 0)
            row, _ = client.request('semantic', f'r{i+1:02d}', fixture, measured=False, minimum=minimum, expected=fixture['metadata']['expected_json'])
            rows.append(row)
    return rows

def summarize(rows):
    groups = {}
    for row in rows:
        if not row['measured']: continue
        key = row['profile'] + '/' + row['stage'] + '/' + ('cold' if row['label'].startswith('cold') else ('warm' if row['label'].startswith('warm') else row['label'].split('-')[0]))
        groups.setdefault(key, []).append(row)
    return {key: {'samples': len(group), 'metrics': {name: {'all': [r[name] for r in group], 'median': statistics.median([r[name] for r in group if numeric(r[name]) is not None]) if any(numeric(r[name]) is not None for r in group) else None} for name in ('client_ttft_s', 'server_prompt_tps', 'server_generation_tps', 'client_wall_s', 'cached_tokens', 'completion_tokens')}} for key, group in groups.items()}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--suite', choices=('core', 'tasks', 'prefix', 'all'), default='all')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--port', type=int, default=8012)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    fixtures = load_fixtures()
    jobs = []
    if a.suite in ('core', 'all'): jobs += [('core', 'baseline', profile) for profile in ('fixed3', 'fixed3lookup')]
    if a.suite in ('tasks', 'all'): jobs += [('tasks', 'baseline', profile) for profile in ('fixed3', 'fixed3lookup')]
    if a.suite in ('prefix', 'all'): jobs += [('prefix', v, 'fixed3lookup') for v in ('baseline', 'optimized')]
    output, workdir, model = a.output_dir.expanduser().resolve(), a.workdir.expanduser().resolve(), a.model_dir.expanduser().resolve()
    if a.dry_run:
        print(json.dumps({'jobs': jobs, 'core_requests': 36 if a.suite in ('core', 'all') else 0, 'task_requests': 24 if a.suite in ('tasks', 'all') else 0, 'prefix_requests': 44 if a.suite in ('prefix', 'all') else 0, 'extra_semantic_requests': 7 if a.suite in ('prefix', 'all') else 0, 'fixtures': list(fixtures), 'output_dir': str(output), 'ephemeral_credentials': True, 'owns_only_its_servers': True}, indent=2)); return
    verify_model(model)
    with socket.socket() as probe: probe.bind(('127.0.0.1', a.port))
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'manifest.json', {'jobs': jobs, 'fixture_manifest_sha256': hashlib.sha256((FIXTURES / 'manifest.json').read_bytes()).hexdigest(), 'model_config_sha256': hashlib.sha256((model / 'config.json').read_bytes()).hexdigest(), 'measurement': 'Historical frozen protocol, relocated durable engine patches; no system limit changes, no resource polling or remote result upload'})
    rows = []; status = 'running'
    try:
        for suite, variant, profile in jobs:
            label = suite + '-' + variant + '-' + profile
            runtime = output / label / 'runtime'
            prepare_runtime(runtime, model, profile)
            key = secrets.token_urlsafe(32)
            concurrency = 8 if suite == 'prefix' else 1
            cmd, source = launch_plan(workdir, model, variant, profile, runtime, a.port, concurrency=concurrency)
            env = environment(key, profile, variant)
            identity = runtime_identity(workdir, model, variant, profile)
            save(output / label / 'launch.json', {'argv': cmd, 'environment': {k: v for k, v in env.items() if k.startswith('OMLX_') and k != 'OMLX_API_KEY'}, 'source_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(), 'source_diff_sha256': hashlib.sha256(subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD'])).hexdigest(), 'runtime_identity': identity})
            log = output / label / 'server.log'
            child = None
            try:
                with log.open('xb') as stream:
                    child = subprocess.Popen(cmd, cwd=source, env=env, stdout=stream, stderr=stream)
                client = Client(a.port, key, output / label / 'requests', runtime / 'cache')
                wait_ready(child, client)
                if suite == 'core': values = core(client, fixtures)
                elif suite == 'tasks': values = tasks(client, fixtures)
                else: values = prefix(client, fixtures, variant == 'optimized', source, model)
                if runtime_identity(workdir, model, variant, profile) != identity:
                    raise RuntimeError('Model or full source/native package changed during benchmark')
                text = log.read_text(errors='replace')
                if 'Qwen qmm NAX dispatch enabled' not in text or ': resident' not in text:
                    raise RuntimeError('Required NAX dispatch or resident PLE route was not observed; see ' + str(log))
                for row in values: row['profile'] = label
                rows += values
                save(output / 'rows.json', rows); save(output / 'summary.json', summarize(rows))
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=45)
                    except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=15)
        status = 'passed'
    except BaseException:
        status = 'failed'
        raise
    finally:
        save(output / 'status.json', {'status': status, 'requests_completed': len(rows), 'all_raw_results_preserved': True})
    print(json.dumps({'status': status, 'output_dir': str(output), 'requests': len(rows)}))

if __name__ == '__main__':
    main()
