#!/usr/bin/env python3
"""Launch the pinned Qwen build with isolated settings and standard oMLX CLI."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen3.8-Flash-Next-oQ6e-mtp'

def check_port_available(port, host='127.0.0.1'):
    # A cleanly stopped previous server can leave accepted sockets in TIME_WAIT.
    # SO_REUSEADDR permits that restart; an active listener still fails bind.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))

def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_model(model):
    m = json.loads((ROOT / 'profiles/qwen/manifest.json').read_text())
    for name, expected in [('config.json', m['model_config_sha256']), ('model.safetensors.index.json', m['model_index_sha256'])]:
        if fingerprint(model / name) != expected:
            raise RuntimeError('Wrong pinned Qwen model: ' + name)
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    for name in set(index['weight_map'].values()):
        if not (model / name).is_file():
            raise RuntimeError('Missing model shard: ' + name)

def settings(profile):
    return {'version': 1, 'models': {MODEL: {'is_pinned': True, 'mtp_enabled': True,
        'mtp_fixed_depth': 3, 'qwen4_ple_ssd_offload': False, 'moe_expert_offload_enabled': False,
        'qwen35_oq_a8_enabled': False, 'turboquant_kv_enabled': False,
        'max_context_window': 262144, 'enable_thinking': False,
        'chat_template_kwargs': {'enable_thinking': False}, 'temperature': 0}}}

def environment(key, profile, variant):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('OMLX_', 'DS4_', 'DS41_', 'LOCAL_MOE_', 'BASELINE_')) and k not in ('PYTHONPATH', 'PYTHONHOME')}
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        PYTHONHASHSEED='0', OMLX_API_KEY=key, OMLX_QWEN4_PLE_MODE='resident',
        OMLX_P2_LOOKUP='1' if profile == 'fixed3lookup' else '0',
        OMLX_P2_LOOKUP_NGRAM='8', OMLX_P2_LOOKUP_MAX='7', OMLX_P2_LOOKUP_LONG='32',
        OMLX_P2_LOOKUP_MAX_LONG='14', OMLX_P2_LOOKUP_GATE='1', OMLX_P2_ROW_EXACT_MAX_ROWS='4')
    if variant == 'optimized':
        env.update(OMLX_PREFIX_BLOCK_SIZE='8192', OMLX_PREFIX_PUBLIC_BOUNDARIES='1')
    return env

def launch_plan(workdir, model, variant, profile, runtime, port, host='127.0.0.1', concurrency=8):
    source = workdir / variant / 'source'
    argv = [str(source / '.venv/bin/omlx'), 'serve', '--model-dir', str(runtime / 'model-registry'),
        '--host', host, '--port', str(port), '--base-path', str(runtime / 'config'),
        '--max-concurrent-requests', str(concurrency), '--memory-guard-gb', '232',
        '--hot-cache-max-size', '0', '--paged-ssd-cache-dir', str(runtime / 'cache'),
        '--paged-ssd-cache-max-size', '50GB']
    return argv, source

def runtime_identity(workdir, model, variant, profile):
    source = workdir / variant / 'source'
    files = {}
    for p in sorted((source / 'omlx').rglob('*')):
        if p.is_file() and p.suffix in ('.py', '.metal', '.cpp', '.h', '.hpp', '.so', '.dylib', '.metallib'):
            files[str(p.relative_to(source))] = fingerprint(p)
    return {'model_path': str(model), 'model_config_sha256': fingerprint(model / 'config.json'),
        'model_index_sha256': fingerprint(model / 'model.safetensors.index.json'),
        'variant': variant, 'profile': profile, 'source_files_sha256': files,
        'prefix_block_size': 8192, 'public_prefix': variant == 'optimized'}

def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def prepare_runtime(runtime, model, profile, reuse=False, identity=None):
    model = model.resolve()
    if runtime.exists():
        if not reuse or identity is None or not (runtime / 'identity.json').is_file():
            raise RuntimeError('Refusing to reuse unverified runtime: ' + str(runtime))
        if json.loads((runtime / 'identity.json').read_text()) != identity:
            raise RuntimeError('Runtime source/model/profile identity changed; choose a new --runtime-dir')
        if not (runtime / 'model-registry' / MODEL).is_symlink() or (runtime / 'model-registry' / MODEL).resolve() != model:
            raise RuntimeError('Runtime model registry changed')
    else:
        runtime.mkdir(parents=True, mode=0o700)
        (runtime / 'model-registry').mkdir()
        (runtime / 'model-registry' / MODEL).symlink_to(model, target_is_directory=True)
        (runtime / 'config').mkdir()
        if identity is not None: atomic_json(runtime / 'identity.json', identity)
    atomic_json(runtime / 'config/model_settings.json', settings(profile))
    # An explicit isolated base path prevents reading ~/.omlx settings.
    atomic_json(runtime / 'config/settings.json', {'auth': {'api_key': None,
        'secret_key': None, 'skip_api_key_verification': False, 'allow_unauthenticated_inference': False},
        'logging': {'log_dir': str(runtime / 'logs')}})

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--variant', choices=('baseline', 'optimized'), default='optimized')
    p.add_argument('--profile', choices=('fixed3', 'fixed3lookup'), default='fixed3lookup')
    p.add_argument('--runtime-dir', type=Path, help='Fresh settings/cache directory; default random directory within workdir')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8012)
    p.add_argument('--max-concurrent-requests', type=int, default=8)
    p.add_argument('--mode', choices=('benchmark', 'cached'), help='benchmark: baseline/fixed3/concurrency1; cached: optimized/fixed3lookup/concurrency8')
    p.add_argument('--dry-run', action='store_true')
    return p

def main(argv=None):
    p = parser(); a = p.parse_args(argv)
    if a.mode:
        a.variant, a.profile, a.max_concurrent_requests = ('baseline', 'fixed3', 1) if a.mode == 'benchmark' else ('optimized', 'fixed3lookup', 8)
    workdir, model = a.workdir.expanduser().resolve(), a.model_dir.expanduser().resolve()
    runtime = a.runtime_dir.expanduser().resolve() if a.runtime_dir else workdir / 'runs' / ('cached' if a.mode == 'cached' else 'serve-' + secrets.token_hex(6))
    cmd, cwd = launch_plan(workdir, model, a.variant, a.profile, runtime, a.port, a.host, a.max_concurrent_requests)
    if a.dry_run:
        env = environment('[ephemeral-or-explicit-env-key]', a.profile, a.variant)
        print(json.dumps({'argv': cmd, 'cwd': str(cwd), 'settings': settings(a.profile), 'environment': {k: v for k, v in env.items() if k.startswith('OMLX_') and k != 'OMLX_API_KEY'}, 'model_dir': str(model), 'runtime': str(runtime)}, indent=2)); return
    verify_model(model)
    check_port_available(a.port, a.host)
    identity = runtime_identity(workdir, model, a.variant, a.profile)
    prepare_runtime(runtime, model, a.profile, reuse=a.mode == 'cached', identity=identity)
    lock = (runtime / '.serve.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    key = os.environ.get('OMLX_API_KEY') or secrets.token_urlsafe(32)
    print('Qwen URL: http://' + a.host + ':' + str(a.port) + '/v1', flush=True)
    if not os.environ.get('OMLX_API_KEY'):
        print('Generated ephemeral API token: ' + key, file=sys.stderr, flush=True)
    child = subprocess.Popen(cmd, cwd=cwd, env=environment(key, a.profile, a.variant))
    try:
        return child.wait()
    except KeyboardInterrupt:
        child.terminate()
        try: child.wait(timeout=45)
        except subprocess.TimeoutExpired: child.kill(); child.wait()
        return 130

if __name__ == '__main__':
    sys.exit(main())
