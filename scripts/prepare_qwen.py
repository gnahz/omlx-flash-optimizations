#!/usr/bin/env python3
"""Build independent pinned Qwen oMLX environments; never touch another runtime."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'profiles/qwen/manifest.json'

def read_manifest():
    value = json.loads(MANIFEST.read_text())
    for item in value['patches']:
        if hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Patch identity changed: ' + item['path'])
    lock = ROOT / 'profiles/qwen/requirements.lock'
    if hashlib.sha256(lock.read_bytes()).hexdigest() != value['requirements_lock_sha256']:
        raise RuntimeError('Qwen dependency lock changed')
    return value

def prepare_commands(workdir, variant, python, skip_build=False):
    m = read_manifest()
    source = workdir / variant / 'source'
    vpython = source / '.venv/bin/python'
    commands = [(['git', 'clone', m['omlx_url'], str(source)], None),
        (['git', '-C', str(source), 'checkout', '--detach', m['omlx_commit']], None)]
    selected = m['patches'][:m['baseline_patch_count']] if variant == 'baseline' else m['patches']
    for item in selected:
        commands.extend([(['git', '-C', str(source), 'apply', '--check', str(ROOT / item['path'])], None),
            (['git', '-C', str(source), 'apply', str(ROOT / item['path'])], None)])
    if skip_build:
        return commands
    commands += [([python, '-m', 'venv', str(source / '.venv')], None),
        ([str(vpython), '-m', 'pip', 'install', '--upgrade', 'pip'], None),
        ([str(vpython), '-m', 'pip', 'install', '-r', str(ROOT / 'profiles/qwen/requirements.lock'), '--no-build-isolation'], source),
        ([str(vpython), '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', '.'], source),
        ([str(vpython), '-m', 'pip', 'check'], source),
        ([str(vpython), '-c', 'from omlx.custom_kernels import native_kernel_status; s=native_kernel_status(); print(s); assert all(v["available"] for v in s.values()), "Native kernels unavailable"'], source)]
    # mlx source packages use setuptools-scm and build helpers before their wheel is built.
    commands.insert(2 + 2 * len(selected) + 2, ([str(vpython), '-m', 'pip', 'install',
        'setuptools==84.0.0', 'wheel==0.48.0', 'cmake==4.4.4', 'nanobind==2.15.0',
        'setuptools-scm==10.3.4', 'mlx==0.32.2', 'numpy==2.3.5'], source))
    return commands

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--variant', choices=('baseline', 'optimized', 'both'), default='both')
    p.add_argument('--python', default=shutil.which('python3.12') or sys.executable, help='Python 3.12 executable (tested: 3.12.14)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--patch-only', action='store_true', help='Create source checkouts without installing dependencies')
    a = p.parse_args(argv)
    variants = ('baseline', 'optimized') if a.variant == 'both' else (a.variant,)
    workdir = a.workdir.expanduser().resolve()
    plans = {v: prepare_commands(workdir, v, a.python, a.patch_only) for v in variants}
    if a.dry_run:
        print(json.dumps({'workdir': str(workdir), 'commands': {v: [{'argv': cmd, 'cwd': str(cwd) if cwd else None} for cmd, cwd in cmds] for v, cmds in plans.items()}, 'environment': {'OMLX_WITH_CUSTOM_KERNEL': '1'}, 'no_model_load': True}, indent=2))
        return
    for variant in variants:
        if (workdir / variant).exists():
            p.error('Refusing to overwrite existing environment: ' + str(workdir / variant))
    if not a.patch_only:
        version = subprocess.check_output([a.python, '-c', 'import sys;print(str(sys.version_info.major)+"."+str(sys.version_info.minor))'], text=True).strip()
        if version != '3.12':
            p.error('Use --python /path/to/python3.12 for the pinned tested dependencies')
        subprocess.run(['xcrun', '-f', 'metal'], check=True)
    m = read_manifest()
    for variant, commands in plans.items():
        (workdir / variant).mkdir(parents=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith('OMLX_') and k not in ('PYTHONPATH', 'PYTHONHOME')}
        env['OMLX_WITH_CUSTOM_KERNEL'] = '1'
        env['PATH'] = str(workdir / variant / 'source/.venv/bin') + os.pathsep + env.get('PATH', '')
        for cmd, cwd in commands:
            subprocess.run(cmd, cwd=cwd, env=env, check=True)
        record = {'variant': variant, 'base_commit': m['omlx_commit'], 'patches': m['patches'][:m['baseline_patch_count']] if variant == 'baseline' else m['patches'], 'requirements_lock_sha256': m['requirements_lock_sha256'], 'native_build_completed': not a.patch_only}
        if not a.patch_only:
            record['pip_freeze'] = subprocess.check_output([str(workdir / variant / 'source/.venv/bin/python'), '-m', 'pip', 'freeze'], text=True)
        (workdir / variant / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'prepared': list(variants), 'workdir': str(workdir)}))

if __name__ == '__main__':
    main()
