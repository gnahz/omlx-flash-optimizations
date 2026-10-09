"""Offline clean patch application against a prepared checkout's Git object DB."""
import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('QWEN_TEST_SOURCE', '/nonexistent'))
if not (SOURCE / '.git').is_dir():
    pytest.skip('Set QWEN_TEST_SOURCE to a prepared Qwen checkout', allow_module_level=True)

def test_ordered_patches_apply_to_exact_clean_v070_and_include_untracked_modules():
    manifest = json.loads((ROOT / 'profiles/qwen/manifest.json').read_text())
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / 'source'
        subprocess.run(['git', 'clone', '--quiet', '--shared', str(SOURCE), str(clone)], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(clone), 'checkout', '--quiet', '--detach', manifest['omlx_commit']], check=True)
        for item in manifest['patches']:
            subprocess.run(['git', '-C', str(clone), 'apply', '--check', str(ROOT / item['path'])], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(clone), 'apply', str(ROOT / item['path'])], check=True, capture_output=True)
        for name in ('omlx/server.py', 'omlx/scheduler.py', 'omlx/cache/paged_cache.py', 'omlx/cache/public_prefix.py',
                'omlx/cache/public_prefix_runtime.py', 'omlx/patches/p2_lookup.py', 'omlx/patches/mlx_lm_mtp/batch_generator.py'):
            actual = (clone / name).read_bytes()
            assert actual == (SOURCE / name).read_bytes()
            ast.parse(actual.decode())

def test_baseline_includes_isolation_and_policy_without_prefix_optimization():
    baseline = SOURCE.parent.parent / 'baseline/source'
    if not (baseline / '.git').is_dir():
        pytest.skip('Prepared baseline sibling checkout is unavailable')
    manifest = json.loads((ROOT / 'profiles/qwen/manifest.json').read_text())
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / 'source'
        subprocess.run(['git', 'clone', '--quiet', '--shared', str(SOURCE), str(clone)], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(clone), 'checkout', '--quiet', '--detach', manifest['omlx_commit']], check=True)
        for item in manifest['patches'][:manifest['baseline_patch_count']]:
            subprocess.run(['git', '-C', str(clone), 'apply', '--check', str(ROOT / item['path'])], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(clone), 'apply', str(ROOT / item['path'])], check=True, capture_output=True)
        for name in ('omlx/server.py', 'omlx/scheduler.py', 'omlx/patches/p2_lookup.py', 'omlx/patches/mlx_lm_mtp/batch_generator.py'):
            assert (clone / name).read_bytes() == (baseline / name).read_bytes()
        assert not (clone / 'omlx/cache/public_prefix.py').exists()
