"""CPU-only execution of the durable Scheduler methods; no engine/GPU import."""
import ast
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import pytest

SOURCE = Path(os.environ.get('QWEN_TEST_SOURCE', '/nonexistent'))
if not (SOURCE / 'omlx/scheduler.py').is_file():
    pytest.skip('Set QWEN_TEST_SOURCE to the optimized prepared checkout', allow_module_level=True)

def method(name):
    tree = ast.parse((SOURCE / 'omlx/scheduler.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    code = ast.Module(body=[function], type_ignores=[])
    namespace = {'__package__': 'qwen_fixture', 'os': os, 'logger': logging.getLogger(__name__)}
    exec(compile(code, 'durable-scheduler-cpu-test', 'exec'), namespace)
    return namespace[name]

def test_geometry_before_original_ssd_alignment_matches_former_wrapper():
    enlarge = method('_enlarge_block_size_for_arrays_cache')
    for block in (4096, 8192, 16384):
        for step in (0, 1024, 16384):
            state = SimpleNamespace(_qwen4_wide_prefill_step=block,
                config=SimpleNamespace(paged_cache_block_size=256, prefill_step_size=step, paged_ssd_cache_dir=None))
            enlarge(state)
            assert state.config.paged_cache_block_size == block
            assert state.config.prefill_step_size == min(step or block, block)

def test_non_qwen_geometry_is_preserved():
    state = SimpleNamespace(_qwen4_wide_prefill_step=0,
        config=SimpleNamespace(paged_cache_block_size=256, prefill_step_size=1024, paged_ssd_cache_dir=None))
    method('_enlarge_block_size_for_arrays_cache')(state)
    assert state.config.paged_cache_block_size == 256
    assert state.config.prefill_step_size == 1024

@pytest.mark.parametrize('native,block,expected', [(True, 4096, 4096), (True, 8192, 8192), (True, 16384, 16384), (False, 4096, 0)])
def test_native_capability_probe_is_preserved(native, block, expected):
    fast = SimpleNamespace(is_native_available=lambda: native, has_symbol=lambda name: True)
    fake = {'qwen_fixture.custom_kernels.glm_moe_dsa': SimpleNamespace(fast=fast),
        'qwen_fixture.custom_kernels.nax': SimpleNamespace(is_nax_available=lambda: True),
        'qwen_fixture.settings': SimpleNamespace(get_system_memory=lambda: 256 * 1024**3)}
    with patch.dict(sys.modules, fake), patch.dict(os.environ, {'OMLX_PREFIX_BLOCK_SIZE': str(block)}):
        state = SimpleNamespace(model=SimpleNamespace(model_type='qwen4_exp'))
        assert method('_detect_qwen4_wide_prefill_step')(state) == expected

def test_standard_callback_binds_policy_without_startup_overlay():
    text = (SOURCE / 'omlx/cache/public_prefix_runtime.py').read_text()
    function = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'stored')
    namespace = {}; exec(compile(ast.Module(body=[function], type_ignores=[]), 'cpu-stored', 'exec'), namespace)
    runtime = SimpleNamespace(stored=namespace['stored'])
    events = []
    state = SimpleNamespace(_public_prefix_policy=SimpleNamespace(bind=lambda *args: events.append(args)),
        paged_cache_manager=SimpleNamespace(blocks={9: SimpleNamespace(block_hash=b'checkpoint')}))
    with patch.dict(sys.modules, {'qwen_fixture.cache.public_prefix_runtime': runtime}):
        method('_on_auxiliary_tail_stored')(state, [1, 2, 3], SimpleNamespace(block_ids=[9]))
    assert events == [([1, 2, 3], b'checkpoint')]
