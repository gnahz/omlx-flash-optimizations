"""Native availability stays mandatory when the bulk-log route is ineligible."""
import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_qwen as benchmark

def ready():
    return {'groups': {name: {'available': True} for name in
        ('bonsai', 'decode_fast', 'glm_moe_dsa', 'minimax_m3', 'qwen35_prefill')},
        'nax_available': True, 'nax_qmm_kernels_built': True}

def test_short_task_routes_need_resident_ple_without_bulk_dispatch_log():
    benchmark.check_native_readiness(ready())
    benchmark.check_dispatch_log('tasks', 'Qwen4-Exp PLE mode: resident\nGDN fused verify engaged')
    with pytest.raises(RuntimeError, match='Resident'):
        benchmark.check_dispatch_log('tasks', 'GDN fused verify engaged')

@pytest.mark.parametrize('suite', ['core', 'prefix'])
def test_long_prompt_jobs_still_require_actual_bulk_nax_dispatch(suite):
    with pytest.raises(RuntimeError, match='NAX bulk'):
        benchmark.check_dispatch_log(suite, 'Qwen4-Exp PLE mode: resident')
    benchmark.check_dispatch_log(suite, 'Qwen4-Exp PLE mode: resident\nQwen qmm NAX dispatch enabled')

@pytest.mark.parametrize('missing', ['native_group', 'nax_available', 'nax_qmm_kernels_built'])
def test_native_readiness_never_accepts_fallback_for_tasks(missing):
    value = ready()
    if missing == 'native_group': value['groups']['qwen35_prefill']['available'] = False
    else: value[missing] = False
    with pytest.raises(RuntimeError, match='unavailable'):
        benchmark.check_native_readiness(value)
