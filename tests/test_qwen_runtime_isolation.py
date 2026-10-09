"""Exercise actual startup path expressions without importing MLX or a server."""
import ast
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

SOURCE = Path(os.environ.get('QWEN_TEST_SOURCE', '/nonexistent'))
if not (SOURCE / 'omlx/server.py').is_file():
    pytest.skip('Set QWEN_TEST_SOURCE to a prepared Qwen checkout', allow_module_level=True)

def startup_nodes():
    module = ast.parse((SOURCE / 'omlx/server.py').read_text())
    return next(n for n in module.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'lifespan')

def test_explicit_runtime_base_never_reads_or_replaces_user_home():
    assignment = next(n for n in startup_nodes().body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'cluster_base' for t in n.targets))
    namespace = {'Path': Path, '_server_state': SimpleNamespace(
        global_settings=SimpleNamespace(base_path='/isolated/run/config'))}
    with patch.object(Path, 'home', side_effect=AssertionError('User home was accessed')):
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<startup-path>', 'exec'), namespace)
    assert namespace['cluster_base'] == Path('/isolated/run/config')

def test_reaper_and_shim_receive_service_owned_paths():
    body = startup_nodes()
    reaper = next(n for n in ast.walk(body) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == 'to_thread'
        and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == 'reap_orphaned_launches')
    shim = next(n for n in ast.walk(body) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == 'ensure_cluster_python_shim')
    calls = []
    async def thread(function, *args):
        calls.append(('reaper', function, args)); return {}
    def ensure(**kwargs):
        calls.append(('shim', kwargs))
    namespace = {'asyncio': SimpleNamespace(to_thread=thread), 'cluster_base': Path('/isolated/run/config'),
        'reap_orphaned_launches': object(), 'ensure_cluster_python_shim': ensure}
    reaper_value = eval(compile(ast.Expression(reaper), '<startup-reaper>', 'eval'), namespace)
    asyncio.run(reaper_value)
    eval(compile(ast.Expression(shim), '<startup-shim>', 'eval'), namespace)
    assert calls[0][2] == (Path('/isolated/run/config/cluster/runtime'),)
    assert calls[1][1] == {'home': Path('/isolated/run/config/home')}
