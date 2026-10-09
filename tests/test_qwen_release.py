"""Portable CPU tests for Qwen launch isolation, frozen protocols, and telemetry."""
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_qwen as prepare
import serve_qwen as serve
import benchmark_qwen as benchmark

class QwenReleaseTests(unittest.TestCase):
    def test_port_probe_rejects_listener_but_allows_clean_tcp_restart(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0)); listener.listen()
            port = listener.getsockname()[1]
            with self.assertRaises(OSError): serve.check_port_available(port)
            with socket.create_connection(('127.0.0.1', port)) as client:
                accepted, _ = listener.accept()
                accepted.shutdown(socket.SHUT_RDWR); accepted.close()
                self.assertEqual(client.recv(1), b'')
        serve.check_port_available(port)

    def test_patch_manifest_and_dependency_identity(self):
        manifest = prepare.read_manifest()
        self.assertEqual(manifest['baseline_patch_count'], 4)
        self.assertEqual(manifest['omlx_commit'], '4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40')

    def test_prepare_bootstraps_before_git_source_dependencies(self):
        commands = prepare.prepare_commands(Path('/tmp/fresh-qwen'), 'optimized', 'python3.12')
        install = [cmd for cmd, _ in commands if '-m' in cmd and 'pip' in cmd]
        self.assertIn('setuptools-scm==10.3.4', install[1])
        self.assertIn('-r', install[2])
        self.assertIn('--no-deps', install[3])

    def test_environment_replaces_foreign_settings_without_retaining_secrets(self):
        with patch.dict('os.environ', {'OMLX_API_KEY': 'old', 'OMLX_PREFIX_BLOCK_SIZE': '16384', 'DS41_NATIVE_METADATA': 'private-path', 'PYTHONPATH': 'oldsource'}):
            env = serve.environment('new', 'fixed3', 'baseline')
        self.assertEqual(env['OMLX_API_KEY'], 'new')
        self.assertEqual(env['OMLX_P2_LOOKUP'], '0')
        self.assertNotIn('PYTHONPATH', env)
        self.assertNotIn('DS41_NATIVE_METADATA', env)
        self.assertNotIn('OMLX_PREFIX_BLOCK_SIZE', env)

    def test_final_profile_pins_resident_weights_and_exact_lookup_geometry(self):
        env = serve.environment('key', 'fixed3lookup', 'optimized')
        self.assertEqual(env['OMLX_QWEN4_PLE_MODE'], 'resident')
        self.assertEqual(env['OMLX_PREFIX_BLOCK_SIZE'], '8192')
        self.assertEqual(env['OMLX_PREFIX_PUBLIC_BOUNDARIES'], '1')
        values = serve.settings('fixed3lookup')['models'][serve.MODEL]
        self.assertFalse(values['qwen4_ple_ssd_offload'])
        self.assertEqual(values['mtp_fixed_depth'], 3)

    def test_verified_cached_runtime_preserves_cache_and_rejects_identity_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); model = base / 'model'; model.mkdir()
            runtime = base / 'runtime'; identity = {'model': 'fixed', 'source': 'fixed'}
            serve.prepare_runtime(runtime, model, 'fixed3lookup', identity=identity)
            (runtime / 'cache').mkdir(); (runtime / 'cache/test.snapshot').write_text('retained')
            serve.prepare_runtime(runtime, model, 'fixed3lookup', reuse=True, identity=identity)
            self.assertTrue((runtime / 'cache/test.snapshot').is_file())
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                serve.prepare_runtime(runtime, model, 'fixed3lookup', reuse=True, identity={'source': 'changed'})
            with self.assertRaisesRegex(RuntimeError, 'unverified runtime'):
                serve.prepare_runtime(runtime, model, 'fixed3lookup')

    def test_runtime_identity_covers_untracked_modules_and_native_binaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root / 'model'; model.mkdir()
            for name in ('config.json', 'model.safetensors.index.json'): (model / name).write_text('{}')
            package = root / 'optimized/source/omlx'; package.mkdir(parents=True)
            module = package / 'untracked_cache.py'; module.write_text('value = 1\n')
            binary = package / 'kernel.so'; binary.write_bytes(b'native1')
            first = serve.runtime_identity(root, model, 'optimized', 'fixed3lookup')
            module.write_text('value = 2\n')
            second = serve.runtime_identity(root, model, 'optimized', 'fixed3lookup')
            self.assertNotEqual(first, second)
            binary.write_bytes(b'native2')
            self.assertNotEqual(second, serve.runtime_identity(root, model, 'optimized', 'fixed3lookup'))

    def test_all_fixture_hashes_and_greedy_request_settings(self):
        fixtures = benchmark.load_fixtures()
        for n in (2048, 8192, 32768):
            fixture = fixtures[f'core{n}']
            self.assertEqual(fixture['metadata']['local_prompt_tokens'], n)
            self.assertEqual(fixture['payload']['max_tokens'], 512)
            self.assertEqual(fixture['payload']['temperature'], 0)
            self.assertFalse(fixture['payload']['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(len(fixtures['public']), 5)
        self.assertEqual(len(fixtures['semantic']['fixtures']), 5)

    def test_core_schedule_is_one_warmup_five_cold_measured_per_context(self):
        class Recorder:
            def __init__(self): self.calls = []
            def request(self, *args, **kwargs): self.calls.append((args, kwargs)); return {}, ''
        client = Recorder(); benchmark.core(client, benchmark.load_fixtures())
        self.assertEqual(len(client.calls), 18)
        self.assertEqual(sum(kwargs['measured'] for _, kwargs in client.calls), 15)
        self.assertTrue(all(k['cold'] and k['fixed512'] for _, k in client.calls))

    def test_prefix_schedule_keeps_public_suffixes_distinct_and_semantics_separate(self):
        class Recorder:
            def __init__(self): self.calls = []
            def request(self, *args, **kwargs): self.calls.append((args, kwargs)); return {}, 'generated code'
        client = Recorder()
        with patch.object(benchmark, 'token_count', return_value=8459):
            benchmark.prefix(client, benchmark.load_fixtures(), True, Path('/unused'), Path('/unused'))
        self.assertEqual(len(client.calls), 29)
        self.assertEqual(sum(k.get('measured', True) for _, k in client.calls), 18)
        public = [a[2]['payload']['messages'][-1]['content'] for a, _ in client.calls if a[0] == 'public-prefix']
        self.assertEqual(len(set(public)), 5)
        semantic = [k for a, k in client.calls if a[0] == 'semantic']
        self.assertEqual(len(semantic), 7)
        self.assertTrue(all(not k['measured'] and not k.get('persist', False) for k in semantic))

    def test_sse_roles_do_not_count_as_first_generated_token(self):
        acc = benchmark.SSEAccumulator()
        acc.line('data: ' + json.dumps({'choices': [{'delta': {'role': 'assistant'}}]}), 10)
        acc.line('', 11)
        self.assertIsNone(acc.first_token_ns)
        acc.line('data: ' + json.dumps({'choices': [{'delta': {'content': 'abc'}}]}), 20)
        acc.line('', 21)
        self.assertEqual(acc.first_token_ns, 20)

    def test_invalid_duplicate_or_type_changed_json_rejected(self):
        self.assertFalse(benchmark.validate_json('{"v":1,"v":2}', {'v': 2}))
        self.assertFalse(benchmark.validate_json('{"v":true}', {'v': 1}))
        self.assertTrue(benchmark.validate_json('{"v":1}', {'v': 1}))

if __name__ == '__main__': unittest.main()
