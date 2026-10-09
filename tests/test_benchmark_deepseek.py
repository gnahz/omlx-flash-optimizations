"""CPU protocol tests: no model loading, GPU import or service access."""
from __future__ import annotations
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_deepseek as bench
import serve_deepseek as serve


class ProtocolTests(unittest.TestCase):
    def test_help_has_no_model_dependencies(self):
        for name in ("benchmark_deepseek.py", "serve_deepseek.py"):
            result = subprocess.run([sys.executable, str(ROOT / "scripts" / name), "--help"],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--work-dir", result.stdout)

    def test_exact_fixtures_are_frozen(self):
        manifest = json.loads((bench.FIXTURES / "manifest.json").read_text())
        for row in manifest["files"]:
            self.assertEqual(serve.sha(bench.FIXTURES / row["path"]), row["sha256"])
        for case in range(3):
            payload = json.loads((bench.FIXTURES / f"case{case}-8k.json").read_bytes())
            self.assertEqual(payload["model"], serve.MODEL_ID)
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(payload["max_tokens"], 512)
            self.assertFalse(payload["enable_thinking"])
            self.assertEqual(hashlib.sha256(payload["messages"][0]["content"].encode()).hexdigest(),
                             manifest["files"][case]["prompt_sha256"])

    def test_full_abba_plan_and_excluded_warmups(self):
        plan = bench.plan(2, [0, 1, 2], "ABBA")
        self.assertEqual(len(plan), 30)
        self.assertEqual(sum(row["warmup"] for row in plan), 6)
        self.assertEqual([row["arm"] for row in plan[:10]], list("ABABBAABBA"))
        self.assertEqual(len(bench.plan(1, [0], "ABBA")), 6)

    def test_long_prompt_keeps_task_and_exact_geometry(self):
        raw = (bench.FIXTURES / "case0-8k.json").read_bytes()
        tokenizer = NS(encode=lambda value, **kw: NS(ids=range(len(value))))
        formatter = lambda messages, **kw: messages[0]["content"]
        original = json.loads(raw)["messages"][0]["content"]
        changed, geometry = bench.long_payload(raw, 0, 4000, tokenizer, formatter)
        prompt = json.loads(changed)["messages"][0]["content"]
        self.assertEqual(len(prompt), 4000)
        self.assertTrue(prompt.endswith(bench.ARCHIVE_SUFFIX + original.split(bench.ARCHIVE_SUFFIX)[1]))
        self.assertEqual(geometry["cpu_prompt_tokens"], 4000)
        self.assertFalse(geometry["historical_output_golden_applicable"])
        malformed = json.loads(raw)
        malformed["messages"][0]["content"] = "unmarked task"
        with self.assertRaises(ValueError):
            bench.long_payload(json.dumps(malformed).encode(), 0, 4000, tokenizer, formatter)

    def test_sse_role_usage_multiline_and_done(self):
        acc = bench.SSEAccumulator()
        acc.line('data: {"choices":[{"delta":{"role":"assistant"}}]}', .1)
        acc.line("", .1)
        self.assertIsNone(acc.first)
        acc.line('data: {"choices":', .2)
        acc.line('data: [{"delta":{"content":"abc"},"finish_reason":"length"}]}', .21)
        acc.line("", .22)
        acc.line('data: {"usage":{"completion_tokens":512},"choices":[]}', .3)
        acc.line("", .3)
        acc.line("data: [DONE]", .4)
        acc.line("", .4)
        self.assertEqual(acc.content, "abc")
        self.assertEqual(acc.first, .2)
        self.assertEqual(acc.usage["completion_tokens"], 512)
        self.assertTrue(acc.done)
        with self.assertRaises(RuntimeError):
            acc.line('data: {"error":{"message":"bad"}}', .5)
            acc.line("", .5)

    def test_missing_done_retains_error_artifacts(self):
        class Response:
            status_code = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def iter_lines(self):
                return iter(['data: {"choices":[{"delta":{"content":"x"},"finish_reason":"length"}]}', ""])
        client = NS(stream=lambda *a, **kw: Response())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            log = root / "server.log"
            log.write_text("")
            with self.assertRaisesRegex(RuntimeError, "Incomplete SSE"):
                bench.request(client, "http://127.0.0.1:8011", b"{}", "key", root / "trial", log)
            self.assertTrue((root / "trial-sse.txt").exists())
            row = json.loads((root / "trial-output.json").read_text())
            self.assertIn("Incomplete SSE", row["error"])

    def test_stats_distinguish_bulk_from_tail_fallback(self):
        good = dict(metadata_prefill_direct_calls=20, metadata_prefill_dense_calls=0,
                    metadata_dense_expert_calls=20, dspark_single_calls=10)
        bench.validate_arm_stats(good, "B")
        with self.assertRaises(RuntimeError):
            bench.validate_arm_stats(dict(good, metadata_prefill_dense_calls=1), "B")
        bench.validate_arm_stats(dict(good, metadata_prefill_direct_calls=0,
                                     metadata_prefill_dense_calls=20), "A")
        with self.assertRaises(RuntimeError):
            bench.validate_arm_stats(dict(good, dspark_batch_calls=1), "B")

    def test_counter_identity_excludes_only_timing(self):
        row = json.loads((bench.FIXTURES / "reference.json").read_text())["rows"][0]
        identity = bench.speculative_identity(row)
        self.assertNotIn("timing_ms", identity)
        self.assertIn("prompt_copy", identity)
        broken = copy.deepcopy(row)
        broken["speculative"]["tokens"] -= 1
        with self.assertRaises(RuntimeError):
            bench.speculative_identity(broken)
        for invalid in (True, 0, -1, float("nan"), float("inf")):
            with self.assertRaises(RuntimeError):
                bench.positive(invalid, "rate")

    def test_summary_discards_warmup_and_counts_actual_tokens(self):
        base = json.loads((bench.FIXTURES / "reference.json").read_text())["rows"][0]
        base.update(server_prompt_eval_seconds=1, ttft_seconds=1, server_model_ttft_seconds=1,
            elapsed_seconds=2, whole_output_sha256="a", server_model_load_seconds=None)
        rows = []
        for warmup, arm, prefill in ((True, "A", 99999), (False, "A", 100), (False, "B", 110)):
            row = copy.deepcopy(base)
            row.update(warmup=warmup, arm=arm, server_prefill_tps=prefill, server_decode_tps=80)
            rows.append(row)
        result = bench.summarize(rows)
        self.assertEqual(result["measured_requests"], 2)
        self.assertEqual(result["measured_actual_generated_tokens"], 1024)
        self.assertAlmostEqual(result["geometric_prefill_change_percent"], 10)
        self.assertFalse(result["agent_task_completion_assessed"])

    def test_url_requires_localhost_and_no_credentials(self):
        self.assertEqual(bench.validate_url("http://127.0.0.1:8011/"), "http://127.0.0.1:8011")
        for invalid in ("https://127.0.0.1", "http://example.com", "http://x@127.0.0.1", "http://127.0.0.1/v1"):
            with self.assertRaises(ValueError): bench.validate_url(invalid)

    def test_isolated_config_profiles_and_fresh_key(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model, work = root / "model", root / "work"
            model.mkdir()
            config, key1, benchmark = serve.prepare_config(work, model, "benchmark", "127.0.0.1", 8011)
            self.assertEqual((config / "models" / serve.MODEL_ID).resolve(), model.resolve())
            self.assertEqual(benchmark["max_concurrent_requests"], 1)
            self.assertFalse(benchmark["prefix_cache_enabled"])
            _, key2, _ = serve.prepare_config(work, model, "benchmark", "127.0.0.1", 8011)
            self.assertNotEqual(key1, key2)
            self.assertEqual((config / "api-key.txt").stat().st_mode & 0o777, 0o600)
            self.assertEqual(serve.profile_settings("cached")["max_concurrent_requests"], 8)
            self.assertTrue(serve.profile_settings("cached")["prefix_cache_enabled"])
            self.assertEqual(serve.FLAGS["DS41_NATIVE_GATE_UP"], "0")
            self.assertEqual(serve.FLAGS["DS41_DRAFT_ASYNC"], "1")

    def test_ambient_engine_flags_cannot_override_owned_auth(self):
        clean = serve.isolated_environment(dict(PATH="/usr/bin", OMLX_API_KEY="external-secret",
            OMLX_PREFIX_BLOCK_SIZE="32", OMLX_MEMORY_GUARD_GB="10", DS41_NATIVE_VERIFY="1",
            BASELINE_RUNTIME_METAL="0", PYTHONPATH="/other/checkout", PYTHONHOME="/other/python"))
        self.assertEqual(clean, {"PATH": "/usr/bin"})

    def test_startup_cluster_shim_and_reaper_are_service_owned(self):
        calls = []
        shim = NS(ensure_cluster_python_shim=lambda **kw: calls.append(("shim", kw)))
        launch = NS(reap_orphaned_launches=lambda **kw: calls.append(("reaper", kw)))
        work = Path("/isolated/reproduction")
        serve.isolate_cluster_runtime(shim, launch, work)
        shim.ensure_cluster_python_shim()
        launch.reap_orphaned_launches()
        shim.ensure_cluster_python_shim(home=Path("/other/home"))
        launch.reap_orphaned_launches(state_dir=Path("/other/runtime"))
        self.assertEqual(calls[0][1]["home"], work / "home")
        self.assertEqual(calls[1][1]["state_dir"], work / "cluster/runtime")
        self.assertEqual(calls[2][1]["home"], work / "home")
        self.assertEqual(calls[3][1]["state_dir"], work / "cluster/runtime")

    def test_wait_idle_retries_only_startup_connection_failures(self):
        class ConnectError(Exception): pass
        class ConnectTimeout(Exception): pass
        httpx = NS(ConnectError=ConnectError, ConnectTimeout=ConnectTimeout)
        ready = {"ready": True, "idle": {"inflight_http": 0}}
        with mock.patch.dict(sys.modules, {"httpx": httpx}), \
                mock.patch.object(bench, "control", side_effect=[ConnectError(), ConnectTimeout(), ready]) as control, \
                mock.patch.object(bench.time, "sleep") as sleep:
            self.assertEqual(bench.wait_idle(None, "http://127.0.0.1:8011", "key"), ready)
            self.assertEqual(control.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
        with mock.patch.dict(sys.modules, {"httpx": httpx}), \
                mock.patch.object(bench, "control", side_effect=RuntimeError("Control HTTP 401")) as control, \
                mock.patch.object(bench.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "Control HTTP 401"):
                bench.wait_idle(None, "http://127.0.0.1:8011", "wrong-key")
            self.assertEqual(control.call_count, 1)
            sleep.assert_not_called()

    def test_wait_idle_connection_retries_keep_original_deadline(self):
        class ConnectError(Exception): pass
        class ConnectTimeout(Exception): pass
        httpx = NS(ConnectError=ConnectError, ConnectTimeout=ConnectTimeout)
        with mock.patch.dict(sys.modules, {"httpx": httpx}), \
                mock.patch.object(bench, "control", side_effect=ConnectError()) as control, \
                mock.patch.object(bench.time, "monotonic", side_effect=[0, 0, .5, 2]), \
                mock.patch.object(bench.time, "sleep"):
            with self.assertRaisesRegex(TimeoutError, "ready and idle"):
                bench.wait_idle(None, "http://127.0.0.1:8011", "key", timeout=1)
            self.assertEqual(control.call_count, 2)


def controller(profile="benchmark"):
    pool = NS(_entries={}, _entry_has_scheduler_work=lambda entry: False)
    server = NS(_server_state=NS(engine_pool=pool))
    prefill = NS(ENABLED=True, forward=lambda: None)
    metadata = NS(_dense_prefill_moe=lambda: None, _dense_expert=lambda: None)
    dspark = NS(DRAFT_ASYNC=True, proposal_forward=lambda: None, proposal_forward_batch=lambda: None)
    cache = NS(ready=True, error=None, snapshot=lambda: {})
    return serve.Controller(server, prefill, metadata, dspark, serve.profile_settings(profile), cache, {})


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    def test_controls_reject_active_and_cached_service(self):
        c = controller()
        c.inflight = 1
        self.assertEqual(c.apply({"metadata_prefill": False})[0], 409)
        c.inflight = 0
        self.assertEqual(c.apply({"metadata_prefill": 1})[0], 400)
        self.assertEqual(c.apply({"gate_up": True})[0], 400)
        self.assertEqual(c.apply({"metadata_prefill": False, "reset_stats": True})[0], 200)
        self.assertFalse(c.prefill.ENABLED)
        self.assertEqual(controller("cached").apply({"metadata_prefill": False})[0], 403)

    async def test_controls_require_local_bearer(self):
        c = controller()
        async def unused(*args): pass
        middleware = serve.ControlMiddleware(unused, c, "key")
        for peer, headers, expected in (("127.0.0.1", [], 401), ("192.0.2.1", [(b"authorization", b"Bearer key")], 401),
                                       ("127.0.0.1", [(b"authorization", b"Bearer key")], 200)):
            messages = []
            async def send(message): messages.append(message)
            await middleware(dict(type="http", path=serve.CONTROL_PATH, method="GET", client=(peer, 1234), headers=headers), unused, send)
            self.assertEqual(messages[0]["status"], expected)


if __name__ == "__main__":
    unittest.main()
