# SPDX-License-Identifier: Apache-2.0
"""Portable decode profile controls: CPU mocks, no ports or model imports."""
from __future__ import annotations

import asyncio
import builtins
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_deepseek as bench
import serve_deepseek as serve


def make_controller(profile="cached", growth=None, scheduler=None, core_flags=None):
    config = NS(max_num_seqs=8, prefill_step_size=8192,
                preserved_prefill_max_defer_seconds=8.0)
    scheduler = scheduler or NS(config=config, _preserves_prefill_chunks=lambda: True)
    core = NS(scheduler=scheduler, **({"_terminal_response_barrier_enabled": True}
                                    if core_flags is None else core_flags))
    pool = NS(_entries={serve.MODEL_ID: NS(engine=NS(_engine=NS(engine=core)),
        is_loading=False, in_use=0)}, _entry_has_scheduler_work=lambda entry: False)
    server = NS(_server_state=NS(engine_pool=pool))
    prefill = NS(ENABLED=True, forward=lambda: None)
    metadata = NS(_dense_prefill_moe=lambda: None, _dense_expert=lambda: None)
    dspark = NS(DRAFT_ASYNC=True, proposal_forward=lambda: None, proposal_forward_batch=lambda: None)
    cache = NS(ready=True, error=None, snapshot=lambda: {"ready": True})
    return serve.Controller(server, prefill, metadata, dspark, serve.profile_settings(profile),
        cache, {"metadata_proof": "proof-sha"}, growth=growth), core


class DecodeProfileTests(unittest.IsolatedAsyncioTestCase):
    def test_profiles_keep_benchmark_growth_disabled_and_cached_enabled(self):
        for name, growth, delay, concurrency in (("benchmark", False, 0.0, 1), ("cached", True, 8.0, 8)):
            with self.subTest(profile=name):
                profile = serve.profile_settings(name)
                self.assertIs(profile["packed_cache_growth"], growth)
                self.assertIs(profile["packed_cache_growth_pingpong"], True)
                self.assertEqual(profile["preserved_prefill_max_defer_seconds"], delay)
                self.assertIs(profile["terminal_response_barrier"], True)
                self.assertEqual(profile["max_concurrent_requests"], concurrency)
                self.assertEqual(profile["process_limit_gib"], 240)
                self.assertEqual(profile["wired_limit_mib"], 245760)
                flags = serve.profile_flags(profile)
                self.assertEqual(flags["DS41_GROWTH"], "1" if growth else "0")
                self.assertEqual(flags["DS41_GROWTH_PINGPONG"], "1")

    def test_environment_discards_ambient_growth_and_auth_before_owned_flags(self):
        for name in ("benchmark", "cached"):
            with self.subTest(profile=name):
                profile = serve.profile_settings(name)
                env = serve.runtime_environment({"PATH": "/bin", "DS41_GROWTH": "host-value",
                    "DS41_GROWTH_PINGPONG": "0", "OMLX_API_KEY": "ambient-key",
                    "PYTHONPATH": "/other/source"}, profile, Path("/work"), Path("/proof"))
                self.assertEqual(env["DS41_GROWTH"], "1" if name == "cached" else "0")
                self.assertEqual(env["DS41_GROWTH_PINGPONG"], "1")
                self.assertEqual(env["DS41_PREFIX_CACHE_GIB"], "3" if name == "cached" else "0")
                self.assertEqual(env["PATH"], "/bin")
                self.assertNotIn("OMLX_API_KEY", env)
                self.assertNotIn("PYTHONPATH", env)

    def test_main_applies_flags_before_first_omlx_import_without_opening_a_port(self):
        original_import, original_open = builtins.__import__, Path.open
        for name in ("benchmark", "cached"):
            with self.subTest(profile=name), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                work, model = root / "work", root / "model"
                (work / "source/omlx/omlx").mkdir(parents=True)
                (work / "source/omlx/omlx/__init__.py").write_text("")
                (model / "exact-metadata").mkdir(parents=True)
                (model / "exact-metadata/metadata-proof.json").write_text("{}")
                args = NS(work_dir=work, model_dir=model, profile=name, host="127.0.0.1",
                          port=18099, metadata_proof=None)
                observed, opened = [], []
                def checked_import(module, *a, **kw):
                    if module == "omlx" or module.startswith("omlx."):
                        observed.append(dict(serve.os.environ))
                        raise RuntimeError("stopped_before_model_import")
                    return original_import(module, *a, **kw)
                def tracked_open(path, *a, **kw):
                    stream = original_open(path, *a, **kw)
                    opened.append(stream)
                    return stream
                socket = mock.MagicMock()
                try:
                    with mock.patch.object(serve, "parser", return_value=NS(parse_args=lambda: args)), \
                            mock.patch.object(serve.subprocess, "check_output", side_effect=lambda command:
                                str(256 * 2**30 if command[-1] == "hw.memsize" else 245760).encode()), \
                            mock.patch.object(serve.socket, "socket", return_value=socket), \
                            mock.patch.object(serve.fcntl, "flock"), \
                            mock.patch.object(serve, "Tee", side_effect=lambda stream, logfile: stream), \
                            mock.patch.object(sys, "path", list(sys.path)), \
                            mock.patch.dict(serve.os.environ, {"DS41_GROWTH": "ambient", "PATH": "/bin"}, clear=True), \
                            mock.patch.object(Path, "open", tracked_open), \
                            mock.patch("builtins.__import__", side_effect=checked_import):
                        with self.assertRaisesRegex(RuntimeError, "stopped_before_model_import"):
                            serve.main()
                finally:
                    for stream in opened:
                        stream.close()
                self.assertEqual(len(observed), 1)
                self.assertEqual(observed[0]["DS41_GROWTH"], "1" if name == "cached" else "0")
                self.assertEqual(observed[0]["DS41_GROWTH_PINGPONG"], "1")
                socket.__enter__.return_value.bind.assert_called_once_with(("127.0.0.1", 18099))

    def test_scheduler_uses_profile_window_and_preserves_resource_limits(self):
        for name, delay, concurrency in (("benchmark", 0.0, 1), ("cached", 8.0, 8)):
            with self.subTest(profile=name):
                config = NS(preserved_prefill_max_defer_seconds=999, max_num_seqs=99)
                result = serve.configure_scheduler(config, serve.profile_settings(name))
                self.assertIs(result, config)
                self.assertEqual(config.preserved_prefill_max_defer_seconds, delay)
                self.assertEqual(config.max_num_seqs, concurrency)
                self.assertEqual(config.prefill_step_size, 8192)
                self.assertIsNone(config.paged_ssd_cache_dir)
                self.assertFalse(config.hot_cache_only)
                self.assertEqual(config.hot_cache_max_size, 0)

    def test_snapshot_reports_loaded_values_instead_of_profile_intent(self):
        growth = NS(ENABLED=False, PINGPONG=False)
        controller, core = make_controller(growth=growth)
        core.scheduler.config.preserved_prefill_max_defer_seconds = 2.5
        core.scheduler._preserves_prefill_chunks = lambda: False
        core._terminal_response_barrier_enabled = False
        row = controller.snapshot()
        self.assertTrue(row["profile"]["packed_cache_growth"])
        self.assertEqual(row["decode_runtime"], {"packed_cache_growth": False, "packed_cache_growth_pingpong": False})
        state = row["schedulers"][serve.MODEL_ID]
        self.assertEqual(state["preserved_prefill_max_defer_seconds"], 2.5)
        self.assertFalse(state["preserve_prefill_chunks"])
        self.assertFalse(state["terminal_response_barrier"])
        growth.ENABLED = growth.PINGPONG = True
        core._terminal_response_barrier_enabled = True
        self.assertEqual(controller.snapshot()["decode_runtime"],
                         {"packed_cache_growth": True, "packed_cache_growth_pingpong": True})
        self.assertTrue(controller.snapshot()["schedulers"][serve.MODEL_ID]["terminal_response_barrier"])

    def test_missing_runtime_fields_are_false_or_unknown_and_never_default_true(self):
        controller, _ = make_controller(scheduler=NS(config=NS()), core_flags={})
        row = controller.snapshot()
        self.assertEqual(row["decode_runtime"], {"packed_cache_growth": False, "packed_cache_growth_pingpong": False})
        actual = row["schedulers"][serve.MODEL_ID]
        self.assertFalse(actual["preserve_prefill_chunks"])
        self.assertFalse(actual["terminal_response_barrier"])
        self.assertIsNone(actual["preserved_prefill_max_defer_seconds"])
        controller.server._server_state.engine_pool = None
        self.assertEqual(controller.snapshot()["schedulers"], {})

    def test_failed_capability_probe_does_not_claim_preservation(self):
        scheduler = NS(config=NS(), _preserves_prefill_chunks=mock.Mock(side_effect=RuntimeError("private details")))
        controller, _ = make_controller(scheduler=scheduler)
        row = controller.snapshot()["schedulers"][serve.MODEL_ID]
        self.assertFalse(row["preserve_prefill_chunks"])
        self.assertEqual(row["preserve_prefill_chunks_error"], "RuntimeError")
        self.assertNotIn("private details", json.dumps(row))

    def test_fingerprint_binds_each_policy_and_preserves_other_identities(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, model, config = root / "source", root / "model", root / "config"
            (source / "omlx").mkdir(parents=True)
            model.mkdir()
            config.mkdir()
            (source / "omlx/__init__.py").write_text("# frozen runtime\n")
            proof = root / "proof.json"
            proof.write_text("{}")
            (model / "config.json").write_text("{}")
            (config / "model_settings.json").write_text("{}")
            profile = serve.profile_settings("cached")
            baseline = serve.cache_fingerprint(source, proof, model, config, profile)
            self.assertEqual(baseline["decode_policy"], serve.decode_policy(profile))
            changes = {"packed_cache_growth": False, "packed_cache_growth_pingpong": False,
                       "preserved_prefill_max_defer_seconds": 0, "terminal_response_barrier": False,
                       "prefill_step_size": 4096}
            for key, value in changes.items():
                with self.subTest(policy=key):
                    current = serve.cache_fingerprint(source, proof, model, config, {**profile, key: value})
                    self.assertNotEqual(hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest(),
                                        hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest())
                    self.assertEqual({k: v for k, v in current.items() if k != "decode_policy"},
                                     {k: v for k, v in baseline.items() if k != "decode_policy"})

    def test_new_cache_namespace_leaves_previous_generation_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            old = work / "cache/exact-prefix-v1/checkpoint.bin"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"previous valid generation")
            config = serve.cache_settings(work, serve.profile_settings("cached"))
            self.assertEqual(Path(config["directory"]), work / "cache/exact-prefix-v2-decode-fixes")
            self.assertEqual(config["capacity_gib"], 3)
            self.assertEqual(config["save_interval_seconds"], 300)
            self.assertTrue(config["enabled"] and config["persistence_enabled"])
            self.assertEqual(old.read_bytes(), b"previous valid generation")
            benchmark = serve.cache_settings(work, serve.profile_settings("benchmark"))
            self.assertFalse(benchmark["enabled"] or benchmark["persistence_enabled"])

    def test_post_permissions_and_allowed_fields_are_unchanged(self):
        controller, core = make_controller("benchmark", growth=NS(ENABLED=False, PINGPONG=True))
        for payload in ({"growth": True}, {"packed_cache_growth": True},
                        {"terminal_response_barrier": False}, {"preserved_prefill_max_defer_seconds": 8.0}):
            with self.subTest(payload=payload):
                self.assertEqual(controller.apply(payload)[0], 400)
        self.assertEqual(controller.apply({"metadata_prefill": False, "reset_stats": True})[0], 200)
        self.assertFalse(controller.prefill.ENABLED)
        self.assertFalse(controller.growth.ENABLED)
        self.assertTrue(core._terminal_response_barrier_enabled)
        controller.inflight = 1
        self.assertEqual(controller.apply({"metadata_prefill": True})[0], 409)
        for payload in ({}, {"metadata_prefill": False}, {"reset_stats": True}):
            self.assertEqual(make_controller("cached")[0].apply(payload)[0], 403)

    def test_new_diagnostics_keep_historical_benchmark_validation_compatible(self):
        controller, _ = make_controller("benchmark", growth=NS(ENABLED=False, PINGPONG=True))
        state = controller.snapshot()
        state["source_unchanged"] = True
        self.assertEqual(state["runtime_flags"], {"gate_up": False, "dspark_async": True})
        bench.validate_server(state, "benchmark")


class DecodeDiagnosticAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_authorized_get_returns_actual_decode_state_without_new_post_controls(self):
        controller, _ = make_controller(growth=NS(ENABLED=True, PINGPONG=True))
        async def unused(*args):
            raise AssertionError("No downstream HTTP or inference is allowed")
        middleware = serve.ControlMiddleware(unused, controller, "local-test-key")
        messages = []
        async def send(message):
            messages.append(message)
        scope = dict(type="http", path=serve.CONTROL_PATH, method="GET", client=("127.0.0.1", 1),
                     query_string=b"", headers=[(b"authorization", b"Bearer local-test-key")])
        await middleware(scope, unused, send)
        self.assertEqual(messages[0]["status"], 200)
        result = json.loads(messages[1]["body"])
        self.assertTrue(result["decode_runtime"]["packed_cache_growth"])
        self.assertTrue(result["schedulers"][serve.MODEL_ID]["terminal_response_barrier"])
        messages.clear()
        scope["method"] = "POST"
        async def receive():
            return {"type": "http.request", "body": b'{"metadata_prefill":false}', "more_body": False}
        await middleware(scope, receive, send)
        self.assertEqual(messages[0]["status"], 403)


if __name__ == "__main__":
    unittest.main()
