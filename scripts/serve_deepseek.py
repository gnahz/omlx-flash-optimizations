#!/usr/bin/env python3
"""Isolated localhost DeepSeek service; --help never imports MLX."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import functools
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading

MODEL_ID = "deepseek-v41-lsq"
CONTROL_PATH = "/api/reproduction"
FLAGS = {"DS41_NATIVE_DECODE": "1", "DS41_NATIVE_VERIFY": "0", "DS41_SPARSE": "1",
         "DS41_MHC": "1", "DS41_GATHER": "1", "DS41_NATIVE_GATE_UP": "0",
         "DS41_NATIVE_METADATA_PREFILL": "1", "DS41_DRAFT_ASYNC": "1"}


def isolated_environment(environment):
    return {name: value for name, value in environment.items()
            if not name.startswith(("OMLX_", "DS41_", "DS4_", "BASELINE_", "LOCAL_MOE_"))
            and name not in ("PYTHONPATH", "PYTHONHOME")}


def isolate_cluster_runtime(worker_shim, launch, work):
    shim, reaper = worker_shim.ensure_cluster_python_shim, launch.reap_orphaned_launches
    def ensure_shim(*, home=None, **kwargs):
        return shim(home=work / "home", **kwargs)
    def reap_owned(state_dir=None, **kwargs):
        return reaper(state_dir=work / "cluster/runtime", **kwargs)
    worker_shim.ensure_cluster_python_shim = ensure_shim
    launch.reap_orphaned_launches = reap_owned


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def loopback(value):
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def runtime_hash(source):
    files = {str(p.relative_to(source)): sha(p) for p in sorted((source / "omlx").rglob("*.py"))}
    for pattern in ("*.so", "*.dylib", "*.metal", "*.h"):
        files.update({str(p.relative_to(source)): sha(p) for p in
            sorted((source / "omlx/custom_kernels/glm_moe_dsa").rglob(pattern))})
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def profile_settings(name):
    if name not in ("benchmark", "cached"):
        raise ValueError("Unknown profile")
    return dict(profile=name, max_concurrent_requests=1 if name == "benchmark" else 8,
                prefix_cache_enabled=name == "cached", process_limit_gib=240,
                wired_limit_mib=245760, prefill_step_size=8192, gate_up=False,
                dspark_async=True, metadata_prefill=True)


def prepare_config(work, model, name, host, port):
    """Create only service-owned settings; the checkpoint remains read-only."""
    profile = profile_settings(name)
    config = work / "config" / name
    config.mkdir(parents=True, exist_ok=True, mode=0o700)
    models = config / "models"
    models.mkdir(exist_ok=True)
    link = models / MODEL_ID
    if link.is_symlink():
        if link.resolve() != model:
            link.unlink()
    elif link.exists():
        raise FileExistsError("Service model view is not a symlink: " + str(link))
    if not link.exists():
        link.symlink_to(model, target_is_directory=True)
    key = secrets.token_hex(32)
    key_path = config / "api-key.txt"
    key_path.write_text(key + "\n")
    key_path.chmod(0o600)
    settings = {
        "version": 1,
        "server": {"host": host, "port": port, "log_level": "info",
                   "server_aliases": [host, "localhost"], "sse_keepalive_mode": "chunk",
                   "burst_decode_mode": "balanced", "preserve_mid_system_cache": True},
        "model": {"model_dirs": [str(models)], "model_dir": str(models), "model_fallback": False},
        "memory": {"prefill_memory_guard": True, "memory_guard_tier": "custom",
                   "memory_guard_custom_ceiling_gb": 240., "soft_threshold": .97,
                   "hard_threshold": .999, "prefill_safe_zone_ratio": .99,
                   "prefill_min_chunk_tokens": 32},
        "scheduler": {"max_concurrent_requests": profile["max_concurrent_requests"],
                      "chunked_prefill": False, "prefill_priority": "context", "decode_fairness": True},
        "cache": {"enabled": False, "hot_cache_only": False,
                  "ssd_cache_dir": str(work / "cache" / name), "hot_cache_max_size": "512MB"},
        "auth": {"api_key": key, "secret_key": secrets.token_hex(32),
                 "skip_api_key_verification": False, "allow_unauthenticated_inference": False},
        "sampling": {"max_context_window": 32768, "max_tokens": 16384, "temperature": 1.,
                     "top_p": .95, "top_k": 0, "repetition_penalty": 1.},
    }
    model_settings = {"version": 1, "models": {MODEL_ID: {
        "model_alias": MODEL_ID, "model_type_override": "vlm", "max_context_window": 1048576,
        "max_tokens": 16384, "force_sampling": False, "is_pinned": True, "is_default": True,
        "deepseek_v41_engram_ssd_offload": True, "deepseek_v41_ced_prefill_enabled": True,
        "moe_expert_offload_enabled": False, "moe_expert_offload_resident_fraction": 1.,
        "turboquant_kv_enabled": False, "qwen35_oq_a8_enabled": False,
        "thinking_budget_enabled": False, "mtp_enabled": True,
        "vlm_mtp_enabled": False, "mtp_num_draft_tokens": 4, "trust_remote_code": False,
    }}}
    for filename, value in (("settings.json", settings), ("model_settings.json", model_settings)):
        path = config / filename
        path.write_text(json.dumps(value, indent=2) + "\n")
        path.chmod(0o600)
    return config, key, profile


def idle_snapshot(state, inflight):
    counts = dict(inflight_http=inflight, models_loading=0, leases=0, scheduler_work=0)
    pool = getattr(state, "engine_pool", None)
    if pool is None:
        return {**counts, "pool_uninitialized": 1}
    for entry in pool._entries.values():
        counts["models_loading"] += int(bool(entry.is_loading))
        counts["leases"] += int(entry.in_use)
        counts["scheduler_work"] += int(bool(pool._entry_has_scheduler_work(entry)))
    return counts


class Controller:
    def __init__(self, server, prefill, metadata, dspark, profile, cache, fingerprint,
                 gate=None, source=None, proof=None):
        self.server, self.prefill, self.dspark = server, prefill, dspark
        self.profile, self.cache, self.fingerprint = profile, cache, fingerprint
        self.gate, self.source, self.proof = gate, source, proof
        self.inflight = 0
        self.admission = asyncio.Lock()
        self.stat_lock = threading.Lock()
        self.stats = {}
        for module, method, label in ((prefill, "forward", "metadata_prefill_direct_calls"),
                (metadata, "_dense_prefill_moe", "metadata_prefill_dense_calls"),
                (metadata, "_dense_expert", "metadata_dense_expert_calls"),
                (dspark, "proposal_forward", "dspark_single_calls"),
                (dspark, "proposal_forward_batch", "dspark_batch_calls")):
            self.install_counter(module, method, label)

    def install_counter(self, module, method, label):
        original = getattr(module, method)
        @functools.wraps(original)
        def counted(*args, **kwargs):
            with self.stat_lock:
                self.stats[label] = self.stats.get(label, 0) + 1
            return original(*args, **kwargs)
        setattr(module, method, counted)

    def snapshot(self, check_sources=False):
        with self.stat_lock:
            stats = dict(self.stats)
        result = dict(pid=os.getpid(), ready=self.cache.ready, error=self.cache.error,
                    profile=self.profile, controls={"metadata_prefill": self.prefill.ENABLED},
                    runtime_flags={"gate_up": bool(self.gate and self.gate.ENABLED),
                                   "dspark_async": self.dspark.DRAFT_ASYNC},
                    idle=idle_snapshot(self.server._server_state, self.inflight), stats=stats,
                    fingerprint=self.fingerprint, cache=self.cache.snapshot())
        if check_sources:
            result["source_unchanged"] = (runtime_hash(self.source) == self.fingerprint["runtime"]
                and sha(self.proof) == self.fingerprint["metadata_proof"])
        return result

    def apply(self, value):
        if self.profile["profile"] != "benchmark":
            return 403, {"error": "Controls require the uncached benchmark profile"}
        if not isinstance(value, dict) or set(value) - {"metadata_prefill", "reset_stats"}:
            return 400, {"error": "Unknown benchmark control"}
        if any(type(v) is not bool for v in value.values()):
            return 400, {"error": "Controls must be booleans"}
        if not self.cache.ready or any(idle_snapshot(self.server._server_state, self.inflight).values()):
            return 409, {"error": "Model must be ready and idle"}
        if "metadata_prefill" in value:
            self.prefill.ENABLED = value["metadata_prefill"]
        if value.get("reset_stats"):
            with self.stat_lock:
                self.stats.clear()
        return 200, self.snapshot()


class ControlMiddleware:
    """Serialize control admission against the complete SSE request lifetime."""
    def __init__(self, app, controller, key):
        self.app, self.controller, self.key = app, controller, key

    async def response(self, send, status, value):
        body = json.dumps(value).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        controller = self.controller
        if scope["path"] == CONTROL_PATH:
            values = [v for k, v in scope.get("headers", []) if k.lower() == b"authorization"]
            peer = scope.get("client")
            if (not peer or not loopback(peer[0]) or len(values) != 1
                    or not hmac.compare_digest(values[0], ("Bearer " + self.key).encode())):
                return await self.response(send, 401, {"error": "Local authorization required"})
            if scope["method"] == "GET":
                return await self.response(send, 200, controller.snapshot(
                    scope.get("query_string") == b"verify_sources=1"))
            if scope["method"] != "POST":
                return await self.response(send, 405, {"error": "Use GET or POST"})
            raw = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                raw.extend(message.get("body", b""))
                if len(raw) > 4096:
                    return await self.response(send, 413, {"error": "Control payload too large"})
                if not message.get("more_body"):
                    break
            try:
                value = json.loads(raw)
            except (ValueError, TypeError):
                return await self.response(send, 400, {"error": "Invalid JSON"})
            async with controller.admission:
                status, result = controller.apply(value)
            return await self.response(send, status, result)
        async with controller.admission:
            controller.inflight += 1
        try:
            await self.app(scope, receive, send)
        finally:
            async with controller.admission:
                controller.inflight -= 1


class Tee:
    def __init__(self, stream, logfile):
        self.stream, self.logfile = stream, logfile
    def write(self, value):
        self.logfile.write(value)
        return self.stream.write(value)
    def flush(self):
        self.logfile.flush()
        self.stream.flush()
    def isatty(self):
        return False


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--model-dir", "--checkpoint", dest="model_dir", type=Path, required=True)
    p.add_argument("--profile", choices=("benchmark", "cached"), default="benchmark")
    p.add_argument("--metadata-proof", type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8011)
    return p


def main():
    args = parser().parse_args()
    if args.host != "127.0.0.1" or not 1024 <= args.port <= 65535:
        raise ValueError("Use 127.0.0.1 and a port between 1024 and 65535")
    work, model = args.work_dir.resolve(), args.model_dir.resolve(strict=True)
    source = work / "source/omlx"
    if not (source / "omlx/__init__.py").is_file():
        raise ValueError("Run prepare before serve")
    proof = (args.metadata_proof or model / "exact-metadata/metadata-proof.json").resolve(strict=True)
    physical = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
    wired = int(subprocess.check_output(["sysctl", "-n", "iogpu.wired_limit_mb"]))
    if physical < 256 * 2**30 or wired < 245760:
        raise RuntimeError("The published profile requires 256 GiB physical and 240 GiB wired capacity")
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lock = (work / "service.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((args.host, args.port))
    config, key, profile = prepare_config(work, model, args.profile, args.host, args.port)
    log_path = logs / ("deepseek-" + args.profile + ".log")
    logfile = log_path.open("a", buffering=1)
    log_path.chmod(0o600)
    sys.stdout, sys.stderr = Tee(sys.stdout, logfile), Tee(sys.stderr, logfile)
    clean = isolated_environment(os.environ)
    os.environ.clear()
    os.environ.update(clean)
    os.environ.update(FLAGS, DS41_NATIVE_METADATA=str(proof),
                      DS41_PREFIX_CACHE_GIB="3" if args.profile == "cached" else "0",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      XDG_CACHE_HOME=str(work / "cache/xdg"), HF_HOME=str(work / "cache/huggingface"))
    sys.path.insert(0, str(source))
    import omlx
    if Path(omlx.__file__).resolve().parent != source / "omlx":
        raise RuntimeError("oMLX was imported from the wrong prepared source")
    from omlx import server, process_memory_enforcer as policy
    from omlx.cli import main as cli_main
    from omlx.cluster import worker_shim, launch as cluster_launch
    from omlx.scheduler import SchedulerConfig
    from omlx.custom_kernels.glm_moe_dsa import fast
    from omlx.patches.deepseek_v41 import dspark, native_gate_up, native_metadata, native_metadata_prefill
    from deepseek_helpers.cache_runtime import CacheRuntime
    required = ("deepseek_v41_grouped_expert", "deepseek_v41_packed_attention",
                "deepseek_mxfp4_gather_qmm_pair_concat_blocks", "deepseek_affine_gather_qmm_pair_concat_blocks",
                native_metadata_prefill.SYMBOL)
    if not fast.is_native_available() or fast.missing_symbols(required):
        raise RuntimeError("Required Metal kernels unavailable; slow fallback is forbidden")
    dspark.DRAFT_ASYNC, native_gate_up.ENABLED, native_metadata_prefill.ENABLED = True, False, True
    policy._PREFILL_HEADROOM_SAFETY["custom"] = .995
    policy._PREFILL_ABORT_MARGIN["custom"] = .995
    isolate_cluster_runtime(worker_shim, cluster_launch, work)
    initialize = server.init_server
    def init(*a, **kw):
        scheduler = kw.get("scheduler_config") or SchedulerConfig()
        scheduler.prefill_step_size, scheduler.max_num_seqs = 8192, profile["max_concurrent_requests"]
        scheduler.prefill_speed_priority = False
        scheduler.paged_ssd_cache_dir = None
        scheduler.hot_cache_only, scheduler.hot_cache_max_size = False, 0
        kw["scheduler_config"] = scheduler
        return initialize(*a, **kw)
    server.init_server = init
    fingerprint = dict(runtime=runtime_hash(source),
                       metadata_proof=sha(proof), chunk_tokens=8192, dspark_async=True,
                       model_settings=sha(config / "model_settings.json"),
                       model_files={name: sha(model / name) for name in
                           ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")
                           if (model / name).exists()})
    cache_config = dict(enabled=args.profile == "cached", persistence_enabled=args.profile == "cached",
                        directory=str(work / "cache/exact-prefix-v1"), capacity_gib=3,
                        save_interval_seconds=300, startup_timeout_seconds=900, shutdown_timeout_seconds=120)
    cache = CacheRuntime(server, cache_config, fingerprint, logs / "cache-shutdown.json").install()
    controller = Controller(server, native_metadata_prefill, native_metadata, dspark, profile, cache, fingerprint,
                            gate=native_gate_up, source=source, proof=proof)
    server.app.add_middleware(ControlMiddleware, controller=controller, key=key)
    (config / "launch.json").write_text(json.dumps(dict(profile=profile, flags=FLAGS,
        fingerprint=fingerprint, host=args.host, port=args.port, log=str(log_path)), indent=2) + "\n")
    print(json.dumps(dict(url=f"http://{args.host}:{args.port}/v1", model=MODEL_ID,
                         key_file=str(config / "api-key.txt"), profile=profile, log=str(log_path))), flush=True)
    sys.argv = ["omlx", "serve", "--model-dir", str(config / "models"), "--base-path", str(config),
                "--host", args.host, "--port", str(args.port), "--max-concurrent-requests",
                str(profile["max_concurrent_requests"]), "--memory-guard-gb", "240", "--no-cache"]
    cli_main()


if __name__ == "__main__":
    main()
