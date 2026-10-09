# SPDX-License-Identifier: Apache-2.0
"""Service-scoped lifecycle for exact CED cache persistence.

All MLX/cache mutations are serialized on the owning engine executor. The SSD
store is read once per model load; requests only use the in-memory manager.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import functools
import json
import logging
import os
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class CacheRuntime:
    def __init__(self, server, config, fingerprint, shutdown_receipt):
        self.server = server
        self.config = config
        self.fingerprint = fingerprint
        self.shutdown_receipt = Path(shutdown_receipt)
        self.models = {}
        self.ready = False
        self.error = None

    async def on_engine(self, core, fn):
        import mlx.core as mx
        def invoke():
            with mx.stream(core._mlx_stream):
                return fn()
        return await asyncio.get_running_loop().run_in_executor(core._mlx_executor, invoke)

    def seed_guard(self, scheduler):
        # Loading engines are not yet published in the pool. Seed their guard
        # with the same live enforcer values before restoring any checkpoint.
        enforcer = self.server._server_state.process_memory_enforcer
        if enforcer is None or not enforcer.prefill_memory_guard:
            raise RuntimeError("Exact-cache startup requires the memory guard")
        breakdown = enforcer._get_ceiling_breakdown()
        reserve = enforcer._hot_cache_reserved_bytes()
        ceiling = enforcer._scheduler_limit_bytes(breakdown["hard_limit"], reserved=reserve)
        scheduler._prefill_memory_guard = True
        scheduler._memory_hard_limit_bytes = ceiling
        scheduler._memory_limit_bytes = int(ceiling * enforcer._soft_threshold)
        scheduler._memory_hard_watermark_bytes = int(ceiling * enforcer._hard_threshold)
        scheduler._memory_abort_limit_bytes = enforcer._scheduler_limit_bytes(
            enforcer._get_abort_limit_bytes(), reserved=reserve)
        scheduler._prefill_abort_margin = enforcer._get_prefill_abort_margin()
        scheduler._memory_limits_propagated = True

    async def initialize(self, core):
        name = core.config.model_name
        entry = self.models[name] = {"state": "restoring", "cache": {}, "completed": []}
        def restore():
            self.seed_guard(core.scheduler)
            manager = core.scheduler._configure_exact_prefix_cache()
            if manager is None:
                raise RuntimeError("DeepSeek exact-prefix manager was not enabled")
            core._service_exact_cache_manager = manager
            if self.config["persistence_enabled"]:
                identity = {"deployment": self.fingerprint, "draft_signature": manager.signature}
                manager.configure_persistence(self.config["directory"], identity,
                    interval_seconds=self.config["save_interval_seconds"])
                manager.restore_persisted()
            entry["cache"] = manager.snapshot_stats()
        await self.on_engine(core, restore)
        entry["state"] = "ready"
        core._service_exact_cache_task = asyncio.create_task(self.periodic(core))

    def manager(self, core):
        manager = core.scheduler._configure_exact_prefix_cache()
        if manager is not core._service_exact_cache_manager:
            raise RuntimeError("Exact-cache draft configuration changed; restart is required")
        return manager

    async def periodic(self, core):
        name = core.config.model_name
        try:
            while True:
                await asyncio.sleep(1)
                def step():
                    scheduler = core.scheduler
                    manager = self.manager(core)
                    if manager is None:
                        return
                    # Re-check on the executor after any preceding request step.
                    if self.config["persistence_enabled"] and not scheduler.has_requests():
                        manager.persistence_step()
                    self.models[name]["cache"] = manager.snapshot_stats()
                    running = len(getattr(scheduler, "running", {}))
                    previous = self.models[name].get("scheduler", {})
                    self.models[name]["scheduler"] = {
                        "max_num_seqs": getattr(getattr(scheduler, "config", None), "max_num_seqs", None),
                        "running_requests": running,
                        "waiting_requests": len(getattr(scheduler, "waiting", [])),
                        "prefilling_requests": len(getattr(scheduler, "prefilling", [])),
                        "max_observed_running": max(running, previous.get("max_observed_running", 0)),
                    }
                    self.models[name]["completed"] = list(
                        getattr(scheduler, "exact_prefix_completed", []))
                await self.on_engine(core, step)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("Exact cache maintenance failed")
            self.models[name]["maintenance_error"] = f"{type(error).__name__}: {error}"
            self.models[name]["state"] = "failed"
            self.ready = False
            self.error = f"{type(error).__name__}: {error}"

    async def flush(self, core, deadline=None):
        task = getattr(core, "_service_exact_cache_task", None)
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        name = core.config.model_name
        if name not in self.models:
            return
        self.ready = False
        self.models[name]["state"] = "saving"
        deadline = deadline or time.monotonic() + self.config["shutdown_timeout_seconds"] - 10
        result = {"pid": os.getpid(), "saved": False, "at": time.time()}
        try:
            while True:
                def step():
                    manager = self.manager(core)
                    pending = manager.persistence_step(force=True) if self.config["persistence_enabled"] else False
                    self.models[name]["cache"] = manager.snapshot_stats()
                    if not pending:
                        persistence = self.models[name]["cache"].get("persistence", {})
                        manager.close_persistence(wait=False)
                        if self.config["persistence_enabled"] and (
                                not persistence.get("enabled") or persistence.get("dirty") or persistence.get("last_error")):
                            raise RuntimeError("Checkpoint save incomplete: " + str(persistence.get("last_error")))
                        manager.clear()
                    return pending
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Exact cache shutdown save exceeded its deadline")
                pending = await asyncio.wait_for(self.on_engine(core, step), timeout=remaining)
                if not pending:
                    break
                await asyncio.sleep(.05)
            result.update(saved=True, cache=self.models[name]["cache"])
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            logger.exception("Exact cache shutdown save failed")
        core._service_exact_cache_manager = None
        self.models[name]["state"] = "stopped"
        self.shutdown_receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.shutdown_receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(self.shutdown_receipt)

    def snapshot(self):
        return {"ready": self.ready, "pid": os.getpid(), "error": self.error,
                "read_policy": "startup_only", "models": self.models}

    def install(self):
        from omlx.engine_core import EngineCore
        from starlette.responses import JSONResponse
        original_start, original_stop = EngineCore.start, EngineCore.stop
        runtime = self

        @functools.wraps(original_start)
        async def start(core):
            if not core.is_running() and getattr(core.model, "model_type", None) == "deepseek_v41":
                if runtime.config["enabled"]:
                    await runtime.initialize(core)
            return await original_start(core)

        @functools.wraps(original_stop)
        async def stop(core):
            deadline = time.monotonic() + runtime.config["shutdown_timeout_seconds"] - 10
            task = getattr(core, "_service_exact_cache_task", None)
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            await original_stop(core)
            if getattr(core, "_service_exact_cache_task", None) is not None:
                await runtime.flush(core, deadline)
                core._service_exact_cache_task = None

        EngineCore.start, EngineCore.stop = start, stop
        original_lifespan = self.server.app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            async with original_lifespan(app):
                async def preload():
                    try:
                        while not runtime.server._server_state.pinned_preload_complete:
                            await asyncio.sleep(.1)
                        await runtime.server._server_state.engine_pool.get_engine("deepseek-v41-lsq")
                        if runtime.config["enabled"] and not any(
                                entry["state"] == "ready" and entry["cache"].get("enabled") and
                                (not runtime.config["persistence_enabled"] or
                                 entry["cache"].get("persistence", {}).get("enabled"))
                                for entry in runtime.models.values()):
                            raise RuntimeError("Model loaded without the required exact-cache runtime")
                        runtime.ready = True
                    except Exception as error:
                        runtime.error = f"{type(error).__name__}: {error}"
                        logger.exception("Exact cache service initialization failed")
                task = asyncio.create_task(preload())
                try:
                    yield
                finally:
                    runtime.ready = False
                    if not task.done():
                        task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
        self.server.app.router.lifespan_context = lifespan

        @self.server.app.middleware("http")
        async def readiness(request, call_next):
            if not runtime.ready and (request.url.path == "/health" or request.url.path in (
                    "/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages")):
                return JSONResponse({"error": {"message": "Model and prefix cache are loading",
                    "type": "service_unavailable"}}, status_code=503)
            return await call_next(request)
        return self
