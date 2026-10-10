#!/usr/bin/env python3
"""Prepare, download, verify, serve, and measure the pinned patch releases."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_REPO = "gnahzhang/DeepSeek-V4.1-Flash-LSQ3-G128-MLX"
QWEN_REPO = "mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp"
QWEN_REVISION = "e171af86f499f1855b0fb71d781105e8dd609610"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(directory: Path) -> int:
    directory = directory.resolve()
    sums = directory / "SHA256SUMS"
    if not sums.is_file():
        raise ValueError("SHA256SUMS is missing; use the published DeepSeek release directory")
    seen: set[str] = set()
    total = 0
    for line in sums.read_text().splitlines():
        if not line:
            continue
        checksum, separator, name = line.partition("  ")
        path = PurePosixPath(name)
        if (not separator or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum)
                or not name or str(path) != name or path.is_absolute() or ".." in path.parts
                or "\\" in name or name in seen):
            raise ValueError("Invalid or duplicate checksum manifest entry")
        seen.add(name)
        target = (directory / name).resolve()
        if not target.is_relative_to(directory) or not target.is_file():
            raise ValueError(f"Missing or escaping release file: {name}")
        if file_sha256(target) != checksum:
            raise ValueError(f"Checksum mismatch: {name}")
        total += 1
        print(f"verified {total}: {name}", flush=True)
    if total == 0:
        raise ValueError("Empty checksum manifest")
    print(json.dumps({"status": "passed", "files": total, "manifest_sha256": file_sha256(sums)}))
    return 0


def release_revision() -> str:
    release = ROOT / "profiles" / "release.json"
    if not release.is_file():
        raise ValueError("Release metadata is missing; specify --revision for an unpublished staging build")
    data = json.loads(release.read_text())
    model = data.get("deepseek", {})
    revision = model.get("revision")
    if model.get("verification_status") != "passed" or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("DeepSeek release is not verified; specify an explicit --revision")
    return revision


def command_for(args: argparse.Namespace) -> list[str]:
    python = sys.executable
    script = ROOT / "scripts"
    if args.action == "prepare":
        uv = shutil.which("uv")
        if not uv:
            raise ValueError("uv is required to prepare the pinned Python environments")
        version = "3.13.15" if args.model == "deepseek" else "3.12.14"
        prefix = [uv, "run", "--no-project", "--python", version, "python"]
        if args.model == "deepseek":
            cmd = prefix + [str(script / "prepare_deepseek.py"), "--work-dir", str(args.work_dir)]
        else:
            cmd = prefix + [str(script / "prepare_qwen.py"), "--workdir", str(args.work_dir)]
    elif args.action == "download":
        hf = shutil.which("hf") or "hf"
        repo = DEEPSEEK_REPO if args.model == "deepseek" else QWEN_REPO
        revision = args.revision or (release_revision() if args.model == "deepseek" else QWEN_REVISION)
        cmd = [hf, "download", repo, "--revision", revision, "--local-dir", str(args.model_dir)]
    elif args.action == "serve":
        if args.model == "deepseek":
            cmd = [str(args.work_dir / "venv/bin/python"), str(script / "serve_deepseek.py"), "--work-dir", str(args.work_dir),
                   "--model-dir", str(args.model_dir), "--profile", args.profile]
        else:
            cmd = [python, str(script / "serve_qwen.py"), "--workdir", str(args.work_dir),
                   "--model-dir", str(args.model_dir), "--mode", args.profile,
                   "--variant", "baseline" if args.profile == "benchmark" else "optimized",
                   "--profile", "fixed3" if args.profile == "benchmark" else "fixed3lookup"]
        if args.port is not None:
            cmd += ["--port", str(args.port)]
    elif args.action == "benchmark":
        if args.model == "deepseek":
            cmd = [str(args.work_dir / "venv/bin/python"), str(script / "benchmark_deepseek.py"), "--work-dir", str(args.work_dir),
                   "--model-dir", str(args.model_dir), "--profile", args.profile,
                   "--suite", args.suite or ("prefix" if args.profile == "cached" else "all")]
        else:
            cmd = [python, str(script / "benchmark_qwen.py"), "--workdir", str(args.work_dir),
                   "--model-dir", str(args.model_dir), "--suite", args.suite or "all"]
        cmd += ["--output-dir", str(args.output_dir or args.work_dir / "results")]
        if args.port is not None:
            cmd += ["--port", str(args.port)]
    else:
        raise ValueError(f"Unsupported action: {args.action}")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "download", "verify", "serve", "benchmark"])
    parser.add_argument("--model", choices=["deepseek", "qwen"], required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--profile", choices=["benchmark", "cached"], default="benchmark")
    parser.add_argument("--suite", choices=["core", "prefix", "tasks", "all"])
    parser.add_argument("--port", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    args = parser.parse_args(argv)
    args.work_dir = (args.work_dir or ROOT / "work" / args.model).expanduser().resolve()
    if args.model_dir is not None:
        args.model_dir = args.model_dir.expanduser().resolve()
    if args.action != "prepare" and args.model_dir is None:
        parser.error("--model-dir is required for this action")
    if args.model == "deepseek" and args.suite == "tasks":
        parser.error("--suite tasks is a Qwen task-replay protocol")
    try:
        if args.action == "verify":
            if args.dry_run:
                print(json.dumps({"action": "verify", "model_dir": str(args.model_dir), "model": args.model}))
                return 0
            if args.model == "deepseek":
                return verify_model(args.model_dir)
            hf = shutil.which("hf")
            if not hf:
                raise ValueError("hf is required to verify the upstream Qwen checkpoint")
            revision = args.revision or QWEN_REVISION
            cmd = [hf, "cache", "verify", QWEN_REPO, "--revision", revision,
                   "--local-dir", str(args.model_dir), "--fail-on-missing-files"]
        else:
            cmd = command_for(args)
        if args.dry_run:
            print(json.dumps({"command": cmd}))
            return 0
        environment = None
        if args.model == "deepseek" and args.action in {"serve", "benchmark"}:
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith(("OMLX_", "DS41_", "DS4_", "BASELINE_", "LOCAL_MOE_"))
                           and key not in {"PYTHONPATH", "PYTHONHOME"}}
        return subprocess.call(cmd, env=environment)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
