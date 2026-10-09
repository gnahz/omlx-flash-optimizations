#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rebuild pinned DeepSeek MLX/oMLX sources in an isolated directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/deepseek"
HELPER_WHEEL = {
    "url": "https://files.pythonhosted.org/packages/dd/cd/4e50bf325100e7165e13d025f264362bf0009196269f9eaf87f2c6e738a2/mlx_metal-0.32.2-py3-none-macosx_26_0_arm64.whl",
    "sha256": "e6abeac9ac5265830c9c1541b6f96e9be37a85c2446763a46ad466c63a3837ab",
    "metallib_sha256": "dc59d1cceb1a5c7e578232e6e41e28e2c73c9463ac6dbc3886c3ee17ffc270ed",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def run(args, *, cwd=None, env=None, capture=False):
    print("+ " + shlex.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True,
                          text=True, capture_output=capture)


def source_receipt(path, spec):
    names = subprocess.check_output(["git", "-C", str(path), "ls-files", "-z",
                                     "--cached", "--others", "--exclude-standard"]).decode().split("\0")
    return dict(revision=spec["revision"], patches=spec["patches"],
                files={name: digest(path / name) for name in sorted(set(names))
                       if name and (path / name).is_file()})


def validate_sources(work, manifest):
    receipt = json.loads((work / "source-receipt.json").read_text())
    for name in ("mlx", "omlx"):
        path = work / "source" / name
        head = run(["git", "-C", path, "rev-parse", "HEAD"], capture=True).stdout.strip()
        if head != manifest[name]["revision"]:
            raise RuntimeError(f"Wrong {name} upstream revision")
        current = source_receipt(path, manifest[name])
        if current != receipt[name]:
            raise RuntimeError(f"Patched {name} source differs from its preparation receipt")
    build_receipt = work / "build-receipt.json"
    if build_receipt.exists():
        built = json.loads(build_receipt.read_text())
        if built["prepared_sources_sha256"] != digest(work / "source-receipt.json"):
            raise RuntimeError("Build receipt refers to another prepared source")
        for relative, expected in built["binaries"].items():
            if digest(work / relative) != expected:
                raise RuntimeError(f"Built native binary checksum mismatch: {relative}")
    print("Pinned patched source receipt verified.")


def checkout(work, name, spec, local_source=None):
    target = work / "source" / name
    if target.exists():
        raise FileExistsError(f"Refuse to replace an existing checkout: {target}; use --check")
    run(["git", "clone", "--quiet", str(local_source or spec["url"]), target])
    run(["git", "-C", target, "checkout", "--quiet", "--detach", spec["revision"]])
    for relative in spec["patches"]:
        patch = ROOT / relative
        run(["git", "-C", target, "apply", "--check", patch])
        run(["git", "-C", target, "apply", patch])
    return target


def helper_asset(work):
    assets = work / "assets"
    assets.mkdir(exist_ok=True)
    wheel = assets / "mlx_metal-0.32.2-macosx26.whl"
    if not wheel.exists():
        temporary = wheel.with_suffix(".download")
        urllib.request.urlretrieve(HELPER_WHEEL["url"], temporary)
        if digest(temporary) != HELPER_WHEEL["sha256"]:
            temporary.unlink()
            raise RuntimeError("Official helper wheel checksum mismatch")
        temporary.rename(wheel)
    if digest(wheel) != HELPER_WHEEL["sha256"]:
        raise RuntimeError("Cached helper wheel checksum mismatch")
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if n.endswith("/mlx.metallib")]
        if len(names) != 1:
            raise RuntimeError("Official wheel must contain one MLX auxiliary library")
        raw = archive.read(names[0])
    if hashlib.sha256(raw).hexdigest() != HELPER_WHEEL["metallib_sha256"]:
        raise RuntimeError("Official auxiliary metallib checksum mismatch")
    target = assets / "stock-0.32.2-auxiliary.metallib"
    target.write_bytes(raw)
    return target


def build(work, manifest, python, jobs):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("The qualified JIT build requires arm64 macOS")
    version = run([python, "-c", "import sys;print('.'.join(map(str,sys.version_info[:2])))"], capture=True).stdout.strip()
    if version != manifest["python"]:
        raise RuntimeError("Use Python 3.13 for the qualified extension ABI")
    venv = work / "venv"
    if not venv.exists():
        run([python, "-m", "venv", venv])
    py = venv / "bin/python"
    env = os.environ.copy()
    env["PATH"] = str(venv / "bin") + os.pathsep + env.get("PATH", "")
    (work / "tmp").mkdir(exist_ok=True)
    env["TMPDIR"] = str(work / "tmp")
    env["PIP_CACHE_DIR"] = str(work / "cache/pip")
    # Install bootstrap build packages first, then frozen runtime without resolution.
    run([py, "-m", "pip", "install", "setuptools==84.0.0", "wheel==0.48.0", "cmake==4.1.0"], env=env)
    run([py, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
         "-r", PROFILE / "requirements.lock.txt"], env=env)
    mlx, omlx = work / "source/mlx", work / "source/omlx"
    asset = helper_asset(work)
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)
    env["MLX_PREAMBLE_PYTHON"] = str(py)
    env["CMAKE_ARGS"] = shlex.join([
        "-DMLX_METAL_JIT=ON", "-DCMAKE_OSX_DEPLOYMENT_TARGET=" + manifest["deployment_target"],
        "-DMLX_AUXILIARY_METALLIB=" + str(asset), "-DPython_EXECUTABLE=" + str(py)])
    run([py, "-m", "pip", "install", "--no-deps", "--no-build-isolation", mlx], env=env)
    # Keep custom kernel nanobind in the same domain/version as fetched MLX v3.0.1.
    nanobind = list(mlx.glob("build/**/_deps/nanobind-src/cmake/nanobind-config.cmake"))
    if len(nanobind) != 1:
        raise RuntimeError("Could not identify the fetched MLX nanobind 3.0.1 CMake package")
    cmake = venv / "bin/cmake"
    kernel_dir = omlx / "omlx/custom_kernels/glm_moe_dsa"
    run([cmake, "-S", kernel_dir / "csrc", "-B", work / "build/glm",
         "-DPython_EXECUTABLE=" + str(py), "-Dnanobind_ROOT=" + str(nanobind[0].parent),
         "-DBASELINE_RUNTIME_METAL=ON", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_OSX_DEPLOYMENT_TARGET=" + manifest["deployment_target"],
         "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(kernel_dir)], env=env)
    run([cmake, "--build", work / "build/glm", "-j", jobs], env=env)
    run([py, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", omlx], env=env)
    run([py, "-m", "pip", "check"], env=env)
    # Import-only verification, no model load and no inference.
    run([py, "-c", "from omlx.custom_kernels.glm_moe_dsa import fast; "
         "assert fast.is_native_available(); "
         "assert not fast.missing_symbols(('deepseek_v41_joint_metadata_gather_qmm',"
         "'deepseek_v41_grouped_expert','deepseek_v41_packed_attention'));print('Native extension symbols verified')"], env=env)
    binaries = list(kernel_dir.glob("_ext*.so")) + list(kernel_dir.glob("*kernel_ops*.dylib"))
    installed_versions = json.loads(run([py, "-c", "import importlib.metadata,json; "
        "print(json.dumps({name:importlib.metadata.version(name) "
        "for name in ('mlx','omlx','nanobind')}))"], env=env, capture=True).stdout)
    (work / "build-receipt.json").write_text(json.dumps(dict(
        helper_wheel=HELPER_WHEEL, deployment_target=manifest["deployment_target"],
        binaries={str(p.relative_to(work)): digest(p) for p in binaries},
        prepared_sources_sha256=digest(work / "source-receipt.json"),
        source_revisions={name: manifest[name]["revision"] for name in ("mlx", "omlx")},
        installed_versions=installed_versions,
        python=version, model_loaded=False, inference_validated=False), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable if sys.version_info[:2] == (3, 13) else "python3.13")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--mlx-source", type=Path, help="Optional local clone/object source, read only")
    parser.add_argument("--omlx-source", type=Path, help="Optional local clone/object source, read only")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check", action="store_true", help="Verify an existing prepared source receipt")
    mode.add_argument("--apply-only", action="store_true", help="Clone and check/apply patches without a build")
    mode.add_argument("--build-only", action="store_true", help="Validate existing source receipts then build")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 32:
        parser.error("--jobs must be between 1 and 32")
    work = args.work_dir.resolve()
    manifest = json.loads((PROFILE / "sources.json").read_text())
    for relative, expected in manifest["patch_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Release patch checksum mismatch: {relative}")
    if args.dry_run:
        print(json.dumps(dict(work_dir=str(work), sources=manifest,
            helper_wheel=HELPER_WHEEL, modes=["apply-only", "check", "build-only"],
            changes_existing_service=False), indent=2))
        return
    if args.check or args.build_only:
        validate_sources(work, manifest)
        if args.check:
            return
    else:
        work.mkdir(parents=True, exist_ok=True)
        (work / "source").mkdir(exist_ok=True)
        sources = {name: checkout(work, name, manifest[name], getattr(args, name + "_source"))
                   for name in ("mlx", "omlx")}
        receipt = {name: source_receipt(path, manifest[name]) for name, path in sources.items()}
        (work / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not args.apply_only:
        build(work, manifest, args.python, args.jobs)
    print("Preparation complete. Start explicitly with:")
    print(shlex.join([str(work / "venv/bin/python"), str(ROOT / "scripts/serve_deepseek.py"),
                      "--work-dir", str(work), "--model-dir", "/absolute/model/path",
                      "--profile", "benchmark"]))


if __name__ == "__main__":
    main()
