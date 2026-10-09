#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a hardlinked HF staging directory without converting any weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from deepseek_helpers.model_package import package_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = package_plan(args.checkpoint, args.metadata_dir)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Preserve the existing stage: {output}")
    if args.dry_run:
        print(json.dumps(plan["manifest"], indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    device = output.parent.stat().st_dev
    if any(p.stat().st_dev != device for p in plan["links"].values()):
        raise ValueError("Hardlink staging must be on the same filesystem as every source")
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".assembling-", dir=output.parent))
    try:
        for name, source in plan["links"].items():
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
        for name, raw in plan["generated"].items():
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        # Keep local absolute source paths out of the uploaded public manifest.
        manifest = {**plan["manifest"], "files": [
            {k: v for k, v in f.items() if k != "source"} for f in plan["manifest"]["files"]]}
        (stage / "allowlist.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(json.dumps({"output_dir": str(output), "file_count": manifest["file_count"],
                      "total_bytes": manifest["total_bytes"], "hashed": False}, indent=2))


if __name__ == "__main__":
    main()
