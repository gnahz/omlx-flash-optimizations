# SPDX-License-Identifier: Apache-2.0
"""Assemble unchanged model files and existing exact dictionaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

SIDECAR = "DeepSeek-V4.1-LSQ3-G128-exact-metadata.json"
MODEL_FILES = (
    "LICENSE", "config.json", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "chat_template.jinja", "ds41-conversion.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError(f"Expected a relative release path: {name!r}")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing file or path escapes release directory: {name}")
    return path


def header_pin(path: Path) -> dict:
    with path.open("rb") as stream:
        size = path.stat().st_size
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated tensor header: {path.name}")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(1024**2, size - 8):
            raise ValueError(f"Invalid tensor header: {path.name}")
        raw = prefix + stream.read(length)
        if len(raw) != length + 8 or not isinstance(json.loads(raw[8:]), dict):
            raise ValueError(f"Invalid tensor header: {path.name}")
    return dict(size=size, header_bytes=len(raw), header_sha256=hashlib.sha256(raw).hexdigest())


def package_plan(checkpoint: Path, metadata_dir: Path) -> dict:
    checkpoint, metadata_dir = checkpoint.resolve(), metadata_dir.resolve()
    config = json.loads(contained(checkpoint, "config.json").read_text())
    index = json.loads(contained(checkpoint, "model.safetensors.index.json").read_text())
    spec = config.get("omlx_deepseek_v41", {})
    if spec.get("version") != 1 or spec.get("preserve_mtp") is not True:
        raise ValueError("Require the full converted checkpoint with preserved native MTP")
    mapping = index["weight_map"]
    weights = set(mapping.values())
    for table in spec.get("engram_tables", {}).values():
        for field in ("weight_file", "scale_file"):
            if table.get(field):
                weights.add(table[field])
    if not spec.get("engram_tables"):
        raise ValueError("Require complete original SSD Engram tables")
    proof_path = contained(metadata_dir, "metadata-proof.json")
    proof = json.loads(proof_path.read_text())
    original_sidecar = json.loads(contained(metadata_dir, SIDECAR).read_text())
    if (proof.get("status") != "passed" or proof.get("original_tensors_retained") is not True
            or proof.get("quantization_codes_changed") is not False
            or original_sidecar.get("requantized") is not False):
        raise ValueError("Require an existing verified lossless metadata proof")
    for field, filename in (("source_config_sha256", "config.json"),
                            ("source_index_sha256", "model.safetensors.index.json")):
        if original_sidecar.get(field) != sha256(checkpoint / filename):
            raise ValueError(f"Metadata source identity mismatch: {filename}")
    records = proof.get("projections", [])
    expected = {f"language_model.layers.{i}.ffn.experts.{p}"
                for i in range(40) for p in ("w1", "w2", "w3")}
    if len(records) != 120 or {r.get("prefix") for r in records} != expected:
        raise ValueError("Require all 120 exact routed-expert metadata projections")
    if (proof.get("exact_dictionary_projections") != 120
            or proof.get("saved_resident_bytes") != sum(r["saved_bytes"] for r in records)):
        raise ValueError("Metadata proof total mismatch")
    pins = original_sidecar.get("files", [])
    original_pins, encoded_pins = {}, {}
    for pin in pins:
        name = pin["path"]
        # Exported packages and the original local cache use different source roots.
        if name.startswith("../"):
            parts = Path(name).parts[1:]
            if parts and parts[0] == checkpoint.name:
                parts = parts[1:]
            relative = Path(*parts).as_posix()
            original_pins[relative] = pin
        else:
            encoded_pins[name] = pin
    if set(original_pins) != weights:
        raise ValueError("Metadata sidecar must pin every original tensor file, including Engram")
    links, portable_pins = {}, []
    for name in sorted(weights):
        path = contained(checkpoint, name)
        pin = header_pin(path)
        if any(original_pins[name].get(key) != value for key, value in pin.items()):
            raise ValueError(f"Original tensor header changed: {name}")
        links[name] = path
        portable_pins.append(dict(path="../" + name, **pin))
    encoded_names = set()
    for record in records:
        name = record["file"]
        if (Path(name).name != name or name in encoded_names
                or record.get("encoded") is not True
                or record.get("all_words_roundtrip_exact") is not True
                or record.get("full_encoded_readback_passed") is not True):
            raise ValueError("Invalid or unverified dictionary projection")
        encoded_names.add(name)
        path = contained(metadata_dir, name)
        pin = header_pin(path)
        if any(encoded_pins.get(name, {}).get(key) != value for key, value in pin.items()):
            raise ValueError(f"Encoded tensor header changed: {name}")
        links["exact-metadata/" + name] = path
        portable_pins.append(dict(path=name, **pin))
    if set(encoded_pins) != encoded_names:
        raise ValueError("Encoded file pins differ from verified projections")
    for name in MODEL_FILES:
        links[name] = contained(checkpoint, name)
    portable_sidecar = dict(format_version=2, requantized=False,
        source_config_sha256=original_sidecar["source_config_sha256"],
        source_index_sha256=original_sidecar["source_index_sha256"], files=portable_pins)
    # Numeric proof is unchanged; remove historical machine-local index paths.
    portable_proof = {k: v for k, v in proof.items() if k not in ("original_index", "external_index")}
    portable_proof["source_proof_sha256"] = sha256(proof_path)
    generated = {
        "exact-metadata/" + SIDECAR: (json.dumps(portable_sidecar, indent=2) + "\n").encode(),
        "exact-metadata/metadata-proof.json": (json.dumps(portable_proof, indent=2) + "\n").encode(),
    }
    files = [dict(path=name, size=path.stat().st_size, source=str(path), method="hardlink")
             for name, path in sorted(links.items())]
    files.extend(dict(path=name, size=len(raw), method="generated") for name, raw in sorted(generated.items()))
    return dict(links=links, generated=generated,
        manifest=dict(version=1, files=files, file_count=len(files),
                      total_bytes=sum(f["size"] for f in files),
                      weights_requantized=False, original_tensors_retained=True,
                      full_file_hashes_computed=False, saved_resident_bytes=proof["saved_resident_bytes"]))
