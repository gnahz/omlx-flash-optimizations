# SPDX-License-Identifier: Apache-2.0
"""Assemble unchanged tensors and exact dictionaries with release metadata."""
from __future__ import annotations

import hashlib
import json
import math
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


def tensor_header(path: Path) -> tuple[dict, dict]:
    """Read only the bounded JSON header; never load tensor payload bytes."""
    with path.open("rb") as stream:
        size = path.stat().st_size
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated tensor header: {path.name}")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(1024**2, size - 8):
            raise ValueError(f"Invalid tensor header: {path.name}")
        raw = prefix + stream.read(length)
        if len(raw) != length + 8:
            raise ValueError(f"Invalid tensor header: {path.name}")
        header = json.loads(raw[8:])
        if not isinstance(header, dict):
            raise ValueError(f"Invalid tensor header: {path.name}")
    return header, dict(size=size, header_bytes=len(raw), header_sha256=hashlib.sha256(raw).hexdigest())


def header_pin(path: Path) -> dict:
    return tensor_header(path)[1]


def logical_parameter_count(checkpoint: Path, config: dict, index: dict) -> int:
    """Count root logical scalars and each configured Engram table once.

    Packed quantized weights are expanded using their stored word width and
    unchanged per-module bitwidth. Their scale/bias arrays and copied auxiliary
    Engram projections do not represent additional logical parameters.
    """
    spec = config["omlx_deepseek_v41"]
    modules = spec["quantized_modules"]
    tables = spec["engram_tables"]
    if not isinstance(modules, dict) or not isinstance(tables, dict):
        raise ValueError("Require per-module quantization and Engram table specifications")
    auxiliary_files = {table[field] for table in tables.values()
                       for field in ("weight_file", "scale_file") if table.get(field)}
    headers, counted = {}, set()

    def entry(filename, name):
        if filename not in headers:
            headers[filename] = tensor_header(contained(checkpoint, filename))[0]
        value = headers[filename].get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Indexed tensor missing from header: {name}")
        shape = value.get("shape")
        if (not isinstance(shape, list)
                or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in shape)):
            raise ValueError(f"Invalid tensor shape: {name}")
        return value

    def count(filename, name, quantization=None):
        identity = (filename, name)
        if identity in counted:
            return 0
        value = entry(filename, name)
        shape = list(value["shape"])
        if quantization is not None:
            bits = quantization.get("bits")
            word_bits = {"U8": 8, "U16": 16, "U32": 32, "U64": 64}.get(value.get("dtype"))
            if (not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0
                    or word_bits is None or bits > word_bits or not shape
                    or shape[-1] * word_bits % bits):
                raise ValueError(f"Invalid packed quantized tensor: {name}")
            shape[-1] = shape[-1] * word_bits // bits
        counted.add(identity)
        return math.prod(shape)

    total = 0
    for name, filename in index["weight_map"].items():
        # Auxiliary shards contain copies of projections already counted at root.
        # Even a previously published index may already contain the table keys.
        if filename in auxiliary_files:
            continue
        entry(filename, name)
        if any(name.endswith(suffix) and name[:-len(suffix)] in modules
               for suffix in (".scales", ".biases", ".scale")):
            continue
        quantization = modules.get(name[:-7]) if name.endswith(".weight") else None
        total += count(filename, name, quantization)
    for module, table in tables.items():
        quantization = table if table.get("bits") is not None else modules.get(module)
        total += count(table["weight_file"], table["weight_key"], quantization)
        if table.get("scale_file"):
            entry(table["scale_file"], table["scale_key"])
    return total


def package_plan(checkpoint: Path, metadata_dir: Path) -> dict:
    checkpoint, metadata_dir = checkpoint.resolve(), metadata_dir.resolve()
    config = json.loads(contained(checkpoint, "config.json").read_text())
    index = json.loads(contained(checkpoint, "model.safetensors.index.json").read_text())
    spec = config.get("omlx_deepseek_v41", {})
    if spec.get("version") != 1 or spec.get("preserve_mtp") is not True:
        raise ValueError("Require the full converted checkpoint with preserved native MTP")
    mapping = dict(index["weight_map"])
    weights = set(mapping.values())
    for table in spec.get("engram_tables", {}).values():
        for field in ("weight_file", "scale_file"):
            if table.get(field):
                weights.add(table[field])
    if not spec.get("engram_tables"):
        raise ValueError("Require complete original SSD Engram tables")
    for table in spec["engram_tables"].values():
        for kind in ("weight", "scale"):
            filename, key = table.get(kind + "_file"), table.get(kind + "_key")
            if kind == "scale" and filename is None and key is None:
                continue
            if not isinstance(filename, str) or not isinstance(key, str) or not key:
                raise ValueError(f"Incomplete Engram {kind} specification")
            if key in mapping and mapping[key] != filename:
                raise ValueError(f"Engram key conflicts with original index: {key}")
            mapping[key] = filename
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
        if name in ("config.json", "model.safetensors.index.json"):
            continue
        links[name] = contained(checkpoint, name)
    # Only release JSON is regenerated; the serving checkpoint stays untouched.
    config["quantization"] = dict(spec["quantized_modules"])
    index["weight_map"] = mapping
    index["metadata"] = {**index.get("metadata", {}),
                         "total_parameters": logical_parameter_count(checkpoint, config, index)}
    config_raw = (json.dumps(config, indent=2) + "\n").encode()
    index_raw = (json.dumps(index, indent=2) + "\n").encode()
    portable_sidecar = dict(format_version=2, requantized=False,
        source_config_sha256=hashlib.sha256(config_raw).hexdigest(),
        source_index_sha256=hashlib.sha256(index_raw).hexdigest(), files=portable_pins)
    # Numeric proof is unchanged; remove historical machine-local index paths.
    portable_proof = {k: v for k, v in proof.items() if k not in ("original_index", "external_index")}
    portable_proof.setdefault("source_proof_sha256", sha256(proof_path))
    generated = {
        "config.json": config_raw,
        "model.safetensors.index.json": index_raw,
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
