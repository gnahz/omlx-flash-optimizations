# SPDX-License-Identifier: Apache-2.0
"""Small real loader/package fixtures; no GPU, full model, or service is used."""
from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from deepseek_helpers.model_package import SIDECAR, header_pin, package_plan, sha256


def native_loader_namespace():
    # Execute the exact published constructor/helpers without importing MLX.
    # Runtime GPU projection methods are defined but never invoked by these tests.
    patch = (ROOT / "patches/deepseek/omlx/0002-exact-metadata-and-direct-prefill.patch").read_text()
    marker = "diff --git a/omlx/patches/deepseek_v41/native_metadata.py "
    section = patch.split(marker, 1)[1].split("\ndiff --git ", 1)[0]
    source = "\n".join(line[1:] for line in section.splitlines()
                       if line.startswith("+") and not line.startswith("+++"))
    # Apply the actual final patch to its literal source, not a mirrored fix.
    portability = ROOT / "patches/deepseek/omlx/0004-portable-release-identities.patch"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "omlx/patches/deepseek_v41/native_metadata.py"
        path.parent.mkdir(parents=True)
        path.write_text(source + "\n")
        subprocess.run(["git", "apply", "--include=omlx/patches/deepseek_v41/native_metadata.py",
                        str(portability)], cwd=directory, check=True, capture_output=True)
        source = path.read_text()
    wanted = {"_sha", "_identity", "_header", "_entry", "MetadataRecord", "NativeMetadataPlan"}
    tree = ast.parse(source)
    nodes = [node for node in tree.body if getattr(node, "name", None) in wanted]
    namespace = dict(dataclass=dataclass, replace=replace, Path=Path, json=json,
                     struct=struct, hashlib=hashlib, math=math,
                     _HEADER_BYTES=1024**2, os=os, _TYPES={"U32": ("uint32", 4), "BF16": ("uint16", 2)})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "published-native-metadata", "exec"), namespace)
    return namespace


def write_tensors(path, tensors):
    header, raw, offset = {}, bytearray(), 0
    for key, (dtype, shape, payload) in tensors.items():
        header[key] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset + len(payload)])
        raw.extend(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw)
    return dict(path=str(path), mtime_ns=path.stat().st_mtime_ns, **header_pin(path))


def fixture(root):
    checkpoint, metadata = root / "checkpoint", root / "old-metadata"
    checkpoint.mkdir()
    metadata.mkdir()
    mapping, modules, records, pins = {}, {}, [], []
    # Raw BF16 words include signed zero, tiny values, infinities and a NaN payload.
    dictionary = sorted([0x00008000, 0x7FC10001, 0xFF807F80, 0x3F803F80])
    indices = list(range(4)) * 8
    codes = struct.pack("<32H", *indices)
    pairs = [dictionary[i] for i in indices]
    scales = struct.pack("<32H", *(pair & 65535 for pair in pairs))
    biases = struct.pack("<32H", *(pair >> 16 for pair in pairs))
    weight = struct.pack("<384I", *range(384))
    table = struct.pack("<4I", *dictionary)
    for layer in range(40):
        for projection in ("w1", "w2", "w3"):
            name = f"language_model.layers.{layer}.ffn.experts.{projection}"
            filename = f"source-{layer:02d}-{projection}.safetensors"
            pin = write_tensors(checkpoint / filename, {
                name + ".weight": ("U32", [2, 8, 24], weight),
                name + ".scales": ("BF16", [2, 8, 2], scales),
                name + ".biases": ("BF16", [2, 8, 2], biases)})
            pin["path"] = "../checkpoint/" + filename
            pins.append(pin)
            mapping.update({name + "." + field: filename for field in ("weight", "scales", "biases")})
            modules[name] = dict(bits=3, group_size=128, mode="affine", quantize_input=True)
            filename = f"layer-{layer:02d}-{projection}.safetensors"
            pin = write_tensors(metadata / filename, {
                "indices": ("U32", [16], codes), "dictionary": ("U32", [4], table)})
            pin["path"] = filename
            pins.append(pin)
            records.append(dict(prefix=name, original_bytes=128, dictionary_values=4,
                encoded=True, encoded_bytes=80, saved_bytes=48,
                payload_sha256=hashlib.sha256(codes + table).hexdigest(),
                source_scales_sha256=hashlib.sha256(scales).hexdigest(),
                source_biases_sha256=hashlib.sha256(biases).hexdigest(),
                all_words_roundtrip_exact=True, full_encoded_readback_passed=True, file=filename))
    pin = write_tensors(checkpoint / "engram/table.safetensors", {
        "engram.weight": ("BF16", [2], b"\x00\x00\x80\x3f")})
    pin["path"] = "../checkpoint/engram/table.safetensors"
    pins.append(pin)
    config = dict(omlx_deepseek_v41=dict(version=1, preserve_mtp=True,
        quantized_modules=modules, engram_tables={"engram": dict(
            weight_file="engram/table.safetensors", weight_key="engram.weight")}))
    (checkpoint / "config.json").write_text(json.dumps(config))
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map=mapping)))
    for name in ("LICENSE", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "ds41-conversion.json"):
        (checkpoint / name).write_text("{}")
    proof = dict(status="passed", original_tensors_retained=True,
        quantization_codes_changed=False, projections=records, exact_dictionary_projections=120,
        saved_resident_bytes=5760, original_index="/private/source.gguf", external_index="/private/output.gguf")
    (metadata / "metadata-proof.json").write_text(json.dumps(proof))
    sidecar = dict(requantized=False, files=pins,
        source_index_sha256=sha256(checkpoint / "model.safetensors.index.json"),
        source_config_sha256=sha256(checkpoint / "config.json"))
    (metadata / SIDECAR).write_text(json.dumps(sidecar))
    return checkpoint, metadata, proof, scales, biases


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.checkpoint, self.metadata, self.proof, self.scales, self.biases = fixture(self.root)
        self.loader = native_loader_namespace()["NativeMetadataPlan"]

    def tearDown(self):
        self.temp.cleanup()

    def stage(self):
        output = self.root / "stage"
        subprocess.run([sys.executable, ROOT / "scripts/package_deepseek.py",
                        "--checkpoint", self.checkpoint, "--metadata-dir", self.metadata,
                        "--output-dir", output], check=True, capture_output=True)
        return output

    def test_hardlinks_proof_identity_and_source_unchanged(self):
        source_hash = sha256(self.metadata / "metadata-proof.json")
        stage = self.stage()
        proof = json.loads((stage / "exact-metadata/metadata-proof.json").read_text())
        self.assertEqual(proof["projections"], self.proof["projections"])
        self.assertNotIn("original_index", proof)
        self.assertNotIn("external_index", proof)
        self.assertEqual(source_hash, proof["source_proof_sha256"])
        self.assertEqual(source_hash, sha256(self.metadata / "metadata-proof.json"))
        self.assertEqual((stage / "source-00-w1.safetensors").stat().st_ino,
                         (self.checkpoint / "source-00-w1.safetensors").stat().st_ino)
        self.assertNotEqual((stage / "exact-metadata/metadata-proof.json").stat().st_ino,
                            (self.metadata / "metadata-proof.json").stat().st_ino)
        self.assertFalse(list(stage.rglob("*.gguf")))
        manifest = json.loads((stage / "allowlist.json").read_text())
        self.assertTrue(any(f["path"] == "engram/table.safetensors" for f in manifest["files"]))
        self.assertNotIn(str(self.root), (stage / "allowlist.json").read_text())

    def test_relocation_new_mtime_and_exact_word_decode(self):
        stage = self.stage()
        moved = self.root / "elsewhere/a model directory"
        shutil.copytree(stage, moved)
        for path in moved.rglob("*"):
            if path.is_file():
                os.utime(path, (1000, 1000))
        plan = self.loader(moved, moved / "exact-metadata/metadata-proof.json")
        self.assertEqual(len(plan.records), 120)
        self.assertEqual(plan.saved_bytes, 5760)
        record = next(iter(plan.records.values()))
        raw = record.path.read_bytes()[record.base:]
        indices = struct.unpack("<32H", raw[:64])
        dictionary = struct.unpack("<4I", raw[64:])
        pairs = [dictionary[index] for index in indices]
        self.assertEqual(struct.pack("<32H", *(p & 65535 for p in pairs)), self.scales)
        self.assertEqual(struct.pack("<32H", *(p >> 16 for p in pairs)), self.biases)
        plan._loaded.update(plan.keys)
        plan.validate_complete()
        # Same-load identities still detect replacement, even if content is exact.
        source = moved / "source-00-w1.safetensors"
        os.utime(source, (2000, 2000))
        with self.assertRaisesRegex(ValueError, "changed during loading"):
            plan.validate_complete()

    def test_corrupt_header_and_missing_engram_refused(self):
        stage = self.stage()
        sidecar = stage / "exact-metadata" / SIDECAR
        raw = json.loads(sidecar.read_text())
        raw["files"][0]["header_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "header identity mismatch"):
            self.loader(stage, stage / "exact-metadata/metadata-proof.json")
        (self.checkpoint / "engram/table.safetensors").unlink()
        with self.assertRaises(ValueError):
            package_plan(self.checkpoint, self.metadata)

    def test_missing_dictionary_and_path_escape_refused(self):
        stage = self.stage()
        sidecar = stage / "exact-metadata" / SIDECAR
        raw = json.loads(sidecar.read_text())
        raw["files"][0]["path"] = "../../../outside.safetensors"
        sidecar.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "escapes checkpoint"):
            self.loader(stage, stage / "exact-metadata/metadata-proof.json")
        (self.metadata / "layer-00-w1.safetensors").unlink()
        with self.assertRaises(ValueError):
            package_plan(self.checkpoint, self.metadata)

    def test_original_payload_not_requantized_and_existing_stage_preserved(self):
        before = sha256(self.checkpoint / "source-00-w1.safetensors")
        stage = self.stage()
        self.assertEqual(sha256(stage / "source-00-w1.safetensors"), before)
        result = subprocess.run([sys.executable, ROOT / "scripts/package_deepseek.py",
            "--checkpoint", self.checkpoint, "--metadata-dir", self.metadata,
            "--output-dir", stage], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sha256(stage / "source-00-w1.safetensors"), before)


if __name__ == "__main__":
    unittest.main()
