import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repro.py"
spec = importlib.util.spec_from_file_location("repro", SCRIPT)
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)


class VerifyTests(unittest.TestCase):
    def test_valid_and_payload_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.bin").write_bytes(b"same header; payload A")
            digest = repro.file_sha256(root / "data.bin")
            (root / "SHA256SUMS").write_text(digest + "  data.bin\n")
            self.assertEqual(repro.verify_model(root), 0)
            (root / "data.bin").write_bytes(b"same header; payload B")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                repro.verify_model(root)

    def test_manifest_rejects_escape_duplicate_and_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "file").write_bytes(b"ok")
            digest = repro.file_sha256(root / "file")
            for contents in [digest + "  ../file\n", digest + "  /file\n", "",
                             (digest + "  file\n") * 2]:
                (root / "SHA256SUMS").write_text(contents)
                with self.assertRaises(ValueError):
                    repro.verify_model(root)

    def test_download_pin_qwen_and_missing_model_path(self):
        self.assertEqual(repro.main(["download", "--model", "qwen", "--model-dir", "/tmp/model", "--dry-run"]), 0)
        with self.assertRaises(SystemExit):
            repro.main(["download", "--model", "qwen", "--dry-run"])

    def test_release_revision_rejects_unverified_and_non_hex_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "profiles").mkdir()
            original_root = repro.ROOT
            repro.ROOT = root
            try:
                for status, revision in [("pending", "a" * 40), ("passed", "z" * 40)]:
                    (root / "profiles/release.json").write_text(json.dumps({"deepseek": {
                        "verification_status": status, "revision": revision}}))
                    with self.assertRaises(ValueError):
                        repro.release_revision()
                (root / "profiles/release.json").write_text(json.dumps({"deepseek": {
                    "verification_status": "passed", "revision": "a" * 40}}))
                self.assertEqual(repro.release_revision(), "a" * 40)
            finally:
                repro.ROOT = original_root


if __name__ == "__main__":
    unittest.main()
