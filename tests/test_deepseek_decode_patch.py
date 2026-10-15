"""Release-delta inventory and provenance checks; no MLX import or inference.

Full clean-base patch application and MLX CPU regressions are separate release
qualification steps. These fast checks prevent accidental scope/hash drift.
"""

import ast
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = "patches/deepseek/omlx/0005-long-context-decode-and-prefill-fairness.patch"
EXPECTED_PATHS = {
    "omlx/scheduler.py",
    "omlx/engine_core.py",
    "tests/test_engine_core.py",
    "tests/test_deepseek_v41_growth_lifecycle_cpu.py",
    "tests/test_engine_terminal_delivery_cpu.py",
    "tests/test_preserved_prefill_decode_priority.py",
}
QUALIFIED_NEW_TESTS = {
    "tests/test_deepseek_v41_growth_lifecycle_cpu.py":
        "d67328b81899d78b612778635e8d8be3edb81e473b6b89c2e34b2b31381fa1a8",
    "tests/test_engine_terminal_delivery_cpu.py":
        "af684c48be904a211e66e1dba001d2409e1e0c41d573236029abd6182dc33b0c",
    "tests/test_preserved_prefill_decode_priority.py":
        "3e1acf906c7a089516b04cfa27bb034654c5e4ccb02549454a54bf34bbad0029",
}


def sections():
    result = {}
    for section in (ROOT / PATCH).read_text().split("diff --git ")[1:]:
        header = section.splitlines()[0]
        left, right = header.split(" b/", 1)
        if left != "a/" + right or right in result:
            raise ValueError("Mismatched or duplicate patch path")
        result[right] = section
    return result


class TestDeepseekDecodePatch(unittest.TestCase):
    def test_manifest_orders_delta_after_its_existing_dependencies(self):
        manifest = json.loads((ROOT / "profiles/deepseek/sources.json").read_text())
        patches = manifest["omlx"]["patches"]
        self.assertEqual(manifest["omlx"]["revision"],
                         "d4298ad1811bdd1b565660263380b52fc4ff71f0")
        self.assertEqual(patches[-1], PATCH)
        self.assertEqual(patches[-3:-1], [
            "patches/deepseek/omlx/0003-exact-prefix-cache-persistence.patch",
            "patches/deepseek/omlx/0004-portable-release-identities.patch",
        ])
        self.assertEqual(len(patches), len(set(patches)))
        for path in patches:
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                                 manifest["patch_sha256"][path])

    def test_delta_has_exactly_six_reviewed_paths(self):
        changed = sections()
        self.assertEqual(set(changed), EXPECTED_PATHS)
        new = {path for path, section in changed.items()
               if "\nnew file mode 100644\n" in section}
        self.assertEqual(new, set(QUALIFIED_NEW_TESTS))
        for path, section in changed.items():
            with self.subTest(path=path):
                self.assertIn("+++ b/" + path + "\n", section)
                self.assertIn("\n@@ ", section)

    def test_new_regressions_are_the_qualified_portable_sources(self):
        changed = sections()
        for path, expected in QUALIFIED_NEW_TESTS.items():
            with self.subTest(path=path):
                section = changed[path]
                self.assertIn("--- /dev/null\n", section)
                source = "\n".join(line[1:] for line in section.splitlines()
                                     if line.startswith("+") and not line.startswith("+++")) + "\n"
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), expected)
                ast.parse(source, filename=path)
                self.assertNotIn("/Users/", source)
                self.assertNotIn("/workspace/local_moe", source)


if __name__ == "__main__":
    unittest.main()
