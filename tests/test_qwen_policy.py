"""CPU-only regressions using the pinned cohort function and a fake advance.

No engine package, MLX, model, service, or network is imported or accessed.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import itertools
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import os
import subprocess
import pytest
TARGET = "_run_verify_cycle_batched"

ROOT = Path(os.environ.get("QWEN_TEST_SOURCE", "/nonexistent"))
if not (ROOT / "omlx").is_dir():
    pytest.skip("Set QWEN_TEST_SOURCE to the prepared optimized checkout", allow_module_level=True)
SOURCE = ROOT / "omlx/patches/mlx_lm_mtp/batch_generator.py"
POLICY_SOURCE = ROOT / "omlx/patches/mlx_lm_mtp/batch_policy.py"
# The third-party recipe is obtained by reversing only our durable guard in
# the already-patched source; no engine imports or startup monkeypatches occur.
DURABLE_TEXT = SOURCE.read_text()
ORIGINAL_TEXT = DURABLE_TEXT.replace(
    "    # Snapshot the current draft source before advance installs the next drafts.\n"
    "    # Lookup drafts have no MTP draft_lps; stale pending on an ordinary\n"
    "    # batched-head chain must not suppress that chain's policy sample.\n"
    "    lookup_pending = [\n"
    "        _p2_lookup.was_pending(state) and not state.draft_lps for state in states\n"
    "    ]\n", "").replace(
    "        and 1 <= depths[0] <= policy.max_depth\n        and not any(lookup_pending)\n",
    "        and depths[0] > 0\n")
_names = itertools.count()


def load_policy():
    namespace = {"__name__": "cpu_policy_only"}
    exec(compile(POLICY_SOURCE.read_text(), str(POLICY_SOURCE), "exec"), namespace)
    return namespace["BatchPolicy"]


Policy = load_policy()


class Clock:
    def __init__(self):
        self.now = 1.0

    def perf_counter(self):
        self.now += 0.01
        return self.now


class Logger:
    def __init__(self):
        self.events = []

    def debug(self, *args):
        self.events.append(("debug", args))

    def info(self, *args):
        self.events.append(("info", args))


class Cohort:
    def __init__(self, case, *, depths=(3, 3), pending=(False, False),
                 probabilities=None, accepted=None, next_pending=None, fixed=False,
                 advanced=None):
        self.case = case
        count = len(depths)
        if probabilities is None:
            probabilities = [([] if hit else ["mtp_lp"] * depth)
                             for hit, depth in zip(pending, depths)]
        if accepted is None:
            accepted = list(depths)
        if next_pending is None:
            next_pending = list(pending)
        if advanced is None:
            advanced = list(range(count))
        self.uids = tuple(range(count))
        self.policy = Policy(self.uids, 3, fixed=fixed)
        self.observe_calls = []
        self.timing_calls = []
        observe = self.policy.observe_mtp
        timing = self.policy.cycle_time_ms

        def tracked_observe(*args, **kwargs):
            self.observe_calls.append((args, kwargs))
            return observe(*args, **kwargs)

        def tracked_timing(*args):
            self.timing_calls.append(args)
            return timing(*args)

        self.policy.observe_mtp = tracked_observe
        self.policy.cycle_time_ms = tracked_timing
        self.states = [SimpleNamespace(
            chain=True, drafts=SimpleNamespace(shape=(depth,)),
            draft_lps=list(lp), pending=hit, controller=f"controller-{index}",
            depth=3, stats=SimpleNamespace(cycles=0, accepts=0), queue=[],
        ) for index, (depth, lp, hit) in enumerate(zip(depths, probabilities, pending))]
        self.batch_state = SimpleNamespace(states=dict(zip(self.uids, self.states)))
        self.gen = SimpleNamespace(uids=list(self.uids), _omlx_mtp_batch_policy=self.policy)
        self.advance_calls = []
        self.logger = Logger()
        package = f"_cpu_lookup_policy_{next(_names)}"
        stub = ModuleType(package + ".fused_batch")

        def advance(gen, batch_state):
            self.advance_calls.append([
                (state.controller, state.depth) for state in self.states
            ])
            for index in advanced:
                state = batch_state.states[index]
                state.stats.cycles += 1
                state.stats.accepts += accepted[index]
                state.queue.extend((index, token, "draft")
                                   for token in range(accepted[index]))
                state.queue.append((index, "bonus", "verify"))
                # The next draft source changes while this cycle is advancing.
                state.pending = next_pending[index]
        stub.advance = advance
        sys.modules[stub.__name__] = stub
        case.addCleanup(sys.modules.pop, stub.__name__, None)
        self.module = ModuleType(package + ".batch_generator")
        self.module.__dict__.update({
            "__package__": package, "Any": object, "_MtpBatchState": object,
            "time": Clock(), "logger": self.logger,
            "_p2_lookup": SimpleNamespace(was_pending=lambda state: state.pending),
            "_emit_ragged_responses": lambda gen, state, rows: rows,
            "_reconcile_mtp_batch_to_standard": lambda gen: True,
            "_drop_mtp_batch_state": lambda *args, **kwargs: None,
        })
        tree = ast.parse(ORIGINAL_TEXT)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == TARGET)
        isolated = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(isolated)
        exec(compile(isolated, str(SOURCE), "exec"), self.module.__dict__)

    def install_durable(self):
        tree = ast.parse(DURABLE_TEXT)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == TARGET)
        isolated = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(isolated)
        exec(compile(isolated, str(SOURCE), "exec"), self.module.__dict__)

    def run(self, *, patched):
        if patched:
            self.install_durable()
        result = getattr(self.module, TARGET)(self.gen, self.batch_state)
        self.case.assertEqual(len(self.advance_calls), 1)
        self.case.assertEqual(self.advance_calls[0], [(None, 3)] * len(self.uids))
        self.case.assertEqual(len(self.timing_calls), 1)
        self.case.assertIsNotNone(self.policy.finished_at)
        self.case.assertEqual(self.policy.last_mode, "mtp")
        for index, state in enumerate(self.states):
            self.case.assertEqual(state.controller, f"controller-{index}")
            self.case.assertEqual(state.depth, 3)
            self.case.assertEqual(state.queue, [])
        return result


class BatchLookupPolicyTests(unittest.TestCase):
    def test_original_fixed_and_adaptive_7_and_14_reproduce_indexerror(self):
        for fixed in (True, False):
            for depth in (7, 14):
                with self.subTest(fixed=fixed, depth=depth):
                    cohort = Cohort(self, depths=(depth, depth), pending=(True, True), fixed=fixed)
                    with self.assertRaises(IndexError):
                        cohort.run(patched=False)
                    # The real error occurs after verify and emission, in policy telemetry.
                    self.assertEqual(len(cohort.advance_calls), 1)
                    self.assertEqual(len(cohort.timing_calls), 1)

    def test_original_early_lookup_rejection_silently_contaminates_acceptance(self):
        cohort = Cohort(self, depths=(7, 7), pending=(True, True), accepted=(2, 2), fixed=True)
        cohort.run(patched=False)
        self.assertNotEqual(cohort.policy.acceptance, [0.6] * 3)

    def test_lookup_depth3_7_14_fixed_adaptive_skip_policy_but_keep_timing(self):
        for fixed in (True, False):
            for depth in (3, 7, 14):
                for accepted in ((depth, depth), (2, 1), (0, 0)):
                    with self.subTest(fixed=fixed, depth=depth, accepted=accepted):
                        cohort = Cohort(self, depths=(depth, depth), pending=(True, True),
                                        accepted=accepted, next_pending=(False, False), fixed=fixed)
                        output = cohort.run(patched=True)
                        self.assertEqual(len(output), 2)
                        self.assertEqual(cohort.observe_calls, [])
                        self.assertEqual(cohort.policy.acceptance, [0.6] * 3)
                        self.assertEqual(cohort.policy.costs, {})
                        self.assertEqual(cohort.policy.cur, 3)
                        self.assertEqual(cohort.policy.losing, 0)

    def test_mixed_lookup_and_chain_cohort_does_not_train_chain_policy(self):
        cohort = Cohort(self, pending=(True, False), accepted=(3, 2))
        cohort.run(patched=True)
        self.assertEqual(cohort.observe_calls, [])
        self.assertEqual(cohort.policy.acceptance, [0.6] * 3)
        self.assertEqual(cohort.policy.costs, {})

    def test_normal_chain3_learns_exactly_as_original(self):
        baseline = Cohort(self, accepted=(3, 2))
        expected = baseline.run(patched=False)
        patched = Cohort(self, accepted=(3, 2))
        actual = patched.run(patched=True)
        self.assertEqual(actual, expected)
        self.assertEqual(patched.policy.acceptance, baseline.policy.acceptance)
        self.assertEqual(dict(patched.policy.costs), dict(baseline.policy.costs))
        self.assertEqual(patched.observe_calls, baseline.observe_calls)
        self.assertNotEqual(patched.policy.acceptance, [0.6] * 3)

    def test_stale_pending_with_nonempty_mtp_probabilities_still_learns(self):
        cohort = Cohort(self, pending=(True, True),
                        probabilities=(["mtp"] * 3, ["mtp"] * 3), accepted=(3, 2))
        cohort.run(patched=True)
        self.assertEqual(len(cohort.observe_calls), 1)
        self.assertIn(3, cohort.policy.costs)

    def test_next_lookup_proposal_does_not_hide_current_chain_sample(self):
        cohort = Cohort(self, pending=(False, False), next_pending=(True, True), accepted=(3, 2))
        cohort.run(patched=True)
        self.assertEqual(len(cohort.observe_calls), 1)
        self.assertNotEqual(cohort.policy.acceptance, [0.6] * 3)

    def test_lookup_consumption_does_not_train_current_lookup_as_next_chain(self):
        cohort = Cohort(self, pending=(True, True), next_pending=(False, False), accepted=(3, 2))
        cohort.run(patched=True)
        self.assertEqual(cohort.observe_calls, [])

    def test_out_of_domain_nonlookup_depth_does_not_index_or_pollute_policy(self):
        cohort = Cohort(self, depths=(7, 7), pending=(False, False))
        cohort.run(patched=True)
        self.assertEqual(cohort.observe_calls, [])
        self.assertEqual(cohort.policy.acceptance, [0.6] * 3)

    def test_unequal_depth_or_incomplete_cycle_keeps_original_exclusion(self):
        for kwargs in ({"depths": (2, 3)}, {"advanced": [0]}):
            with self.subTest(kwargs=kwargs):
                cohort = Cohort(self, **kwargs)
                cohort.run(patched=True)
                self.assertEqual(cohort.observe_calls, [])

    def test_interrupted_timing_remains_excluded_from_costs(self):
        cohort = Cohort(self, accepted=(3, 2))
        cohort.policy.interrupt_timing()
        cohort.run(patched=True)
        self.assertEqual(cohort.observe_calls[0][0][2], None)
        self.assertEqual(cohort.policy.costs, {})

    def test_policy_uid_mismatch_preserves_unobserved_path(self):
        cohort = Cohort(self, pending=(True, True))
        cohort.policy.uids = (99, 100)
        cohort.install_durable()
        result = getattr(cohort.module, TARGET)(cohort.gen, cohort.batch_state)
        self.assertEqual(len(result), 2)
        self.assertEqual(cohort.timing_calls, [])
        self.assertEqual(cohort.observe_calls, [])

    def test_durable_function_is_byte_identical_to_qualified_overlay_result(self):
        def body(text):
            return next(ast.get_source_segment(text, n) for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == TARGET) + "\n"
        original = body(ORIGINAL_TEXT)
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),
            "e4953a8cb5376fd4bde22dbcbd43a124b85705bb5653b06b6630b53bf7538375")
        expected = original.replace(
            "    depths = [int(state.drafts.shape[0]) if state.chain else 1 for state in states]\n",
            "    # Snapshot the current draft source before advance installs the next drafts.\n"
            "    # Lookup drafts have no MTP draft_lps; stale pending on an ordinary\n"
            "    # batched-head chain must not suppress that chain's policy sample.\n"
            "    lookup_pending = [\n"
            "        _p2_lookup.was_pending(state) and not state.draft_lps for state in states\n"
            "    ]\n"
            "    depths = [int(state.drafts.shape[0]) if state.chain else 1 for state in states]\n").replace(
            "        and depths[0] > 0\n",
            "        and 1 <= depths[0] <= policy.max_depth\n        and not any(lookup_pending)\n")
        self.assertEqual(body(DURABLE_TEXT), expected)

    def test_standard_launchers_have_no_runtime_source_exec_or_inspect(self):
        release = Path(__file__).resolve().parents[1]
        for name in ("serve_qwen.py", "benchmark_qwen.py"):
            text = (release / "scripts" / name).read_text()
            self.assertNotIn("inspect.getsource", text)
            self.assertNotIn("exec(compile", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
