#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Small synthetic GPU ABI/arithmetic checks, without loading model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import mlx.core as mx
    import numpy as np
    from omlx.custom_kernels.glm_moe_dsa import fast
    mx.set_default_device(mx.gpu)
    mx.random.seed(20261010)
    weight, scales, biases = mx.quantize(
        mx.random.normal((2, 64, 256)).astype(mx.bfloat16), group_size=128, bits=3)
    mx.eval(weight, scales, biases)
    pair_words = (np.asarray(scales.view(mx.uint16)).astype(np.uint32)
                  | (np.asarray(biases.view(mx.uint16)).astype(np.uint32) << 16))
    dictionary, inverse = np.unique(pair_words, return_inverse=True)
    packed = inverse.astype("<u2").reshape(-1).view("<u4")
    words, table = mx.array(packed), mx.array(dictionary)
    checks = []
    for routed in (63, 64, 65, 129):
        x = mx.random.normal((17, 1, 256)).astype(mx.bfloat16)
        experts = mx.array(np.sort(np.arange(routed, dtype=np.uint32) % 2))
        left = mx.arange(routed, dtype=mx.uint32) % 17
        for strided in (False, True):
            if strided:
                value = mx.stack([x, x], axis=-1)[..., 0]
                ids = mx.stack([experts, experts], axis=-1)[..., 0]
                rows = mx.stack([left, left], axis=-1)[..., 0]
            else:
                value, ids, rows = x, experts, left
            expected = mx.gather_qmm(mx.contiguous(value), weight, scales, biases,
                lhs_indices=mx.contiguous(rows), rhs_indices=mx.contiguous(ids),
                transpose=True, group_size=128, bits=3, mode="affine", sorted_indices=True)
            actual = fast.deepseek_v41_joint_metadata_gather_qmm(
                value, weight, words, table, rows, ids)
            mx.eval(expected, actual)
            if not np.array_equal(np.asarray(expected.view(mx.uint16)),
                                  np.asarray(actual.view(mx.uint16))):
                raise RuntimeError(f"Native dictionary NAX arithmetic mismatch: {routed}, {strided}")
            checks.append(dict(routed_rows=routed, strided=strided, bitwise_equal=True))
    report = dict(status="passed", synthetic_GPU_cases=len(checks), cases=checks,
                  full_model_loaded=False, model_performance_measured=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
