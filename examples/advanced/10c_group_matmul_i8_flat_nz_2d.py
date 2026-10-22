# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""2-D tensor-level INT8 grouped matmul with a flat NZ weight.

Investigates the gap between the current 3-D/per-group-NZ AutoTile case and
the measured 2-D "autot" case (total ~300us, AICore ~260us on the same A5):

    current 10_group_matmul_i8_nz.py
        w : [G, K, N] per-group NZ
        matmul: 2-D A x 3-D B with a leading unit batch dim
        312.83us total / ~281us AICore

    this file
        w : [G*K, N] flat NZ (all groups blocked together)
        matmul: plain 2-D A x 2-D B, one SPMD block per group
        same TILE_K=256, TILE_N=256, K-panel pipeline as the autot case

The only structural difference from the 3-D case is the weight layout and the
drop of the batch dim. If this reproduces the ~300us total / ~260us AICore
number, the 3-D batch path is the thing to fix; if it stays at ~313us, the
2-D vs 3-D distinction is not the source and we need to inspect the autot's
measurement/commit.

This uses the flat NZ layout, not AscendC's per-group layout. A winning result
would be followed by either (a) converting the weight to flat NZ offline or
(b) adding a 2-D view of the per-group NZ tensor in the compiler.

Run:

    # device_wall benchmark
    python examples/advanced/10c_group_matmul_i8_flat_nz_2d.py -p a5 -d 0 --rounds 20

    # same measurement as the autot (swimlane level 4, one DFX run, no benchmark)
    python examples/advanced/10c_group_matmul_i8_flat_nz_2d.py -p a5 -d 0 --profile
    python examples/utils/swimlane_summary.py <printed record path>
"""

import argparse
import time
from pathlib import Path

import pypto.language as pl
import torch

from pypto.runtime import RunConfig, benchmark

# ---------------------------------------------------------------------------
# Shape / tiling
# ---------------------------------------------------------------------------

GROUP = 32
GROUP_M = 256
K, N = 3584, 3072
M = GROUP_M * GROUP
ROWS_W = GROUP * K

TILE_K = 256
TILE_N = 256
N_TILES = N // TILE_N
K_CHUNKS = K // TILE_K
NZ_C0 = 32

assert K % TILE_K == 0 and N % TILE_N == 0 and M % GROUP_M == 0
assert ROWS_W % 16 == 0 and N % NZ_C0 == 0, "flat NZ fractal alignment"
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"


# ---------------------------------------------------------------------------
# Kernel: 2-D A x 2-D B, one block per group
# ---------------------------------------------------------------------------


@pl.jit.incore
def _matmul_group_2d(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[ROWS_W, N], pl.INT8, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One group strip: 12 N tiles, each a 14-panel K pipeline."""
    g = pl.tile.get_block_idx()
    x_base = g * GROUP_M
    w_base = g * K

    for n_i in pl.range(N_TILES):
        n0 = n_i * TILE_N

        a = x[x_base : x_base + GROUP_M, 0:TILE_K]
        b = w[w_base : w_base + TILE_K, n0 : n0 + TILE_N]
        acc = pl.matmul(a, b, out_dtype=pl.INT32)

        for kb in pl.pipeline(1, K_CHUNKS, stage=2):
            k0 = kb * TILE_K
            a = x[x_base : x_base + GROUP_M, k0 : k0 + TILE_K]
            b = w[w_base + k0 : w_base + k0 + TILE_K, n0 : n0 + TILE_N]
            acc = pl.matmul_acc(acc, a, b)

        out = pl.assemble(out, acc, [x_base, n0])

    return out


@pl.jit
def group_matmul_i8_flat_nz_2d(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[ROWS_W, N], pl.INT8, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One SPMD launch: 32 blocks, one group per block."""
    with pl.spmd(GROUP):
        out = _matmul_group_2d(x, w, out)
    return out


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def to_nz_flat_int8(w: torch.Tensor) -> torch.Tensor:
    """Reorder a logical ``[G*K, N]`` INT8 weight into flat NZ bytes.

    For the whole flattened matrix the blocked order is
    ``[N/32, R/16, 16, 32]``. This is the flat layout PyPTO supports today;
    it is not AscendC's per-group NZ layout.

    Args:
        w: Logical row-major INT8 weight of shape ``[G*K, N]``.

    Returns:
        A tensor with the same shape as ``w``, holding flat NZ bytes.

    Raises:
        AssertionError: If rows are not 16-aligned or columns are not 32-aligned.
    """
    rows, cols = w.shape
    assert rows % 16 == 0, f"NZ needs 16-aligned rows, got {rows}"
    assert cols % NZ_C0 == 0, f"NZ needs {NZ_C0}-aligned columns, got {cols}"
    return (
        w.reshape(rows // 16, 16, cols // NZ_C0, NZ_C0)
        .permute(2, 0, 1, 3)
        .contiguous()
        .reshape(rows, cols)
    )


def grouped_matmul_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Exact INT32 reference: per-group ``x_g @ w_g`` in FP64, narrowed once."""
    out = torch.empty((M, N), dtype=torch.int32)
    for g in range(GROUP):
        rows = slice(g * GROUP_M, (g + 1) * GROUP_M)
        out[rows] = (x[rows].double() @ w[g].double()).to(torch.int32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="2-D flat-NZ INT8 grouped matmul")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim", "a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="single DFX run (swimlane level 4 + PMU) instead of benchmark; use with swimlane_summary.py",
    )
    parser.add_argument("--artifacts-dir", default="build_output/gmm_flat_nz_2d_profile")
    args = parser.parse_args()

    if args.platform == "a5sim":
        print("warning: 256x256 INT32 accumulators fill L0C; older a5sim executors may crash")

    if args.profile:
        cfg = RunConfig(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=4,
            enable_pmu=2,
            enable_dep_gen=True,
            save_kernels=True,
            save_kernels_dir=args.artifacts_dir,
        )
    else:
        cfg = RunConfig(platform=args.platform, device_id=args.device)

    torch.manual_seed(0)
    print("generating inputs ...")
    x = torch.randint(-127, 128, (M, K), dtype=torch.int8)
    w = torch.randint(-127, 128, (GROUP, K, N), dtype=torch.int8)
    w_nz = to_nz_flat_int8(w.reshape(ROWS_W, N))
    out = torch.zeros((M, N), dtype=torch.int32)

    print(
        f"compiling: {GROUP} blocks, 2-D matmul, TILE_K={TILE_K}, TILE_N={TILE_N}, "
        f"flat NZ weight ..."
    )
    t0 = time.perf_counter()
    compiled = group_matmul_i8_flat_nz_2d.compile(x, w_nz, out, config=cfg)
    print(f"compiled in {time.perf_counter() - t0:.1f}s; artifacts: {compiled.output_dir}")

    compiled(x, w_nz, out, config=cfg)

    print("verifying bit-exact against FP64 reference ...")
    expected = grouped_matmul_ref(x, w)
    if not torch.equal(out, expected):
        diff = (out - expected).abs()
        raise SystemExit(f"FAIL: max abs diff {diff.max().item()}, mismatches {int((diff != 0).sum())}")
    print("OK  bit-exact")

    if args.profile:
        roots = [Path(args.artifacts_dir)]
        output_dir = getattr(compiled, "output_dir", None)
        if output_dir:
            roots.append(Path(output_dir))
        records = []
        for root in roots:
            if root.exists():
                records.extend(root.rglob("chip_swimlane_records.json"))
        print("profile artifacts:")
        for record in sorted(set(records)):
            print(f"  {record}")
        if records:
            print("\nsummarize the AICore span with:")
            print(f"  python examples/utils/swimlane_summary.py {sorted(set(records))[0]}")
        else:
            print(f"no chip_swimlane_records.json under {[str(root) for root in roots]}")
        return

    print(f"benchmark: {args.warmup} warmup + {args.rounds} measured launches ...")
    stats = benchmark(compiled, [x, w_nz, out], rounds=args.rounds, warmup=args.warmup, config=cfg)

    flops = 2.0 * M * K * N
    median_us = stats.device_us_median
    tflops = flops / (median_us * 1e-6) / 1e12 if median_us > 0 else 0.0
    print(
        f"device_us: median={median_us:.2f} min={stats.device_us_min:.2f} "
        f"mean={stats.device_us_mean:.2f} stdev={stats.device_us_stdev:.2f} max={stats.device_us_max:.2f}"
    )
    print(f"device TFLOPS at median: {tflops:.1f}  ({flops / 1e9:.1f} GFLOP total)")
    print("reference: 3-D per-group NZ AutoTile = 312.83us; reported 2-D ND autot ~300us total/~260us AICore")


if __name__ == "__main__":
    main()
