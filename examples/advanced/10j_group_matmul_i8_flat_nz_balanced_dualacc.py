# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Balanced AutoTile with two 128-row accumulators: overlap fixpipe with cube.

10i reaches AICore 331.7us on 28 AIC using one 256x256 INT32 accumulator, which
is the entire 256 KB L0C. With a single accumulator the next tile's cube work
cannot start until the fixpipe has drained the same buffer.

This variant splits the 256-row output strip into two 128-row accumulators
(2 x 128 KB) and keeps the N tile at 256. Both accumulators share the same B
panel (256 x 256) and run through the same K pipeline, so the MTE2 byte count is
unchanged; what changes is that the two accumulator chains give the scheduler an
independent 128 KB half to work on while the other half drains. L0C is exactly
full (2 x 128 KB), matching AscendC's two-half use of the same 256 KB.

The work unit is still one [256, 256] output strip: 32 groups x 12 N tiles =
384 tasks, grid-strided over pl.system.available_cluster_count() blocks (28 on
the test device, 13 or 14 tasks each).

Weight layout is the flat [G*K, N] NZ form, as in 10i.

Measured result: 378.9us AICore makespan (97% core busy) on the 28-AIC test
device with PYPTO_PTOAS_SYNC=bufid, slower than 10i's 331.7us. The shared B
panel and two MMA chains raised cube/L0 pressure more than the extra L0C
independence saved; this file is kept as the negative result.

Run on A5 (profile first, then benchmark):

    PYPTO_PTOAS_SYNC=bufid python examples/advanced/10j_group_matmul_i8_flat_nz_balanced_dualacc.py \
        -p a5 -d 0 --profile
    python examples/utils/swimlane_summary.py <printed record path>

    PYPTO_PTOAS_SYNC=bufid python examples/advanced/10j_group_matmul_i8_flat_nz_balanced_dualacc.py \
        -p a5 -d 0 --rounds 20
"""

import argparse
import time
from pathlib import Path

import pypto.language as pl
import torch

from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime import RunConfig, benchmark

# ---------------------------------------------------------------------------
# Shape / tiling
# ---------------------------------------------------------------------------

GROUP = 32
GROUP_M = 256
K, N = 3584, 3072
M = GROUP_M * GROUP
ROWS_W = GROUP * K

TILE_M = 128
TILE_N = 256
TILE_K = 256
M_TILES_PER_STRIP = GROUP_M // TILE_M
N_TILES = N // TILE_N
K_CHUNKS = K // TILE_K
TOTAL_TILES = GROUP * N_TILES
NZ_C0 = 32

assert K % TILE_K == 0 and N % TILE_N == 0 and GROUP_M % TILE_M == 0
assert M_TILES_PER_STRIP == 2, "the dual-accumulator split is exactly 2 x 128 rows"
assert ROWS_W % 16 == 0 and N % NZ_C0 == 0, "flat NZ fractal alignment"
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"
assert TOTAL_TILES == 384


# ---------------------------------------------------------------------------
# Kernel: device-sized grid-stride, two 128-row accumulator chains per task
# ---------------------------------------------------------------------------


@pl.jit.incore
def _matmul_dual_tasks(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[ROWS_W, N], pl.INT8, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One task = two [TILE_M, TILE_N] outputs sharing a B panel."""
    block = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    per_block = (TOTAL_TILES + grid - 1) // grid

    for i in pl.range(per_block):
        t = block + i * grid
        if t < TOTAL_TILES:
            g = t // N_TILES
            n_i = t % N_TILES
            x_base = g * GROUP_M
            x_half = x_base + TILE_M
            w_base = g * K
            n0 = n_i * TILE_N

            a0 = x[x_base : x_base + TILE_M, 0:TILE_K]
            a1 = x[x_half : x_half + TILE_M, 0:TILE_K]
            b = w[w_base : w_base + TILE_K, n0 : n0 + TILE_N]
            acc0 = pl.matmul(a0, b, out_dtype=pl.INT32)
            acc1 = pl.matmul(a1, b, out_dtype=pl.INT32)

            for kb in pl.pipeline(1, K_CHUNKS, stage=2):
                k0 = kb * TILE_K
                a0 = x[x_base : x_base + TILE_M, k0 : k0 + TILE_K]
                a1 = x[x_half : x_half + TILE_M, k0 : k0 + TILE_K]
                b = w[w_base + k0 : w_base + k0 + TILE_K, n0 : n0 + TILE_N]
                acc0 = pl.matmul_acc(acc0, a0, b)
                acc1 = pl.matmul_acc(acc1, a1, b)

            out = pl.assemble(out, acc0, [x_base, n0])
            out = pl.assemble(out, acc1, [x_half, n0])

    return out


@pl.jit
def group_matmul_i8_flat_nz_balanced_dualacc(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[ROWS_W, N], pl.INT8, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One SPMD launch sized to the runtime's AIC count."""
    with pl.spmd(pl.system.available_cluster_count()):
        out = _matmul_dual_tasks(x, w, out)
    return out


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def to_nz_flat_int8(w: torch.Tensor) -> torch.Tensor:
    """Reorder a logical ``[G*K, N]`` INT8 weight into flat NZ bytes."""
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
    parser = argparse.ArgumentParser(description="Balanced AutoTile dual-accumulator flat-NZ GMM")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim", "a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--planner",
        choices=["pypto", "dsa_rp", "ptoas"],
        default="pypto",
        help="on-chip memory planner",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="single DFX run (swimlane level 4 + PMU) instead of benchmark; use with swimlane_summary.py",
    )
    parser.add_argument("--artifacts-dir", default="build_output/gmm_flat_nz_dualacc_profile")
    args = parser.parse_args()

    if args.platform == "a5sim":
        print("warning: 2 x 128x256 INT32 accumulators fill L0C; older a5sim executors may crash")

    planners = {
        "pypto": MemoryPlanner.PYPTO,
        "dsa_rp": MemoryPlanner.DSA_RP,
        "ptoas": MemoryPlanner.PTOAS,
    }
    planner = planners[args.planner]

    if args.profile:
        cfg = RunConfig(
            platform=args.platform,
            device_id=args.device,
            memory_planner=planner,
            enable_chip_swimlane=4,
            enable_pmu=2,
            enable_dep_gen=True,
            save_kernels=True,
            save_kernels_dir=args.artifacts_dir,
        )
    else:
        cfg = RunConfig(platform=args.platform, device_id=args.device, memory_planner=planner)

    torch.manual_seed(0)
    print("generating inputs ...")
    x = torch.randint(-127, 128, (M, K), dtype=torch.int8)
    w = torch.randint(-127, 128, (GROUP, K, N), dtype=torch.int8)
    w_flat = w.reshape(ROWS_W, N)
    w_nz = to_nz_flat_int8(w_flat)
    out = torch.zeros((M, N), dtype=torch.int32)

    print(
        f"compiling: device-sized dual-accumulator grid over {TOTAL_TILES} strips, "
        f"TILE_M={TILE_M} x2, TILE_N={TILE_N}, TILE_K={TILE_K}, flat NZ weight ..."
    )
    t0 = time.perf_counter()
    compiled = group_matmul_i8_flat_nz_balanced_dualacc.compile(x, w_nz, out, config=cfg)
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
    print(f"raw device_us samples: {[round(v, 2) for v in stats.samples]}")


if __name__ == "__main__":
    main()
