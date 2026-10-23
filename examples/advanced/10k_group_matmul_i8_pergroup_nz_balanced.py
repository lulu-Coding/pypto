# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Balanced per-group-NZ grouped matmul: the AscendC weight layout, device-sized grid.

10i reaches 331.7-338.5us AICore on the 28-AIC test device with a flat [G*K, N]
NZ weight. The AscendC GMM baseline (aclnnGroupedMatmulWeightNz, INT8, same
shape and board) measures 322.0-323.4us with weights packed as per-group NZ
[G, N/32, K/16, 16, 32]. This file removes the layout difference.

The kernel body is the measured 10_group_matmul_i8_nz tensor-level schedule:
2-D activation against a 3-D [1, K, N] weight slice, L1_K=256 panels, stage-2
K pipeline, AutoTileMatmulL0 for the L1->L0 split. The dispatch is 10i's:
pl.system.available_cluster_count() blocks grid-striding the 384 [256, 256]
output tiles, so all 28 AIC stay busy.

Measured on the 28-AIC test device with PYPTO_PTOAS_SYNC=bufid: 332.8us AICore
makespan at 97.2% core busy, bit-exact. The CANN 9.3
aclnnGroupedMatmulWeightNz reference (same shape, dtype, per-group NZ weight
and board; scripts/bench/ascendc_gmm_int8_bench.cpp) measures 286.6 / 288.4 /
290.9us msprof kernel Task Duration with Block Dim 28, or 322-323us by ACL
events around a warmed launch.

The pipe breakdown explains the remaining gap:

    pipe     PyPTO (PMU)   AscendC (msprof)
    cube     240.6us       243.3us
    MTE2     313.9us       269.0us
    fixpipe   25.3us       111.0us
    total    323.3us       278.1us (aic_time), 286.6-290.9us task

Cube is already matched; both kernels are MTE2-bound at ~97% MTE2 pipe
occupancy and move the same ~25.1 MB per core. The remaining ~45us is MTE2
efficiency (89 vs 82 GB/s per core), not fixpipe/cube serialization. AscendC's
111us fixpipe is overlapped under MTE2, so unit-flag overlap alone would not
close the gap. This is task #12: improve MTE2 bursts/L2 reuse.

Run on A5 (profile first, then benchmark):

    PYPTO_PTOAS_SYNC=bufid python examples/advanced/10k_group_matmul_i8_pergroup_nz_balanced.py \
        -p a5 -d 0 --profile
    python examples/utils/swimlane_summary.py <printed record path>

    PYPTO_PTOAS_SYNC=bufid python examples/advanced/10k_group_matmul_i8_pergroup_nz_balanced.py \
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

L1_K = 256
TILE_N = 256
N_TILES = N // TILE_N
N_PANELS = K // L1_K
TOTAL_TILES = GROUP * N_TILES
NZ_C0 = 32

assert K % L1_K == 0 and N % TILE_N == 0 and M % GROUP_M == 0
assert K % 16 == 0 and N % NZ_C0 == 0, "per-group NZ fractal alignment"
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"
assert TOTAL_TILES == 384


# ---------------------------------------------------------------------------
# Kernel: device-sized grid-stride over per-group-NZ output tiles
# ---------------------------------------------------------------------------


@pl.jit.incore
def _matmul_tasks(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[GROUP, K, N], pl.INT8, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One task = one [GROUP_M, TILE_N] output tile, per-group NZ weight."""
    block = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    per_block = (TOTAL_TILES + grid - 1) // grid

    for i in pl.range(per_block):
        t = block + i * grid
        if t < TOTAL_TILES:
            g = t // N_TILES
            n_i = t % N_TILES
            row = g * GROUP_M
            n0 = n_i * TILE_N

            x_head = x[row : row + GROUP_M, 0:L1_K]
            w_head = w[g : g + 1, 0:L1_K, n0 : n0 + TILE_N]
            acc = pl.matmul(x_head, w_head, out_dtype=pl.INT32)

            for kb in pl.pipeline(1, N_PANELS, stage=2):
                k1 = kb * L1_K
                x_k = x[row : row + GROUP_M, k1 : k1 + L1_K]
                w_k = w[g : g + 1, k1 : k1 + L1_K, n0 : n0 + TILE_N]
                acc = pl.matmul_acc(acc, x_k, w_k)

            out[row : row + GROUP_M, n0 : n0 + TILE_N] = pl.reshape(acc, [GROUP_M, TILE_N])

    return out


@pl.jit
def group_matmul_i8_pergroup_nz_balanced(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[GROUP, K, N], pl.INT8, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One SPMD launch sized to the runtime's AIC count."""
    with pl.spmd(pl.system.available_cluster_count()):
        out = _matmul_tasks(x, w, out)
    return out


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def to_nz_per_group(w: torch.Tensor) -> torch.Tensor:
    """Reorder a logical ``[G, K, N]`` INT8 weight into per-group NZ bytes.

    For each group the blocked order is ``[N/32, K/16, 16, 32]``; the logical
    shape is preserved so the DSL slicing stays ``[g, k, n]``.
    """
    groups, rows, cols = w.shape
    assert rows % 16 == 0, f"NZ needs 16-aligned rows, got {rows}"
    assert cols % NZ_C0 == 0, f"NZ needs {NZ_C0}-aligned columns, got {cols}"
    return (
        w.reshape(groups, rows // 16, 16, cols // NZ_C0, NZ_C0)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .reshape(groups, rows, cols)
    )


def grouped_matmul_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Exact INT32 reference: per-group ``x_g @ w_g`` in FP64, narrowed once."""
    out = torch.empty((M, N), dtype=torch.int32)
    for g in range(GROUP):
        rows = slice(g * GROUP_M, (g + 1) * GROUP_M)
        out[rows] = (x[rows].double() @ w[g].double()).to(torch.int32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Balanced per-group-NZ INT8 grouped matmul")
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
    parser.add_argument("--artifacts-dir", default="build_output/gmm_pergroup_nz_profile")
    args = parser.parse_args()

    if args.platform == "a5sim":
        print("warning: 256x256 accumulator fills L0C; older a5sim executors may crash")

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
    w_nz = to_nz_per_group(w)
    out = torch.zeros((M, N), dtype=torch.int32)

    print(
        f"compiling: device-sized grid over {TOTAL_TILES} tiles, per-group NZ weight, "
        f"L1_K={L1_K}, planner={args.planner} ..."
    )
    t0 = time.perf_counter()
    compiled = group_matmul_i8_pergroup_nz_balanced.compile(x, w_nz, out, config=cfg)
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
