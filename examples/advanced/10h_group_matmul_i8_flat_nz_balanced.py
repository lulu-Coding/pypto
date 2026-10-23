# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device-sized, balanced flat-NZ pipeline: 384 output tiles over the real AIC count.

10g launches one resident block per group (32 blocks). On a device with fewer
AIC than groups those blocks run in two waves: the profile showed 28 visible AIC,
four cores processing two groups, and a 569.8us makespan for ~300us of per-block
work. The grid was the problem, not the arithmetic.

This variant keeps 10g's manual per-tile schedule (seed matmul, stage-2 flat L0
pipeline, peeled Final chunk, Final store) but changes the work decomposition:

    grid      : pl.system.available_cluster_count()  (28 on the test device)
    work unit : one [TILE_M, TILE_N] output tile, addressed by a flat task index
    mapping   : for t in range(block, 384, grid):
                    g = t // N_TILES ; n_i = t % N_TILES

Every block takes 13 or 14 of the 384 tiles, so the launch is balanced on the
device it actually lands on. The divisions are accepted by the NZ offset prover:
floor division and modulo by a positive constant inherit the dividend's sign.

The trade-off is per-tile pipeline fill/drain: the N loop of 10g no longer
carries a pipeline across adjacent N tiles, but 10g already restarted its
pipeline for every N tile (seed and final chunks are peeled), so the measured
difference isolates the grid placement.

Weight layout is the flat [G*K, N] NZ form, as in 10g. This is not AscendC's
per-group NZ form; the layout is addressed after the scheduling result.

On A5, exporting ``PYPTO_PTOAS_SYNC=bufid`` selects ptoas's buffer-id
synchronization instead of the default event sync. The generated code replaces
the M->FIX set_flag/wait_flag pair with get_buf/rls_buf(PIPE_FIX); on the 28-AIC
test device this measured 354.6us AICore makespan at 97.7% core busy, versus
376.4us for the default event path. Both were bit-exact. The option is opt-in
because ptoas implements buffer-id sync only for A5 and does not convert every
GM dependency.

Run on A5 (profile first, then benchmark):

    PYPTO_PTOAS_SYNC=bufid python examples/advanced/10h_group_matmul_i8_flat_nz_balanced.py \
        -p a5 -d 0 --profile
    python examples/utils/swimlane_summary.py <printed record path>

    PYPTO_PTOAS_SYNC=bufid python examples/advanced/10h_group_matmul_i8_flat_nz_balanced.py \
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
# Shape and AscendC arch35 base block
# ---------------------------------------------------------------------------

GROUP = 32
GROUP_M = 256
K, N = 3584, 3072
M = GROUP_M * GROUP
ROWS_W = GROUP * K

TILE_M, TILE_N = 256, 256
L0_K = 128
FLAT_STAGE = 2
N_TILES = N // TILE_N
N_CHUNKS = K // L0_K
TOTAL_TILES = GROUP * N_TILES
NZ_C0 = 32

L0C_BYTES = 256 * 1024
L0A_BYTES = 64 * 1024
L0B_BYTES = 64 * 1024
MAT_BYTES = 512 * 1024

assert GROUP_M == TILE_M and M % TILE_M == 0
assert K % L0_K == 0 and N % TILE_N == 0
assert TILE_M * TILE_N * 4 <= L0C_BYTES, "accumulator exceeds A5 L0C"
assert FLAT_STAGE * (TILE_M * L0_K) <= L0A_BYTES, "flat-pipeline L0A budget exceeds L0A"
assert FLAT_STAGE * (L0_K * TILE_N) <= L0B_BYTES, "flat-pipeline L0B budget exceeds L0B"
assert FLAT_STAGE * (TILE_M * L0_K + L0_K * TILE_N) <= MAT_BYTES, "flat-pipeline Mat budget exceeds L1"
assert ROWS_W % 16 == 0 and N % NZ_C0 == 0, "flat NZ fractal alignment"
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"
assert N_CHUNKS >= 2, "need a seed and at least one peeled final chunk"
assert TOTAL_TILES == 384


# ---------------------------------------------------------------------------
# Kernel: device-sized grid-stride over independent output tiles
# ---------------------------------------------------------------------------


@pl.jit.incore
def _gemm_tasks(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[ROWS_W, N], pl.INT8, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """Grid-stride over [TILE_M, TILE_N] tiles; one tile per loop iteration."""
    block = pl.tile.get_block_idx()
    grid = pl.tile.get_block_num()
    # The last iteration of the trailing blocks is predicated off. The loop
    # itself keeps a constant start/step so its index stays provably
    # non-negative for the NZ offset prover.
    per_block = (TOTAL_TILES + grid - 1) // grid

    for i in pl.range(per_block):
        t = block + i * grid
        if t < TOTAL_TILES:
            g = t // N_TILES
            n_i = t % N_TILES
            row = g * TILE_M
            wbase = g * K
            n0 = n_i * TILE_N

            # Seed: the first L0 chunk defines the accumulator.
            a_seed = pl.load(x, [row, 0], [TILE_M, L0_K], target_memory=pl.MemorySpace.Mat)
            b_seed = pl.load(w, [wbase, n0], [L0_K, TILE_N], target_memory=pl.MemorySpace.Mat)
            acc = pl.tile.matmul(
                pl.tile.extract(a_seed, 0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                pl.tile.extract(b_seed, 0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
            )

            # Interior accumulations stay on the partial phase.
            for kc in pl.pipeline(1, N_CHUNKS - 1, stage=FLAT_STAGE):
                k0 = kc * L0_K
                a_p = pl.load(x, [row, k0], [TILE_M, L0_K], target_memory=pl.MemorySpace.Mat)
                b_p = pl.load(w, [wbase + k0, n0], [L0_K, TILE_N], target_memory=pl.MemorySpace.Mat)
                acc = pl.tile.matmul_acc(
                    acc,
                    pl.tile.extract(a_p, 0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                    pl.tile.extract(b_p, 0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
                    acc_phase=pl.AccPhase.Partial,
                )

            # Peel the last K chunk as the Final producer.
            k_tail = (N_CHUNKS - 1) * L0_K
            a_tail = pl.load(x, [row, k_tail], [TILE_M, L0_K], target_memory=pl.MemorySpace.Mat)
            b_tail = pl.load(w, [wbase + k_tail, n0], [L0_K, TILE_N], target_memory=pl.MemorySpace.Mat)
            acc = pl.tile.matmul_acc(
                acc,
                pl.tile.extract(a_tail, 0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                pl.tile.extract(b_tail, 0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
                acc_phase=pl.AccPhase.Final,
            )

            out = pl.store(acc, [row, n0], out, st_phase=pl.STPhase.Final)

    return out


@pl.jit
def group_matmul_i8_flat_nz_balanced(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[ROWS_W, N], pl.INT8, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One SPMD launch sized to the runtime's AIC count."""
    with pl.spmd(pl.system.available_cluster_count()):
        out = _gemm_tasks(x, w, out)
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
    parser = argparse.ArgumentParser(description="Device-sized balanced flat-NZ INT8 grouped matmul")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim", "a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--planner",
        choices=["pypto", "dsa_rp", "ptoas"],
        default="pypto",
        help="on-chip memory planner; the manual L0 tiles are 32 KB so all three fit",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="single DFX run (swimlane level 4 + PMU) instead of benchmark; use with swimlane_summary.py",
    )
    parser.add_argument("--artifacts-dir", default="build_output/gmm_flat_nz_balanced_profile")
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
    w_flat = w.reshape(ROWS_W, N)
    w_nz = to_nz_flat_int8(w_flat)
    out = torch.zeros((M, N), dtype=torch.int32)

    print(
        f"compiling: device-sized grid over {TOTAL_TILES} output tiles, "
        f"block {TILE_M}x{TILE_N}x{L0_K}, flat NZ weight, planner={args.planner} ..."
    )
    t0 = time.perf_counter()
    compiled = group_matmul_i8_flat_nz_balanced.compile(x, w_nz, out, config=cfg)
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
