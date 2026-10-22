# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Per-output-tile SPMD grouped matmul: 384 blocks, one [256,256] tile each.

The 2-D flat-NZ kernel (10c) reaches AICore ~259us, but its cube/MTE2 work
already matches AscendC (cube 208.9us, MTE2 ~206us). The remaining ~50us is
in-kernel pipeline fill/drain at the 12 N-tile boundaries in each group strip.
Each tile restarts a K pipeline and drains its accumulator; one resident block
cannot overlap those boundaries across N tiles.

This test splits the work into 384 independent output-tile blocks
(32 groups x 12 N tiles) dispatched as waves over 32 cores. Each block is
1/12 of today's per-core work, so the runtime can start the next tile while the
previous one drains. The trade-off is ~12 task starts per core (~0.8us each).

Weight layout: flat ND, matching the measured 2-D autot (~260us AICore). The
flat-NZ layout is not used here because the block index needs
``g = block_idx // 12`` and the NZ offset prover cannot carry non-negativity
through that floor division. If the task split wins, the layout is addressed
separately (flat NZ with precomputed offsets, per-group NZ 2-D view, or an
offline conversion).

Reference on A5:
    10c 2-D flat NZ, 32 resident blocks      AICore 259-264us
    2-D ND autot, 32 resident blocks         AICore ~260us
    AscendC GMM                              AICore 208.5us / 215us

Run:

    python examples/advanced/10e_group_matmul_i8_pertile_tasks.py -p a5 -d 0 --rounds 20

    # single DFX run; read with swimlane_summary.py (avoids the benchmark path)
    python examples/advanced/10e_group_matmul_i8_pertile_tasks.py -p a5 -d 0 --profile
    python examples/utils/swimlane_summary.py <printed record path>
"""

import argparse
import time
from pathlib import Path

import pypto.language as pl
import torch

from pypto.runtime import RunConfig, benchmark

GROUP = 32
GROUP_M = 256
K, N = 3584, 3072
M = GROUP_M * GROUP
ROWS_W = GROUP * K

TILE_K = 256
TILE_N = 256
N_TILES = N // TILE_N
K_CHUNKS = K // TILE_K
TOTAL_TILES = GROUP * N_TILES          # 384 blocks, 12 waves on 32 cores

assert K % TILE_K == 0 and N % TILE_N == 0 and M % GROUP_M == 0
assert TOTAL_TILES == 384
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"


@pl.jit.incore
def _one_output_tile(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[ROWS_W, N], pl.INT8],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One SPMD block = one group x one N tile."""
    t = pl.tile.get_block_idx()
    g = t // N_TILES
    n_i = t - g * N_TILES
    x_base = g * GROUP_M
    w_base = g * K
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
def group_matmul_i8_pertile(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[ROWS_W, N], pl.INT8],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One SPMD launch with one block per output tile (384 blocks)."""
    with pl.spmd(TOTAL_TILES):
        out = _one_output_tile(x, w, out)
    return out


def grouped_matmul_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Exact INT32 reference: per-group ``x_g @ w_g`` in FP64, narrowed once."""
    out = torch.empty((M, N), dtype=torch.int32)
    for g in range(GROUP):
        rows = slice(g * GROUP_M, (g + 1) * GROUP_M)
        out[rows] = (x[rows].double() @ w[g].double()).to(torch.int32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-output-tile SPMD INT8 grouped matmul")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim", "a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="single DFX run (swimlane level 4 + PMU) instead of benchmark",
    )
    parser.add_argument("--artifacts-dir", default="build_output/gmm_pertile_profile")
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
    w_flat = w.reshape(ROWS_W, N)
    out = torch.zeros((M, N), dtype=torch.int32)

    print(
        f"compiling: {TOTAL_TILES} per-tile SPMD blocks ({N_TILES} waves per core), "
        f"TILE_K={TILE_K}, TILE_N={TILE_N}, flat ND weight ..."
    )
    t0 = time.perf_counter()
    compiled = group_matmul_i8_pertile.compile(x, w_flat, out, config=cfg)
    print(f"compiled in {time.perf_counter() - t0:.1f}s; artifacts: {compiled.output_dir}")

    compiled(x, w_flat, out, config=cfg)

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
            print("\nsummarize with:")
            print(f"  python examples/utils/swimlane_summary.py {sorted(set(records))[0]}")
        else:
            print(f"no chip_swimlane_records.json under {[str(root) for root in roots]}")
        return

    print(f"benchmark: {args.warmup} warmup + {args.rounds} measured launches ...")
    stats = benchmark(compiled, [x, w_flat, out], rounds=args.rounds, warmup=args.warmup, config=cfg)

    flops = 2.0 * M * K * N
    median_us = stats.device_us_median
    tflops = flops / (median_us * 1e-6) / 1e12 if median_us > 0 else 0.0
    print(
        f"device_us: median={median_us:.2f} min={stats.device_us_min:.2f} "
        f"mean={stats.device_us_mean:.2f} stdev={stats.device_us_stdev:.2f} max={stats.device_us_max:.2f}"
    )
    print(f"device TFLOPS at median: {tflops:.1f}  ({flops / 1e9:.1f} GFLOP total)")
    print("reference: 10c 2-D flat NZ (32 strip blocks) AICore ~259us; AscendC AIC 215-220us")


if __name__ == "__main__":
    main()
