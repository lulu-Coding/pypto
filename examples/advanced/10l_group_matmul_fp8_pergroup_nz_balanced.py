# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FP8E4M3 grouped matmul, no scale, FP16 output: AscendC's dtype and layout.

The user's AscendC GMM baseline is FP8E4M3 x FP8E4M3 -> FP16 with no dequant
scale, per-group NZ weight, on a 32-AIC Ascend950DT. This file is the same
mathematical case on the 28-AIC PyPTO test device: 10k's schedule (device-sized
grid, per-group NZ [G, N/32, K/16, 16, 32] weight, L1_K=256 panels, stage-2
pipeline) with the data path changed to FP8.

The A5 Cube accumulates FP8 in FP32 (pto-isa TMATMUL supports the plain
`(f32, fp8, fp8)` triple; no MX scale is involved). The accumulator stays FP32
through the K loop and the final store writes it straight into the FP16 output:
`tile.store` performs no dtype check and the plain FIXPIPE writeback is exactly
the f32 -> f16 narrowing (F322F16). Do not write an explicit `pl.cast(..., "rint")`
here — the autotiler folds an f32->f16 cast only into a later matmul's Acc->Mat
writeback, not into an Acc->GM store, so an explicit cast becomes a 256x256
Vector tcvt (640 KB of Vec staging, over the UB limit). `out_dtype=pl.FP32` is
required on the first matmul: `tensor.matmul_acc` demands the accumulator dtype
to be the Cube's FP32 accumulator.

Inputs are integers in [-4, 4], exactly representable in E4M3. Every product and
partial sum is then an exact integer well below 2^24, so the FP32 accumulation
order does not matter and the FP16 output is still bit-exact against
`(x.float() @ w.float()).half()`. Values do not affect cube timing.

Run on A5 (profile first, then benchmark):

    PYPTO_PTOAS_SYNC=bufid python examples/advanced/10l_group_matmul_fp8_pergroup_nz_balanced.py \
        -p a5 -d 0 --profile
    python examples/utils/swimlane_summary.py <printed record path>

    PYPTO_PTOAS_SYNC=bufid python examples/advanced/10l_group_matmul_fp8_pergroup_nz_balanced.py \
        -p a5 -d 0 --rounds 20

The reference to compare against is the user's AscendC custom no-scale FP8 kernel
(220us on a 32-AIC 950DT). CANN's aclnnGroupedMatmulWeightNz cannot reproduce it:
its 950 checker rejects FP8 with an FP16 output and a null scale
(`161002 Parameter scale ... cannot be nullptr`). A same-board AscendC number
would need that custom kernel's source; this file provides the PyPTO-side number.
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
assert TOTAL_TILES == 384


# ---------------------------------------------------------------------------
# Kernel: device-sized grid-stride over per-group-NZ output tiles
# ---------------------------------------------------------------------------


@pl.jit.incore
def _matmul_tasks(
    x: pl.Tensor[[M, K], pl.FP8E4M3FN],
    w: pl.Tensor[[GROUP, K, N], pl.FP8E4M3FN, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.FP16]],
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
            acc = pl.matmul(x_head, w_head, out_dtype=pl.FP32)

            for kb in pl.pipeline(1, N_PANELS, stage=2):
                k1 = kb * L1_K
                x_k = x[row : row + GROUP_M, k1 : k1 + L1_K]
                w_k = w[g : g + 1, k1 : k1 + L1_K, n0 : n0 + TILE_N]
                acc = pl.matmul_acc(acc, x_k, w_k)

            # Acc (FP32) -> FP16 GM: the FIXPIPE narrowing is part of the store.
            out[row : row + GROUP_M, n0 : n0 + TILE_N] = pl.reshape(acc, [GROUP_M, TILE_N])

    return out


@pl.jit
def group_matmul_fp8_pergroup_nz_balanced(
    x: pl.Tensor[[M, K], pl.FP8E4M3FN],
    w: pl.Tensor[[GROUP, K, N], pl.FP8E4M3FN, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.FP16]],
):
    """One SPMD launch sized to the runtime's AIC count."""
    with pl.spmd(pl.system.available_cluster_count()):
        out = _matmul_tasks(x, w, out)
    return out


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def to_nz_per_group(w: torch.Tensor) -> torch.Tensor:
    """Reorder a logical ``[G, K, N]`` FP8 weight into per-group NZ bytes.

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
    """Exact FP32 accumulation, FP16 writeback: per-group ``x_g @ w_g``."""
    x_f32 = x.to(torch.float32)
    out = torch.empty((M, N), dtype=torch.float16)
    for g in range(GROUP):
        rows = slice(g * GROUP_M, (g + 1) * GROUP_M)
        out[rows] = (x_f32[rows] @ w[g].to(torch.float32)).to(torch.float16)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Balanced per-group-NZ FP8 grouped matmul")
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
    parser.add_argument("--artifacts-dir", default="build_output/gmm_fp8_pergroup_nz_profile")
    args = parser.parse_args()

    if args.platform == "a5sim":
        print("warning: 256x256 FP32 accumulator fills L0C; older a5sim executors may crash")

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
    # [-4, 4] are exactly representable in E4M3; products/partial sums stay exact
    # integers in FP32, so FP16 writeback is deterministic and bit-comparable.
    x = torch.randint(-4, 5, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
    w = torch.randint(-4, 5, (GROUP, K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
    w_nz = to_nz_per_group(w)
    out = torch.zeros((M, N), dtype=torch.float16)

    print(
        f"compiling: device-sized grid over {TOTAL_TILES} tiles, FP8E4M3 per-group NZ weight, "
        f"FP16 out, L1_K={L1_K}, planner={args.planner} ..."
    )
    t0 = time.perf_counter()
    compiled = group_matmul_fp8_pergroup_nz_balanced.compile(x, w_nz, out, config=cfg)
    print(f"compiled in {time.perf_counter() - t0:.1f}s; artifacts: {compiled.output_dir}")

    compiled(x, w_nz, out, config=cfg)

    print("verifying bit-exact against FP32-accumulation reference ...")
    expected = grouped_matmul_ref(x, w)
    if not torch.equal(out, expected):
        diff = (out.float() - expected.float()).abs()
        mismatches = int((out != expected).sum())
        raise SystemExit(f"FAIL: max abs diff {diff.max().item()}, mismatches {mismatches} of {out.numel()}")
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
