# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Per-group NZ INT8 grouped matmul on Ascend950DT (A5, 32 AIC).

Benchmark shape:

    x  : [8192, 3584]    INT8   M = 32 groups x 256 rows
    w  : [32, 3584, 3072] INT8  logical [G, K, N], each group stored in its own
                                PTO NZ fractal block
    out: [8192, 3072]    INT32  raw accumulator, no dequant

Weight layout. This version matches the AscendC reference: the NZ blocks are
built per group, i.e. the physical order is
``[G, N/32, K/16, 16, 32]`` for INT8. The DSL keeps the logical ``[G, K, N]``
shape and annotates the tensor as ``pl.NZ``; ``to_nz_per_group`` produces the
bytes.

Kernel layout. One SPMD block per group (32 groups = 32 AIC on this device):

    - K is explicitly split into seven 512-deep panels; the head panel seeds the
      accumulator and the remaining six run under ``pl.pipeline(stage=2)``.
    - N is split into twelve 256-wide blocks; each block is one [1, 256, 256]
      INT32 accumulator, which is the AscendC arch35 base block.
    - L0 tiling inside each panel is left to AutoTileMatmulL0. This is required
      by the current NZ milestone: a 2-D rank-reduced view of a 3-D NZ tensor is
      not a supported load source, so the supported grouped path is the
      tensor-level ``2-D @ 3-D`` matmul, which AutoTile then lowers to L0.

Run on A5:

    python examples/advanced/10_group_matmul_i8_nz.py -p a5 -d 0 --rounds 20

The [256, 256] INT32 output block fills L0C exactly. Older a5sim executors may
crash on this tiling; use the real device.
"""

import argparse
import time

import pypto.language as pl
import torch

from pypto.runtime import RunConfig, benchmark

# ---------------------------------------------------------------------------
# Shape and arch35 design point
# ---------------------------------------------------------------------------

GROUP = 32
GROUP_M = 256
K, N = 3584, 3072
M = GROUP_M * GROUP

N_TILE = 256                             # AscendC baseN
L1_K = 512                               # AscendC stepK = 4, L0 chunk 128
N_TILES = N // N_TILE                    # 12 N blocks per group
N_PANELS = K // L1_K                     # 7 K panels

NZ_C0 = 32                               # 256 bits / INT8 = 32 elements

assert GROUP_M == 256
assert K % L1_K == 0 and N % N_TILE == 0
assert K % 16 == 0 and N % NZ_C0 == 0, "NZ fractal alignment"
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@pl.jit
def group_matmul_i8_nz(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[GROUP, K, N], pl.INT8, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One block per group; per-group NZ weight; K-panel pipeline.

    The 3-D weight slice keeps its leading unit dim so the NZ tensor view stays
    rank-3, which is the shape the current grouped-matmul NZ lowering supports.
    The matmul broadcasts the 2-D activation against that unit batch dim, and
    AutoTileMatmulL0 selects the L1->L0 split inside each 512-deep K panel.
    """
    for g in pl.spmd(GROUP):
        row = g * GROUP_M

        for n_i in pl.range(N_TILES):
            n0 = n_i * N_TILE

            x_head = x[row : row + GROUP_M, 0:L1_K]
            w_head = w[g : g + 1, 0:L1_K, n0 : n0 + N_TILE]
            acc = pl.matmul(x_head, w_head, out_dtype=pl.INT32)

            for kb in pl.pipeline(1, N_PANELS, stage=2):
                k1 = kb * L1_K
                x_k = x[row : row + GROUP_M, k1 : k1 + L1_K]
                w_k = w[g : g + 1, k1 : k1 + L1_K, n0 : n0 + N_TILE]
                acc = pl.matmul_acc(acc, x_k, w_k)

            out[row : row + GROUP_M, n0 : n0 + N_TILE] = pl.reshape(acc, [GROUP_M, N_TILE])

    return out


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def to_nz_per_group(w: torch.Tensor) -> torch.Tensor:
    """Reorder a logical ``[G, K, N]`` INT8 weight into per-group NZ bytes.

    The logical shape is preserved; only the byte order changes. For each group
    the blocked order is ``[N/32, K/16, 16, 32]``.

    Args:
        w: Logical row-major INT8 weight of shape ``[G, K, N]``.

    Returns:
        A tensor with the same shape as ``w``, holding per-group NZ bytes.

    Raises:
        AssertionError: If K is not 16-aligned or N is not 32-aligned.
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
        acc = x[rows].double() @ w[g].double()
        out[rows] = acc.to(torch.int32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-group NZ INT8 grouped matmul (A5)")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim", "a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if args.platform == "a5sim":
        print("warning: 256x256 INT32 output blocks fill L0C; older a5sim executors crash on this tiling")

    cfg = RunConfig(platform=args.platform, device_id=args.device)

    torch.manual_seed(0)
    print("generating inputs ...")
    x = torch.randint(-127, 128, (M, K), dtype=torch.int8)
    w = torch.randint(-127, 128, (GROUP, K, N), dtype=torch.int8)
    w_nz = to_nz_per_group(w)
    out = torch.zeros((M, N), dtype=torch.int32)

    print(
        f"compiling: {GROUP} groups x {N_TILES} N blocks, "
        f"K panel {L1_K}, per-group NZ weight ..."
    )
    t0 = time.perf_counter()
    compiled = group_matmul_i8_nz.compile(x, w_nz, out, config=cfg)
    print(f"compiled in {time.perf_counter() - t0:.1f}s; artifacts: {compiled.output_dir}")

    compiled(x, w_nz, out, config=cfg)

    print("verifying bit-exact against FP64 reference ...")
    expected = grouped_matmul_ref(x, w)
    if not torch.equal(out, expected):
        diff = (out - expected).abs()
        raise SystemExit(f"FAIL: max abs diff {diff.max().item()}, mismatches {int((diff != 0).sum())}")
    print("OK  bit-exact")

    print(f"benchmark: {args.warmup} warmup + {args.rounds} measured launches ...")
    stats = benchmark(
        compiled,
        [x, w_nz, out],
        rounds=args.rounds,
        warmup=args.warmup,
        config=cfg,
    )

    flops = 2.0 * M * K * N
    median_us = stats.device_us_median
    tflops = flops / (median_us * 1e-6) / 1e12 if median_us > 0 else 0.0
    print(
        f"device_us: median={median_us:.2f} min={stats.device_us_min:.2f} "
        f"mean={stats.device_us_mean:.2f} stdev={stats.device_us_stdev:.2f} "
        f"max={stats.device_us_max:.2f}"
    )
    print(f"device TFLOPS at median: {tflops:.1f}  ({flops / 1e9:.1f} GFLOP total)")
    print(f"raw device_us samples: {[round(v, 2) for v in stats.samples]}")
    if stats.all_zero_device:
        print("note: raw device timings are all zero — expected on simulator, not on a5")


if __name__ == "__main__":
    main()
