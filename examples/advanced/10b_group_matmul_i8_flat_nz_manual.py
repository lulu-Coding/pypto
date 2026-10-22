# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Manual flat-L0 pipeline prototype: flat NZ INT8 group matmul.

This is the performance prototype for the grouped-matmul investigation. It
keeps the manual tile-level schedule that the compiler-driven AutoTile path
cannot express: one SPMD block per group, a 256x256x128 base block, and a FLAT
``pl.pipeline`` over all 28 L0 K-chunks (stage=2) so MTE1/L0 loads can overlap
the cube. The seed chunk is peeled once; every later chunk is uniform.

Weight layout caveat: this prototype uses the flat ``[G*K, N]`` NZ form (all
groups blocked together), not the AscendC per-group ``[G, N/32, K/16, 16, 32]``
form. The flat form is supported by PyPTO today and lets us measure the manual
pipeline without a compiler change. If it wins, the follow-up is to add a 2-D
per-group NZ view and port the same schedule to the AscendC layout.

Reference points on Ascend950DT_9582 (32 AIC, same 8192x3584x3072 INT8 shape):

    AutoTile, per-group NZ, L1_K=256, PYPTO   312.83us  (the current best)
    AutoTile, per-group NZ, L1_K=512, PYPTO   362.69us
    AscendC GMM (msprof kernel time)          220us

The PMU breakdown of the 312.83us run showed cube 208.5us and MTE2 ~203us --
essentially AscendC's work -- with ~40us of in-kernel pipe stalls and ~26us of
non-overlapped MTE1/FIX. This prototype targets exactly those stalls.

Run on A5:

    python examples/advanced/10b_group_matmul_i8_flat_nz_manual.py -p a5 -d 0 --rounds 20
"""

import argparse
import time

import pypto.language as pl
import torch

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


# ---------------------------------------------------------------------------
# Kernel: one resident SPMD block per group, flat L0-chunk pipeline
# ---------------------------------------------------------------------------


@pl.jit.incore
def _gemm_strip(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[ROWS_W, N], pl.INT8, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One [TILE_M, N] strip: seed chunk + flat 28-chunk L0 pipeline."""
    # Keep both offsets as plain products of the block index. The NZ offset
    # prover tracks non-negativity through get_block_idx and products, but not
    # through the equivalent ``(row // GROUP_M) * K`` floor-division chain.
    block = pl.tile.get_block_idx()
    row = block * TILE_M
    wbase = block * K

    for n_i in pl.range(N_TILES):
        n0 = n_i * TILE_N

        # Seed: the first L0 chunk defines the accumulator.
        a_seed = pl.load(x, [row, 0], [TILE_M, L0_K], target_memory=pl.MemorySpace.Mat)
        b_seed = pl.load(w, [wbase, n0], [L0_K, TILE_N], target_memory=pl.MemorySpace.Mat)
        acc = pl.tile.matmul(
            pl.tile.extract(a_seed, 0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
            pl.tile.extract(b_seed, 0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
        )

        # Flat pipeline: every remaining L0 chunk is the same load/extract/MAD.
        for kc in pl.pipeline(1, N_CHUNKS, stage=FLAT_STAGE):
            k0 = kc * L0_K
            a_p = pl.load(x, [row, k0], [TILE_M, L0_K], target_memory=pl.MemorySpace.Mat)
            b_p = pl.load(w, [wbase + k0, n0], [L0_K, TILE_N], target_memory=pl.MemorySpace.Mat)
            acc = pl.tile.matmul_acc(
                acc,
                pl.tile.extract(a_p, 0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                pl.tile.extract(b_p, 0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
            )

        out = pl.store(acc, [row, n0], out)

    return out


@pl.jit
def group_matmul_i8_flat_nz_manual(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[ROWS_W, N], pl.INT8, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One SPMD launch: 32 blocks, one group strip per block."""
    with pl.spmd(GROUP):
        out = _gemm_strip(x, w, out)
    return out


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def to_nz_flat_int8(w: torch.Tensor) -> torch.Tensor:
    """Reorder a logical ``[G*K, N]`` INT8 weight into flat NZ bytes.

    The logical shape is preserved; the contiguous bytes follow
    ``[N/32, R/16, 16, 32]`` for the whole flattened matrix. This is NOT the
    per-group NZ layout used by AscendC; it is the layout PyPTO supports today
    without a 3-D view.

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
    parser = argparse.ArgumentParser(description="Manual flat-L0 pipeline, flat NZ grouped matmul")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim", "a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if args.platform == "a5sim":
        print("warning: 256x256 accumulator fills L0C; older a5sim executors may crash")

    cfg = RunConfig(platform=args.platform, device_id=args.device)

    torch.manual_seed(0)
    print("generating inputs ...")
    x = torch.randint(-127, 128, (M, K), dtype=torch.int8)
    w = torch.randint(-127, 128, (GROUP, K, N), dtype=torch.int8)
    w_flat = w.reshape(ROWS_W, N)
    w_nz = to_nz_flat_int8(w_flat)
    out = torch.zeros((M, N), dtype=torch.int32)

    print(
        f"compiling: {GROUP} SPMD blocks, flat L0 pipeline stage={FLAT_STAGE}, "
        f"block {TILE_M}x{TILE_N}x{L0_K}, flat NZ weight ..."
    )
    t0 = time.perf_counter()
    compiled = group_matmul_i8_flat_nz_manual.compile(x, w_nz, out, config=cfg)
    print(f"compiled in {time.perf_counter() - t0:.1f}s; artifacts: {compiled.output_dir}")

    compiled(x, w_nz, out, config=cfg)

    print("verifying bit-exact against FP64 reference ...")
    expected = grouped_matmul_ref(x, w)
    if not torch.equal(out, expected):
        diff = (out - expected).abs()
        raise SystemExit(f"FAIL: max abs diff {diff.max().item()}, mismatches {int((diff != 0).sum())}")
    print("OK  bit-exact")

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
    print("reference: AutoTile per-group NZ L1_K=256 = 312.83us; AscendC kernel = 220us")


if __name__ == "__main__":
    main()
