# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""High-parallelism INT8 group matmul on A5 at G=32 scale, manual-dependency edition.

    x    : [GROUP*M, K]   INT8   group g's activations at rows [g*M, (g+1)*M)
    w    : [GROUP*K, N]   INT8   group g's weights      at rows [g*K, (g+1)*K)
    scale: [GROUP*M, 1]   FP32   per-row scale (fill a band with one value for per-group scale)
    out  : [GROUP*M, N]   FP16
    acc_gm (device-side scratch): [GROUP*M, N] INT32, ~3.2 GB at the default shape.

Why manual deps (this is the load-bearing design decision):

    The auto-dependency tensormap emits EVERY older-version entry of the same
    base tensor as a fan-in edge (check_overlap's version gate returns OTHER
    before the byte-range filter, dep_compute.h), and entries for one-shot
    disjoint writes are never retired (removal needs same-version COVERED).
    A long ``acc_gm = pl.assemble(acc_gm, ...)`` chain therefore grows every
    consumer task's fan-in linearly, and the HBG runtime hard-caps it at
    CHIP_MAX_FANIN=128 with no spill pool -- 24576 gemm writes cannot fit no
    matter how the loops are structured. ``pl.manual_scope()`` bypasses the
    tensormap entirely (compute_task_fanin returns immediately); the only true
    dependency is declared explicitly: deq(row-tile) waits for gemm(row-tile).
    Everything else is disjoint by construction.

Task shape: one gemm task + one deq task per [TILE_M, N] row strip
(N // TILE_N output tiles strip-walked inside each kernel):

    gemm_tile: head-peel + pl.pipeline K loop per n-tile, INT32 acc, assembles
               acc_gm                                        (cube)
    deq_tile : DEQ_ROWS strips, cast -> row-scale -> FP16    (vector)
    2048 row strips x 2 = 49152 tasks; 18 AIC + 36 AIV saturated, deq(i)
    overlaps gemm(j != i) via the single explicit edge per pair.

Run:
    python examples/advanced/05_group_matmul_i8.py                # a5, device 0
    python examples/advanced/05_group_matmul_i8.py -p a5sim      # simulator first
"""

import argparse
import time

import pypto.language as pl
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import RunConfig

# ============ shape config (edit freely) ============
GROUP = 32
M, K, N = 8192, 3584, 3072
TILE_M, TILE_K, TILE_N = 128, 512, 256   # acc 128KB = L0C/2; K panels x2 = 384KB Mat
DEQ_ROWS = 64                            # dequant strip rows (Vec peak ~128KB)
OUT_DTYPE = pl.FP16
SCALE = 0.001                            # guarded: 127*127*K*SCALE must stay < 65504
# ===================================================

L0C_BYTES = 256 * 1024                   # A5 Acc (L0C)
MAT_BYTES = 512 * 1024                   # A5 Mat (L1)
VEC_BYTES = 240 * 1024                   # A5 safe Vec (UB)
CUBE_CORES, AIV_CORES = 18, 36

N_ROW_TILES = GROUP * M // TILE_M
ROWS_X = GROUP * M                       # x / scale / acc_gm / out rows
ROWS_W = GROUP * K                       # w rows

# Config sanity: every hardware bound is checked here so a bad edit fails at
# import with the reason, never silently on device.
assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0, "tiles must divide their dims"
assert K // TILE_K >= 2 and TILE_M % DEQ_ROWS == 0
for _d in (M, K, N, TILE_M, TILE_K, TILE_N, DEQ_ROWS):
    assert _d % 16 == 0, f"{_d} is not a multiple of 16 (cube fractal alignment)"
assert TILE_M * TILE_N * 4 <= L0C_BYTES, "acc tile exceeds A5 L0C (256KB)"
assert TILE_M * TILE_N * 8 <= L0C_BYTES, "acc tile > L0C/2: dbC ping-pong will not fit"
assert 2 * (TILE_M * TILE_K + TILE_K * TILE_N) <= MAT_BYTES, \
    "K panels x pipeline double-buffer exceed Mat (512KB); lower TILE_K"
assert DEQ_ROWS * TILE_N * 8 <= VEC_BYTES, \
    "dequant strip live set (INT32+FP32) exceeds safe Vec (240KB); lower DEQ_ROWS"
assert 127 * 127 * K * SCALE < 65504, \
    f"scale {SCALE} overflows FP16 at K={K}: need < {65504 / (127 * 127 * K):.2e}"


@pl.program
class GroupMatmulI8:
    @pl.function(type=pl.FunctionType.InCore)
    def gemm_tile(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        acc_gm: pl.InOut[pl.Tensor[[ROWS_X, N], pl.INT32]],
        row: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        """Cube: full [TILE_M, N] strip, INT8 x INT8 -> INT32, one n-tile at a time.

        The canonical projection.py K-loop form: pipeline from 0 with the
        ``if k0 == 0`` peel inside the body. The group's w band offset is
        derived kernel-side (row // M): orchestration-level FloorDiv lowers to
        arith.divsi on the MLIR index type, which is not a legal op.
        """
        wbase = (row // M) * K
        for n_i in pl.range(N // TILE_N):
            n0 = n_i * TILE_N
            for kb in pl.pipeline(0, K // TILE_K, stage=2):
                k0 = kb * TILE_K
                a = x[row : row + TILE_M, k0 : k0 + TILE_K]
                b = w[wbase + k0 : wbase + k0 + TILE_K, n0 : n0 + TILE_N]
                if k0 == 0:
                    acc = pl.matmul(a, b, out_dtype=pl.INT32)
                else:
                    acc = pl.matmul_acc(acc, a, b)
            acc_gm = pl.assemble(acc_gm, acc, [row, n0])
        return acc_gm

    @pl.function(type=pl.FunctionType.InCore)
    def deq_tile(
        self,
        acc_gm: pl.Tensor[[ROWS_X, N], pl.INT32],
        scale: pl.Tensor[[ROWS_X, 1], pl.FP32],
        out: pl.InOut[pl.Tensor[[ROWS_X, N], OUT_DTYPE]],
        row: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[ROWS_X, N], OUT_DTYPE]:
        """Vector: INT32 -> FP32 -> xrow-scale -> FP16, DEQ_ROWS strips at a time."""
        for n_i in pl.range(N // TILE_N):
            n0 = n_i * TILE_N
            for r_i in pl.range(TILE_M // DEQ_ROWS):
                r0 = r_i * DEQ_ROWS
                t = acc_gm[row + r0 : row + r0 + DEQ_ROWS, n0 : n0 + TILE_N]
                s = scale[row + r0 : row + r0 + DEQ_ROWS, :]
                deq = pl.row_expand_mul(pl.cast(t, target_type=pl.FP32, mode="none"), s)
                out = pl.assemble(out, pl.cast(deq, target_type=OUT_DTYPE), [row + r0, n0])
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        scale: pl.Tensor[[ROWS_X, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[ROWS_X, N], OUT_DTYPE]],
    ) -> pl.Tensor[[ROWS_X, N], OUT_DTYPE]:
        acc_gm = pl.create_tensor([GROUP * M, N], dtype=pl.INT32)
        # Manual deps: the auto tensormap cannot carry 24576 one-shot writes
        # under CHIP_MAX_FANIN=128 (see module docstring). The only true
        # dependency is deq(strip) <- gemm(strip); every region is disjoint.
        with pl.manual_scope():
            for t in pl.parallel(0, N_ROW_TILES):
                row = t * TILE_M
                acc_gm, g_tid = pl.submit(self.gemm_tile, x, w, acc_gm, row)
                out, _ = pl.submit(self.deq_tile, acc_gm, scale, out, row, deps=[g_tid])
        return out


_BACKENDS = {
    "a5": BackendType.Ascend950,
    "a5sim": BackendType.Ascend950,
    "a2a3": BackendType.Ascend910B,
    "a2a3sim": BackendType.Ascend910B,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", default="a5", help="target platform (default: a5)")
    parser.add_argument("-d", "--device", type=int, default=0, help="device id (default: 0)")
    args = parser.parse_args()

    cfg = RunConfig(platform=args.platform, device_id=args.device)
    torch.manual_seed(0)
    print("generating inputs ...")
    x = torch.randint(-127, 128, (GROUP * M, K), dtype=torch.int8)
    w = torch.randint(-127, 128, (GROUP * K, N), dtype=torch.int8)
    scale = torch.full((GROUP * M, 1), SCALE, dtype=torch.float32)
    out = torch.zeros(GROUP * M, N, dtype=torch.float16)

    print(f"compiling: {2 * N_ROW_TILES} tasks (gemm+deq per row strip), "
          f"acc_gm = {GROUP * M * N * 4 / 2**30:.1f} GB ...")
    t0 = time.perf_counter()
    compiled = ir.compile(GroupMatmulI8, backend_type=_BACKENDS[args.platform])
    print(f"compiled in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    compiled(x, w, scale, out, config=cfg)
    print(f"done in {time.perf_counter() - t0:.1f}s")

    print("verifying per group ...")
    for g in range(GROUP):
        ref = (x[g * M : (g + 1) * M].float() @ w[g * K : (g + 1) * K].float() * scale).half()
        got = out[g * M : (g + 1) * M]
        assert torch.allclose(got.float(), ref.float(), rtol=1e-2, atol=1e-2), (
            f"group {g}: max err = {(got.float() - ref.float()).abs().max().item()}"
        )
    print(f"OK  tasks = {2 * N_ROW_TILES}  (AIC x{CUBE_CORES} + AIV x{AIV_CORES} saturated)")


if __name__ == "__main__":
    main()
