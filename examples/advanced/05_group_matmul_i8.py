# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""High-parallelism INT8 group matmul on A5 at G=32 scale: staged GEMM || dequant dual pipeline.

    x    : [GROUP*M, K]   INT8   group g's activations at rows [g*M, (g+1)*M)
    w    : [GROUP*K, N]   INT8   group g's weights      at rows [g*K, (g+1)*K)
    scale: [GROUP*M, 1]   FP32   per-row scale (fill a band with one value for per-group scale)
    out  : [GROUP*M, N]   FP16
    acc_gm (device-side scratch): [GROUP*M, N] INT32, ~3.2 GB at the default shape.

Why staged-only (no mixed epilogue) at this scale:

    1. ChipCallable registers at most 1024 distinct child kernels (MaxChildren,
       runtime/src/common/task_interface/callable.h). pl.unroll'ed pl.at scopes
       each become their own kernel (baked literal offsets), so GROUP x N_TILES
       unrolled copies -- split further into AIC+AIV by ExpandMixedKernel --
       blew past that limit at G=32 x N/128. All loops here are runtime
       range/parallel instead: the whole program is exactly TWO child kernels
       (gemm + deq) regardless of GROUP/N.
    2. The vector epilogue's live set (INT32 tile + FP32 tile, ~8B/elem) must
       fit the 240KB safe Vec UB, so the dequant works on DEQ_ROWS strips.

Parallelism:

    AIC 18 cores: GROUP x (M/TILE_M) x (N/TILE_N) gemm tasks (24576 at defaults)
    AIV 36 cores: one deq task per gemm tile, strip-looped inside; SSA deps let
                  deq(i) overlap gemm(j != i), so both pipe lanes stay saturated.
    Within a gemm task: pl.pipeline stage=2 double-buffers the K panels; the
    [TILE_M, TILE_N] INT32 accumulator is L0C/2, leaving room for AutoTile dbC.

Run:
    python examples/advanced/05_group_matmul_i8.py                # a5, device 0
    python examples/advanced/05_group_matmul_i8.py -p a5sim      # simulator first
"""

import argparse
import time

import pypto.language as pl
import torch
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


@pl.jit
def group_matmul_i8(x: pl.Tensor, w: pl.Tensor, scale: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Staged dual pipeline: cube GEMM into an INT32 GM scratch, vector dequant out.

    All loops are runtime range/parallel -- the outlined scope bodies are shared,
    so the compiled program carries exactly two child kernels (gemm, deq) no
    matter how large GROUP / N grow.
    """
    acc_gm = pl.create_tensor([GROUP * M, N], dtype=pl.INT32)
    for g in pl.range(GROUP):
        for m0 in pl.parallel(0, M, TILE_M):
            for n_i in pl.range(N // TILE_N):
                n0 = n_i * TILE_N
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="gemm"):
                    acc = pl.create_tensor([TILE_M, TILE_N], dtype=pl.INT32)
                    for kb in pl.pipeline(0, K // TILE_K, stage=2):
                        k0 = kb * TILE_K
                        a = x[g * M + m0 : g * M + m0 + TILE_M, k0 : k0 + TILE_K]
                        b = w[g * K + k0 : g * K + k0 + TILE_K, n0 : n0 + TILE_N]
                        if k0 == 0:
                            acc = pl.matmul(a, b, out_dtype=pl.INT32)
                        else:
                            acc = pl.matmul_acc(acc, a, b)
                    acc_gm = pl.assemble(acc_gm, acc, [g * M + m0, n0])
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="deq"):
                    for r_i in pl.range(TILE_M // DEQ_ROWS):
                        r0 = r_i * DEQ_ROWS
                        row = g * M + m0 + r0
                        t = acc_gm[row : row + DEQ_ROWS, n0 : n0 + TILE_N]
                        s = scale[row : row + DEQ_ROWS, :]
                        deq = pl.row_expand_mul(pl.cast(t, target_type=pl.FP32, mode="none"), s)
                        out = pl.assemble(out, pl.cast(deq, target_type=OUT_DTYPE), [row, n0])
    return out


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

    n_tiles = GROUP * (M // TILE_M) * (N // TILE_N)
    print(f"running: {n_tiles} gemm + {n_tiles} deq tasks, "
          f"acc_gm = {GROUP * M * N * 4 / 2**30:.1f} GB ...")
    t0 = time.perf_counter()
    group_matmul_i8(x, w, scale, out, config=cfg)
    print(f"done in {time.perf_counter() - t0:.1f}s (first call includes compile)")

    print("verifying per group ...")
    for g in range(GROUP):
        ref = (x[g * M : (g + 1) * M].float() @ w[g * K : (g + 1) * K].float() * scale).half()
        got = out[g * M : (g + 1) * M]
        assert torch.allclose(got.float(), ref.float(), rtol=1e-2, atol=1e-2), (
            f"group {g}: max err = {(got.float() - ref.float()).abs().max().item()}"
        )
    print(f"OK  tasks = {2 * n_tiles}  (AIC x{CUBE_CORES} + AIV x{AIV_CORES} saturated)")


if __name__ == "__main__":
    main()
