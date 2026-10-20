# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""High-parallelism INT8 group matmul on A5: INT8 x INT8 -> INT32, manual deps.

Shape semantics (aligned with the user's group_matmul_autot.py):

    GROUP_M = 256 rows per group, G = 32 groups
    M = GROUP_M * G = 8192 TOTAL rows (NOT per-group M=8192)
    x  : [M, K]      INT8   group g's activations at rows [g*GROUP_M, (g+1)*GROUP_M)
    w  : [G*K, N]    INT8   group g's weights      at rows [g*K, (g+1)*K)
    out: [M, N]      INT32  = x @ w per group

    Total work = 8192 * 3584 * 3072 = 90.3 GMAC (NOT 2.89 TMAC -- earlier
    revisions of this file misread M as per-group, giving x=[262144, K]).

Parallel structure: one task per [TILE_M, N] row strip, zero dependencies
(disjoint outputs), manual scope (auto tensormap cannot carry one-shot
disjoint writes under CHIP_MAX_FANIN=128).

Run:
    python examples/advanced/05_group_matmul_i8.py                # a5, device 0
"""

import argparse
import time

import pypto.language as pl
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import RunConfig

# ============ shape config (edit freely) ============
GROUP = 32                               # number of groups
GROUP_M = 256                            # rows per group
K, N = 3584, 3072                        # contraction / output dims
M = GROUP_M * GROUP                      # 8192 TOTAL rows
TILE_M, TILE_K, TILE_N = 128, 512, 256   # acc 128KB = L0C/2; K panels x2 = 384KB Mat
# ===================================================

L0C_BYTES = 256 * 1024
MAT_BYTES = 512 * 1024
CUBE_CORES = 18

N_STRIPS = M // TILE_M                   # 64 strips (not 2048!)
ROWS_X = M                               # x / out rows: 8192
ROWS_W = GROUP * K                       # w rows: 114688

assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0
assert K // TILE_K >= 2
for _d in (M, K, N, TILE_M, TILE_K, TILE_N):
    assert _d % 16 == 0, f"{_d} is not a multiple of 16"
assert TILE_M * TILE_N * 4 <= L0C_BYTES, "acc tile exceeds A5 L0C (256KB)"
assert TILE_M * TILE_N * 8 <= L0C_BYTES, "acc tile > L0C/2: dbC ping-pong will not fit"
assert 2 * (TILE_M * TILE_K + TILE_K * TILE_N) <= MAT_BYTES
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"


@pl.program
class GroupMatmulI8:
    @pl.function(type=pl.FunctionType.InCore)
    def gemm_strip(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.InOut[pl.Tensor[[ROWS_X, N], pl.INT32]],
        row: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        """Cube: full [TILE_M, N] strip, INT8 x INT8 -> INT32.

        The group's w-band offset is derived kernel-side (row // GROUP_M):
        orchestration-level FloorDiv lowers to arith.divsi on the MLIR index
        type, which is not a legal op.
        """
        wbase = (row // GROUP_M) * K
        for n_i in pl.range(N // TILE_N):
            n0 = n_i * TILE_N
            a = x[row : row + TILE_M, 0:TILE_K]
            b = w[wbase : wbase + TILE_K, n0 : n0 + TILE_N]
            acc = pl.matmul(a, b, out_dtype=pl.INT32)
            for kb in pl.pipeline(1, K // TILE_K, stage=2):
                k0 = kb * TILE_K
                a = x[row : row + TILE_M, k0 : k0 + TILE_K]
                b = w[wbase + k0 : wbase + k0 + TILE_K, n0 : n0 + TILE_N]
                acc = pl.matmul_acc(acc, a, b)
            out = pl.assemble(out, acc, [row, n0])
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.Out[pl.Tensor[[ROWS_X, N], pl.INT32]],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        with pl.manual_scope():
            for t in pl.parallel(0, N_STRIPS):
                row = t * TILE_M
                out, _ = pl.submit(self.gemm_strip, x, w, out, row)
        return out


_BACKENDS = {
    "a5": BackendType.Ascend950,
    "a5sim": BackendType.Ascend950,
    "a2a3": BackendType.Ascend910B,
    "a2a3sim": BackendType.Ascend910B,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    cfg = RunConfig(platform=args.platform, device_id=args.device)
    torch.manual_seed(0)
    print("generating inputs ...")
    x = torch.randint(-127, 128, (M, K), dtype=torch.int8)
    w = torch.randint(-127, 128, (GROUP * K, N), dtype=torch.int8)
    out = torch.zeros(M, N, dtype=torch.int32)

    print(f"compiling: {N_STRIPS} strip tasks, x [{M},{K}], w [{GROUP*K},{N}] ...")
    t0 = time.perf_counter()
    compiled = ir.compile(GroupMatmulI8, backend_type=_BACKENDS[args.platform])
    print(f"compiled in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    compiled(x, w, out, config=cfg)
    print(f"done in {time.perf_counter() - t0:.4f}s")

    print("verifying per group (FP64 reference is exact) ...")
    for g in range(GROUP):
        ref = (
            x[g * GROUP_M : (g + 1) * GROUP_M].double()
            @ w[g * K : (g + 1) * K].double()
        ).to(torch.int32)
        got = out[g * GROUP_M : (g + 1) * GROUP_M]
        assert torch.equal(got, ref), (
            f"group {g}: max err = {(got - ref).abs().max().item()}"
        )
    print(f"OK  bit-exact  tasks = {N_STRIPS}  "
          f"total = {M * K * N * 2 / 1e9:.1f} GFLOP  (AIC x{CUBE_CORES})")


if __name__ == "__main__":
    main()
