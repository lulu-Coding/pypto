# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""INT8 group matmul, scope-split variant of the n-tile optimization.

Same contract as 05/05b (INT8 x INT8 -> INT32, manual deps, A5 tiles):

    x  : [GROUP*M, K]  INT8     w: [GROUP*K, N] INT8     out: [GROUP*M, N] INT32

What changes vs 05b:

    05b keeps all N_TASKS submits in ONE manual scope. A scope's task ring
    only reclaims slots at scope_end, so once the task count passes the
    runtime's ring_task_window the ring saturates and the run deadlocks
    (SCOPE_DEADLOCK, orch_error_code=1) -- 05b papered over it with the
    PTO2_RING_TASK_WINDOW env knob, which is runtime-version-specific.

    This variant opens one manual_scope PER ROW STRIP, inside the parallel
    loop: at most N_COL_TILES (12) tasks are live per scope, so the default
    ring window is always sufficient -- no env vars, no config overrides,
    portable across runtime versions.

    Task shape is 05b's: one [TILE_M, TILE_N] output tile per task over full
    K (single software-pipeline group per kernel, so the L0B double-buffer
    fits without co-resident competition -- the PH-MR-001 fix).

Run:
    python examples/advanced/05c_group_matmul_i8_scopesplit.py             # a5
    python examples/advanced/05c_group_matmul_i8_scopesplit.py -p a5sim
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
# ===================================================

L0C_BYTES = 256 * 1024                   # A5 Acc (L0C)
MAT_BYTES = 512 * 1024                   # A5 Mat (L1)
CUBE_CORES = 18
READY_QUEUE_CEILING = 32768              # HBG single-queue population limit

N_ROW_TILES = GROUP * M // TILE_M
N_COL_TILES = N // TILE_N
N_TASKS = N_ROW_TILES * N_COL_TILES
ROWS_X = GROUP * M
ROWS_W = GROUP * K

assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0, "tiles must divide their dims"
assert K // TILE_K >= 2
for _d in (M, K, N, TILE_M, TILE_K, TILE_N):
    assert _d % 16 == 0, f"{_d} is not a multiple of 16 (cube fractal alignment)"
assert TILE_M * TILE_N * 4 <= L0C_BYTES, "acc tile exceeds A5 L0C (256KB)"
assert TILE_M * TILE_N * 8 <= L0C_BYTES, "acc tile > L0C/2: dbC ping-pong will not fit"
assert 2 * (TILE_M * TILE_K + TILE_K * TILE_N) <= MAT_BYTES, \
    "K panels x pipeline double-buffer exceed Mat (512KB); lower TILE_K"
assert N_TASKS <= READY_QUEUE_CEILING, \
    f"{N_TASKS} tasks exceed the {READY_QUEUE_CEILING} ready-queue ceiling; enlarge tiles"
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"


@pl.program
class GroupMatmulI8ScopeSplit:
    @pl.function(type=pl.FunctionType.InCore)
    def gemm_ntile(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.InOut[pl.Tensor[[ROWS_X, N], pl.INT32]],
        row: pl.Scalar[pl.INDEX],
        n0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        """Cube: ONE [TILE_M, TILE_N] output tile over full K.

        Single pipeline group per kernel: the [TILE_K, TILE_N] Right chunks
        double-buffer in L0B without co-resident competition. Head-peel form
        (if/else peel is illegal in an InCore kernel: Acc phi).
        """
        wbase = (row // M) * K
        a = x[row : row + TILE_M, 0:TILE_K]
        b = w[wbase : wbase + TILE_K, n0 : n0 + TILE_N]
        acc = pl.matmul(a, b, out_dtype=pl.INT32)
        for kb in pl.pipeline(1, K // TILE_K, stage=2):
            k0 = kb * TILE_K
            a = x[row : row + TILE_M, k0 : k0 + TILE_K]
            b = w[wbase + k0 : wbase + k0 + TILE_K, n0 : n0 + TILE_N]
            acc = pl.matmul_acc(acc, a, b)
        return pl.assemble(out, acc, [row, n0])

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.Out[pl.Tensor[[ROWS_X, N], pl.INT32]],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        # One manual scope PER STRIP: at most N_COL_TILES tasks live in any
        # single scope, so the default ring_task_window always suffices and
        # slots are reclaimed at each strip's scope_end. No env knobs.
        for t in pl.parallel(0, N_ROW_TILES):
            row = t * TILE_M
            with pl.manual_scope():
                for n_i in pl.range(N_COL_TILES):
                    n0 = n_i * TILE_N
                    out, _ = pl.submit(self.gemm_ntile, x, w, out, row, n0)
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
    out = torch.zeros(GROUP * M, N, dtype=torch.int32)

    print(f"compiling: {N_TASKS} gemm tasks (scope split per strip) ...")
    t0 = time.perf_counter()
    compiled = ir.compile(GroupMatmulI8ScopeSplit, backend_type=_BACKENDS[args.platform])
    print(f"compiled in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    compiled(x, w, out, config=cfg)
    print(f"done in {time.perf_counter() - t0:.1f}s")

    print("verifying per group (FP64 reference is exact: |acc| < 2^53) ...")
    for g in range(GROUP):
        ref = (x[g * M : (g + 1) * M].double() @ w[g * K : (g + 1) * K].double()).to(torch.int32)
        got = out[g * M : (g + 1) * M]
        assert torch.equal(got, ref), (
            f"group {g}: max err = {(got - ref).abs().max().item()}"
        )
    print(f"OK  bit-exact  tasks = {N_TASKS}  (AIC x{CUBE_CORES} saturated)")


if __name__ == "__main__":
    main()
