# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TILE_M=256 + N-split INT8 group matmul: better MMAD intensity + load balance.

Two changes from 06 (344us):

1. TILE_M 128 -> 256: MMAD intensity jumps from 85 to 128 MAC/B, matching
   AscendC's per-core throughput. Each MMAD is [256,128] @ [128,256] = 8.4M
   MACs (2x larger than 06's [128,128] @ [128,256]).

2. N-split dispatch: with TILE_M=256, M/TILE_M = 32 tasks (poor balance on
   18 cores = 1.78 waves). Splitting N in half gives 64 tasks = 3.6 waves.
   Each task processes [256, N/2] = 6 n-tiles of [256, 256].

Resource budget (INT8, A5):
    L0C: [256, 256] x 4B = 256KB (full, single slot, no dbC — same as AscendC)
    L0A: 2 stages x [256, 128] x 1B = 64KB (full)
    L0B: 2 stages x [128, 256] x 1B = 64KB (full)
    Mat: 2 stages x (256xL1_K + L1_Kx256) x 1B; L1_K=128 -> 128KB (25% of 512KB)

Run:
    python examples/advanced/06d_group_matmul_i8_m256.py             # a5
"""

import argparse
import time

import pypto.language as pl
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import RunConfig

# ============ shape config ============
GROUP = 32
GROUP_M = 256
K, N = 3584, 3072
M = GROUP_M * GROUP                      # 8192 total

TILE_M = 256                             # AscendC block size (2x MMAD intensity)
TILE_N = 256                             # one n-tile at a time
L0_K = 128                               # L0 chunk depth
L1_K = 128                               # Mat panel depth (L0_K-sized, no nested loop)
N_SPLITS = 2                             # split N into halves for load balance
# ======================================

L0C_BYTES = 256 * 1024
L0A_BYTES = 64 * 1024
L0B_BYTES = 64 * 1024
MAT_BYTES = 512 * 1024
CUBE_CORES = 18

N_TILES_N_HALF = (N // N_SPLITS) // TILE_N   # 6 n-tiles per half
N_K_CHUNKS = K // L0_K                       # 28 K-chunks per n-tile
N_TASKS = (M // TILE_M) * N_SPLITS           # 32 * 2 = 64 tasks
ROWS_X = M
ROWS_W = GROUP * K

assert M % TILE_M == 0 and N % TILE_N == 0 and K % L0_K == 0
assert (N // N_SPLITS) % TILE_N == 0, "N split must align with TILE_N"
for _d in (M, K, N, TILE_M, TILE_N, L0_K):
    assert _d % 16 == 0, f"{_d} is not a multiple of 16"
assert TILE_M * TILE_N * 4 <= L0C_BYTES, "acc exceeds L0C"
assert 2 * (TILE_M * L0_K) <= L0A_BYTES, "L0A double-buffer does not fit"
assert 2 * (L0_K * TILE_N) <= L0B_BYTES, "L0B double-buffer does not fit"
assert 2 * (TILE_M * L1_K + L1_K * TILE_N) <= MAT_BYTES, "Mat panels don't fit"
assert 127 * 127 * K < 2**31


@pl.program
class GroupMatmulI8M256:
    @pl.function(type=pl.FunctionType.InCore)
    def gemm_half(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.InOut[pl.Tensor[[ROWS_X, N], pl.INT32]],
        row: pl.Scalar[pl.INDEX],
        n_start: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        """Cube: one [TILE_M, N/2] half-strip = 6 n-tiles of [256, 256].

        Each n-tile: head-peel + K-pipeline over L0_K chunks.
        The n-loop is pl.range (sequential) — same structure as 06,
        just with TILE_M=256 and half-N range.
        """
        wbase = (row // GROUP_M) * K

        for n_i in pl.range(N_TILES_N_HALF):
            n0 = n_start + n_i * TILE_N

            # Head-peel: first K-chunk defines the accumulator
            a_panel = pl.load(x, [row, 0], [TILE_M, L1_K],
                              target_memory=pl.MemorySpace.Mat)
            b_panel = pl.load(w, [wbase, n0], [L1_K, TILE_N],
                              target_memory=pl.MemorySpace.Mat)
            acc = pl.tile.matmul(
                pl.tile.extract(a_panel, 0, 0, [TILE_M, L0_K],
                                target_memory=pl.MemorySpace.Left),
                pl.tile.extract(b_panel, 0, 0, [L0_K, TILE_N],
                                target_memory=pl.MemorySpace.Right),
            )

            # K-pipeline for remaining chunks
            for kc in pl.pipeline(1, N_K_CHUNKS, stage=2):
                k0 = kc * L0_K
                a_panel = pl.load(x, [row, k0], [TILE_M, L1_K],
                                  target_memory=pl.MemorySpace.Mat)
                b_panel = pl.load(w, [wbase + k0, n0], [L1_K, TILE_N],
                                  target_memory=pl.MemorySpace.Mat)
                a_l0 = pl.tile.extract(a_panel, 0, 0, [TILE_M, L0_K],
                                       target_memory=pl.MemorySpace.Left)
                b_l0 = pl.tile.extract(b_panel, 0, 0, [L0_K, TILE_N],
                                       target_memory=pl.MemorySpace.Right)
                acc = pl.tile.matmul_acc(acc, a_l0, b_l0)

            # FIXPIPE drain
            out = pl.store(acc, [row, n0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.Out[pl.Tensor[[ROWS_X, N], pl.INT32]],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        # 64 tasks: (32 groups) x (2 N-halves) on 18 cores = 3.6 waves
        # Each task: [256, N/2] output = 6 n-tiles of [256, 256]
        with pl.manual_scope():
            for t in pl.parallel(0, N_TASKS):
                strip = t // N_SPLITS
                half = t % N_SPLITS
                row = strip * TILE_M
                n_start = half * (N // N_SPLITS)
                out, _ = pl.submit(self.gemm_half, x, w, out, row, n_start)
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

    print(f"compiling: {N_TASKS} tasks ({M//TILE_M} strips x {N_SPLITS} N-halves), "
          f"block [{TILE_M}x{TILE_N}x{L0_K}] ...")
    t0 = time.perf_counter()
    compiled = ir.compile(GroupMatmulI8M256, backend_type=_BACKENDS[args.platform])
    print(f"compiled in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    compiled(x, w, out, config=cfg)
    print(f"done in {time.perf_counter() - t0:.4f}s")

    print("verifying per group ...")
    for g in range(GROUP):
        ref = (
            x[g * GROUP_M : (g + 1) * GROUP_M].double()
            @ w[g * K : (g + 1) * K].double()
        ).to(torch.int32)
        got = out[g * GROUP_M : (g + 1) * GROUP_M]
        assert torch.equal(got, ref), (
            f"group {g}: max err = {(got - ref).abs().max().item()}"
        )
    print(f"OK  bit-exact  tasks = {N_TASKS}  "
          f"block = {TILE_M}x{TILE_N}x{L0_K}  "
          f"total = {M * K * N * 2 / 1e9:.1f} GFLOP  (AIC x{CUBE_CORES})")


if __name__ == "__main__":
    main()
