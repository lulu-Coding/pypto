# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""INT8 group matmul: deep L1 panel (stepK=4) COMBINED with N-split dispatch.

Two previously-separate wins are merged here for the first time:

  06  (32be702b)  TILE_M=256 + L1_K=512 (stepK=4)        32 tasks   -> 344us  (best)
  06d (c0259587)  TILE_M=256 + N-split -> 64 tasks        L1_K=128  -> 349us

06 got the deep MTE2 prefetch (L1 panel = 4 L0 chunks) but stayed on 32 tasks
(32 / 18 cores = 1.78 waves).  06d fixed the load balance (64 tasks = 3.6 waves)
but paid for it with a shallow L1_K=128 panel (stepK=1, no nested k-loop).

06e = 06's exact per-task tiling (256x256x128, L1_K=512) with 06d's N-split
dispatch.  Per-task work is halved (6 n-blocks instead of 12) so the runtime has
finer units to feed 18 cores, while every L1 panel still feeds 4 L0 chunks.

Shape: M = GROUP_M * GROUP = 8192 TOTAL rows (256 per group, NOT per-group 8192).

    x  : [M, K]        INT8   group g at rows [g*GROUP_M, (g+1)*GROUP_M)
    w  : [GROUP*K, N]  INT8   group g at rows [g*K, (g+1)*K)
    out: [M, N]        INT32  raw INT32 (no dequant)

N_SPLITS is the knob: 1 = plain 06 behaviour (32 tasks), 2 = 64 tasks,
4 = 128 tasks (watch the dispatch cost ~0.8us/task).  Balance note: M/TILE_M=32
groups x N_SPLITS is never an exact multiple of 18 for uniform tasks, so expect
a ~12.5% tail-wave loss at every split; only the *dispatch* cost grows with
N_SPLITS.  If dispatch dominates, N_SPLITS=1 wins; if core feeding dominates,
N_SPLITS=2 does.

Run:
    python examples/advanced/06e_group_matmul_i8_deep_split.py             # a5
    python examples/advanced/06e_group_matmul_i8_deep_split.py -p a5sim
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
K, N = 3584, 3072
M = GROUP_M * GROUP                      # 8192 total rows
TILE_M, TILE_N = 256, 256               # AscendC arch35 exact: 256x256 fills L0C/L0A/L0B
L0_K = 128                              # L0 chunk depth: L0B/2/(TILE_N*1B)
L1_K = 512                              # AscendC arch35: stepK=4, deep MTE2 prefetch
N_SPLITS = 2                            # split each group's N across N_SPLITS tasks
# ===================================================

# ---- hardware budgets (A5 / Ascend950) ----
L0C_BYTES = 256 * 1024
L0A_BYTES = 64 * 1024
L0B_BYTES = 64 * 1024
MAT_BYTES = 512 * 1024
CUBE_CORES = 18

N_TILES_PER_TASK = (N // N_SPLITS) // TILE_N   # 6 n-blocks per task at split=2
N_TASKS = (M // TILE_M) * N_SPLITS             # 32 strips x split

ROWS_X = M                               # 8192
ROWS_W = GROUP * K                       # 114688

assert N % N_SPLITS == 0 and (N // N_SPLITS) % TILE_N == 0, "N split must align with TILE_N"
assert M % TILE_M == 0 and K % L1_K == 0 and L1_K % L0_K == 0
assert N_TILES_PER_TASK >= 1
assert K // L1_K >= 2
for _d in (M, K, N, TILE_M, TILE_N, L0_K, L1_K):
    assert _d % 16 == 0, f"{_d} is not a multiple of 16"
assert L0_K == (L0B_BYTES // 2) // (TILE_N * 1) & ~15, "L0_K must follow the arch35 derivation"
assert TILE_M * TILE_N * 4 <= L0C_BYTES, "acc block exceeds A5 L0C (256KB)"
assert 2 * (TILE_M * L0_K) <= L0A_BYTES, "L0A double-buffer does not fit"
assert 2 * (L0_K * TILE_N) <= L0B_BYTES, "L0B double-buffer does not fit"
assert 2 * (TILE_M * L1_K + L1_K * TILE_N) <= MAT_BYTES, \
    "Mat panels x pipeline double-buffer exceed L1; lower L1_K"
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"


@pl.program
class GroupMatmulI8DeepSplit:
    @pl.function(type=pl.FunctionType.InCore)
    def gemm_block(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.InOut[pl.Tensor[[ROWS_X, N], pl.INT32]],
        row: pl.Scalar[pl.INDEX],
        col: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        """Cube: one [TILE_M, N/N_SPLITS] block, AscendC arch35 deep pipeline."""
        wbase = (row // GROUP_M) * K
        col_end = col + (N // N_SPLITS)
        for n_i in pl.range(col // TILE_N, col_end // TILE_N):
            n0 = n_i * TILE_N
            # Head-peel panel kb==0 (its first L0 chunk defines the accumulator)
            a_panel = pl.load(x, [row, 0], [TILE_M, L1_K], target_memory=pl.MemorySpace.Mat)
            b_panel = pl.load(w, [wbase, n0], [L1_K, TILE_N], target_memory=pl.MemorySpace.Mat)
            acc = pl.tile.matmul(
                pl.tile.extract(a_panel, 0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                pl.tile.extract(b_panel, 0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
            )
            for kk in pl.range(1, L1_K // L0_K):
                k0 = kk * L0_K
                acc = pl.tile.matmul_acc(
                    acc,
                    pl.tile.extract(a_panel, 0, k0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                    pl.tile.extract(b_panel, k0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
                )
            for kb in pl.pipeline(1, K // L1_K, stage=2):
                k1 = kb * L1_K
                a_panel = pl.load(x, [row, k1], [TILE_M, L1_K], target_memory=pl.MemorySpace.Mat)
                b_panel = pl.load(w, [wbase + k1, n0], [L1_K, TILE_N], target_memory=pl.MemorySpace.Mat)
                for kk in pl.range(L1_K // L0_K):
                    k0 = kk * L0_K
                    a_l0 = pl.tile.extract(a_panel, 0, k0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left)
                    b_l0 = pl.tile.extract(b_panel, k0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right)
                    acc = pl.tile.matmul_acc(acc, a_l0, b_l0)
            out = pl.store(acc, [row, n0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.Out[pl.Tensor[[ROWS_X, N], pl.INT32]],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        with pl.manual_scope():
            for t in pl.parallel(0, N_TASKS):
                strip = t // N_SPLITS
                split = t % N_SPLITS
                row = strip * TILE_M
                col = split * (N // N_SPLITS)
                out, _ = pl.submit(self.gemm_block, x, w, out, row, col)
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

    print(f"compiling: {N_TASKS} tasks ({M // TILE_M} strips x {N_SPLITS} N-splits), "
          f"block {TILE_M}x{TILE_N}x{L0_K}, L1 panel K={L1_K} (stepK={L1_K // L0_K}) ...")
    t0 = time.perf_counter()
    compiled = ir.compile(GroupMatmulI8DeepSplit, backend_type=_BACKENDS[args.platform])
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
    print(f"OK  bit-exact  tasks = {N_TASKS}  block = {TILE_M}x{TILE_N}x{L0_K}  "
          f"total = {M * K * N * 2 / 1e9:.1f} GFLOP  (AIC x{CUBE_CORES})")


if __name__ == "__main__":
    main()
