# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""INT8 group matmul, N-BLOCK pipeline (06 variant G) -- retry of the 06c attempt.

In 06 the outer ``for n_i in pl.range(N // TILE_N)`` is a serial loop: each
n-block does a head-peel panel load, a k-pipeline, then a store. The next
n-block's first panel load therefore cannot overlap the current n-block's tail
(or its store).  AscendC's preload-async does exactly that overlap -- it queues
L1 tiles so the load for one output block overlaps the drain of the previous.

06g is the minimal way to ask the compiler for it: keep 06's kernel body
byte-for-byte, but turn the outer n-loop into ``pl.pipeline(..., stage=2)`` so
the n-transitions pipeline too (now a nest: inner k-pipeline inside an outer
n-pipeline).

*** This is the 06c experiment.  06c failed to compile with a TileView parser
*** error, but that predates the PH-MR-001 pipeline fix (1b41affe), so it is
*** worth re-testing.  If it fails again, capture the exact parser error.

Shape: M = GROUP_M * GROUP = 8192 TOTAL rows (256 per group, NOT per-group 8192).

Run:
    python examples/advanced/06g_group_matmul_i8_npipe.py -p a5sim
    python examples/advanced/06g_group_matmul_i8_npipe.py            # a5
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
L1_K = 256                              # half 06's panel: nested n*k buffering must fit Mat
N_STAGE = 2                             # n-block pipeline depth
# ===================================================

L0C_BYTES = 256 * 1024
L0A_BYTES = 64 * 1024
L0B_BYTES = 64 * 1024
MAT_BYTES = 512 * 1024
CUBE_CORES = 18

N_STRIPS = M // TILE_M                   # 32 strips (one per group)
ROWS_X = M                               # 8192
ROWS_W = GROUP * K                       # 114688

assert M % TILE_M == 0 and N % TILE_N == 0 and K % L1_K == 0 and L1_K % L0_K == 0
assert K // L1_K >= 2
for _d in (M, K, N, TILE_M, TILE_N, L0_K, L1_K):
    assert _d % 16 == 0, f"{_d} is not a multiple of 16"
assert L0_K == (L0B_BYTES // 2) // (TILE_N * 1) & ~15, "L0_K must follow the arch35 derivation"
assert TILE_M * TILE_N * 4 <= L0C_BYTES, "acc block exceeds A5 L0C (256KB)"
assert 2 * (TILE_M * L0_K) <= L0A_BYTES, "L0A double-buffer does not fit"
assert 2 * (L0_K * TILE_N) <= L0B_BYTES, "L0B double-buffer does not fit"
# A nest of two pipelines multiplies their buffering: the panel held per (n,k)
# step is duplicated N_STAGE x (inner k stage = 2) times.  At L1_K=512 that is
# 2*2*256KB = 1MB and the old compiler rejected it ("Mat buffer usage (1048576)
# exceeds platform limit (524288)"); L1_K=256 brings it to 2*2*128KB = 512KB.
assert N_STAGE * 2 * (TILE_M * L1_K + L1_K * TILE_N) <= MAT_BYTES, \
    "nested n/k pipeline buffers exceed L1; lower L1_K or N_STAGE"
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"


@pl.program
class GroupMatmulI8NPipe:
    @pl.function(type=pl.FunctionType.InCore)
    def gemm_strip(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.InOut[pl.Tensor[[ROWS_X, N], pl.INT32]],
        row: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        """Cube: one [TILE_M, N] strip. 06 body; outer n-loop pipelined."""
        wbase = (row // GROUP_M) * K
        for n_i in pl.pipeline(0, N // TILE_N, stage=N_STAGE):
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

    print(f"compiling: {N_STRIPS} strip tasks, n-pipeline stage={N_STAGE}, "
          f"block {TILE_M}x{TILE_N}x{L0_K}, L1 panel K={L1_K} ...")
    t0 = time.perf_counter()
    compiled = ir.compile(GroupMatmulI8NPipe, backend_type=_BACKENDS[args.platform])
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
    print(f"OK  bit-exact  tasks = {N_STRIPS}  block = {TILE_M}x{TILE_N}x{L0_K}  "
          f"n-stage = {N_STAGE}  total = {M * K * N * 2 / 1e9:.1f} GFLOP  (AIC x{CUBE_CORES})")


if __name__ == "__main__":
    main()
