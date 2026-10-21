# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resident-block SPMD INT8 group matmul: compile-time core assignment.

The AscendC dispatch model: usedCoreNum blocks, ONE RESIDENT PER CORE, each
looping over its share of the block grid inside the kernel. No per-strip
dispatch overhead (single spmd_submit launch).

    pl.spmd_submit(core_num=18) — 18 blocks, one per cube core
    get_block_idx() → core ID (0..17)
    Grid-stride: strip = core_idx + i * 18

Shape: M = GROUP_M * GROUP = 8192 total (256 per group × 32 groups).

Run:
    python examples/advanced/08_group_matmul_i8_resident.py             # a5
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
GROUP_M = 256                            # rows per group
K, N = 3584, 3072
M = GROUP_M * GROUP                      # 8192 total
TILE_M, TILE_N = 128, 256
L0_K = 128
L1_K = 256
# ======================================

L0C_BYTES = 256 * 1024
L0A_BYTES = 64 * 1024
L0B_BYTES = 64 * 1024
MAT_BYTES = 512 * 1024
CUBE_CORES = 18

N_STRIPS = M // TILE_M                   # 64 strips
N_K_CHUNKS = K // L0_K
N_TILES_N = N // TILE_N
ROWS_X = M
ROWS_W = GROUP * K

assert M % TILE_M == 0 and N % TILE_N == 0 and K % L1_K == 0 and L1_K % L0_K == 0
for _d in (M, K, N, TILE_M, TILE_N, L0_K, L1_K):
    assert _d % 16 == 0
assert TILE_M * TILE_N * 4 <= L0C_BYTES
assert 2 * (TILE_M * L0_K) <= L0A_BYTES
assert 2 * (L0_K * TILE_N) <= L0B_BYTES
assert 2 * (TILE_M * L1_K + L1_K * TILE_N) <= MAT_BYTES
assert 127 * 127 * K < 2**31


@pl.program
class GroupMatmulI8Resident:
    @pl.function(type=pl.FunctionType.InCore)
    def gemm_block(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.InOut[pl.Tensor[[ROWS_X, N], pl.INT32]],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        """One RESIDENT block = one cube core, grid-stride over strips."""
        core = pl.tile.get_block_idx()
        cnt = (N_STRIPS - core + CUBE_CORES - 1) // CUBE_CORES
        for i in pl.range(cnt):
            s = core + i * CUBE_CORES
            row = s * TILE_M
            wbase = (row // GROUP_M) * K
            for n_i in pl.range(N_TILES_N):
                n0 = n_i * TILE_N
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
                for kk in pl.range(1, L1_K // L0_K):
                    k0 = kk * L0_K
                    acc = pl.tile.matmul_acc(
                        acc,
                        pl.tile.extract(a_panel, 0, k0, [TILE_M, L0_K],
                                        target_memory=pl.MemorySpace.Left),
                        pl.tile.extract(b_panel, k0, 0, [L0_K, TILE_N],
                                        target_memory=pl.MemorySpace.Right),
                    )
                for kb in pl.pipeline(1, K // L1_K, stage=2):
                    k1 = kb * L1_K
                    a_panel = pl.load(x, [row, k1], [TILE_M, L1_K],
                                      target_memory=pl.MemorySpace.Mat)
                    b_panel = pl.load(w, [wbase + k1, n0], [L1_K, TILE_N],
                                      target_memory=pl.MemorySpace.Mat)
                    for kk in pl.range(L1_K // L0_K):
                        k0 = kk * L0_K
                        a_l0 = pl.tile.extract(a_panel, 0, k0, [TILE_M, L0_K],
                                               target_memory=pl.MemorySpace.Left)
                        b_l0 = pl.tile.extract(b_panel, k0, 0, [L0_K, TILE_N],
                                               target_memory=pl.MemorySpace.Right)
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
        out, _ = pl.spmd_submit(self.gemm_block, x, w, out, core_num=CUBE_CORES)
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

    print(f"compiling: {CUBE_CORES} resident blocks, ~{N_STRIPS // CUBE_CORES} strips each ...")
    t0 = time.perf_counter()
    compiled = ir.compile(GroupMatmulI8Resident, backend_type=_BACKENDS[args.platform])
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
    print(f"OK  bit-exact  {CUBE_CORES} resident blocks  "
          f"total = {M * K * N * 2 / 1e9:.1f} GFLOP  (AIC x{CUBE_CORES})")


if __name__ == "__main__":
    main()
