# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SPMD-fanout variant of the tile-level INT8 group matmul (06 -> 07).

Same tile-level kernel body as 06 (AscendC arch35 translation: Mat panels in
a stage-2 pipeline, tile.extract to L0A/L0B, head-peel + matmul_acc, one
FIXPIPE drain per n-block). What changes is the DISPATCH model:

    06: pl.submit per strip -- 2048 dispatch events through the scheduler
        (~0.8us each on the real A5, ~16% of the 10ms baseline wall).
    07: ONE pl.spmd_submit(core_num=N_STRIPS) -- the AscendC model: a single
        launch fans the strips out as blocks; the scheduler spreads them over
        the 18 AIC cores and each block derives its strip from
        tile.get_block_idx() (strip = block_idx * TILE_M -- multiplication
        only in the orchestration sense; the group's w-band offset divides
        inside the kernel, where the ccec path makes it legal).

With one task writing `out`, the auto tensormap carries a single edge and no
manual scope is needed at all.

Run:
    python examples/advanced/07_group_matmul_i8_spmd.py             # a5
    python examples/advanced/07_group_matmul_i8_spmd.py -p a5sim
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
TILE_M, TILE_N = 128, 256               # sim-safe; 256x256 (Huawei exact) crashes the old a5sim
L0_K = 128                              # L0 chunk depth: L0B/2/(TILE_N*1B)
L1_K = 256                              # Mat panel depth (Huawei: 512)
# ===================================================

L0C_BYTES = 256 * 1024
L0A_BYTES = 64 * 1024
L0B_BYTES = 64 * 1024
MAT_BYTES = 512 * 1024
CUBE_CORES = 18

N_STRIPS = GROUP * M // TILE_M
ROWS_X = GROUP * M
ROWS_W = GROUP * K

assert M % TILE_M == 0 and N % TILE_N == 0 and K % L1_K == 0 and L1_K % L0_K == 0
assert K // L1_K >= 2
for _d in (M, K, N, TILE_M, TILE_N, L0_K, L1_K):
    assert _d % 16 == 0, f"{_d} is not a multiple of 16 (cube fractal alignment)"
assert L0_K == (L0B_BYTES // 2) // (TILE_N * 1) & ~15, "L0_K must follow the arch35 derivation"
assert TILE_M * TILE_N * 4 <= L0C_BYTES, "acc block exceeds A5 L0C (256KB)"
assert 2 * (TILE_M * L0_K) <= L0A_BYTES, "L0A double-buffer does not fit"
assert 2 * (L0_K * TILE_N) <= L0B_BYTES, "L0B double-buffer does not fit"
assert 2 * (TILE_M * L1_K + L1_K * TILE_N) <= MAT_BYTES, \
    "Mat panels x pipeline double-buffer exceed L1 (512KB); lower L1_K"
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"


@pl.program
class GroupMatmulI8Spmd:
    @pl.function(type=pl.FunctionType.InCore)
    def gemm_strip(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.InOut[pl.Tensor[[ROWS_X, N], pl.INT32]],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        """One SPMD block = one [TILE_M, N] strip (same body as 06)."""
        t = pl.tile.get_block_idx()
        row = t * TILE_M
        wbase = (row // M) * K
        for n_i in pl.range(N // TILE_N):
            n0 = n_i * TILE_N
            # Head-peel panel kb==0: its first L0 chunk defines the accumulator
            # (if/else peel is illegal in an InCore kernel -- Acc phi).
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
        # ONE dispatch fans the strips out as blocks; the scheduler spreads
        # them over the cube cores. Single task -> single out edge -> no
        # manual scope needed.
        out, _ = pl.spmd_submit(self.gemm_strip, x, w, out, core_num=N_STRIPS)
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

    print(f"compiling: 1 spmd dispatch x {N_STRIPS} blocks, block {TILE_M}x{TILE_N}x{L0_K}, "
          f"L1 panel K={L1_K} ...")
    t0 = time.perf_counter()
    compiled = ir.compile(GroupMatmulI8Spmd, backend_type=_BACKENDS[args.platform])
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
    print(f"OK  bit-exact  1 dispatch x {N_STRIPS} blocks  (AIC x{CUBE_CORES})")


if __name__ == "__main__":
    main()
