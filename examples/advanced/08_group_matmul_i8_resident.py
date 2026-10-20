# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resident-block SPMD variant: the actual AscendC dispatch model (07 -> 08).

07 launched core_num=N_STRIPS (2048) blocks on 18 cores: the runtime
materialises one device task per block and dispatches them in WAVES
(114 wave transitions), which measured SLOWER than plain per-strip submits
(12.3ms vs 10ms). AscendC does the opposite: usedCoreNum blocks, ONE
RESIDENT PER CORE, each looping over its share of the M x N block grid
inside the kernel.

    08: pl.spmd_submit(core_num=CUBE_CORES) -- 18 blocks, one per cube core,
        no waves, 18 device tasks total. Each block round-robins over the
        strips (strip = block_idx + i * CUBE_CORES), the per-core strip count
        derived kernel-side with a floordiv (ccec path):

            cnt = (N_STRIPS - block_idx + CUBE_CORES - 1) // CUBE_CORES

        No guard-if needed (a runtime-bound pl.range covers the 113/114
        imbalance: 2048 = 18*113 + 14). Kernel body per strip is identical
        to 06/07.

Run:
    python examples/advanced/08_group_matmul_i8_resident.py             # a5
    python examples/advanced/08_group_matmul_i8_resident.py -p a5sim
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
class GroupMatmulI8Resident:
    @pl.function(type=pl.FunctionType.InCore)
    def gemm_block(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.InOut[pl.Tensor[[ROWS_X, N], pl.INT32]],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        """One RESIDENT block = one cube core, looping over its strips."""
        core = pl.tile.get_block_idx()
        cnt = (N_STRIPS - core + CUBE_CORES - 1) // CUBE_CORES
        for i in pl.range(cnt):
            s = core + i * CUBE_CORES          # round-robin: <=1 strip imbalance
            row = s * TILE_M
            wbase = (row // M) * K
            for n_i in pl.range(N // TILE_N):
                n0 = n_i * TILE_N
                # Head-peel panel kb==0 (its first L0 chunk defines the
                # accumulator; if/else peel is illegal -- Acc phi).
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
        # 18 resident blocks, one per cube core -- the AscendC dispatch model.
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
    parser.add_argument("-p", "--platform", default="a5", help="target platform (default: a5)")
    parser.add_argument("-d", "--device", type=int, default=0, help="device id (default: 0)")
    args = parser.parse_args()

    cfg = RunConfig(platform=args.platform, device_id=args.device)
    torch.manual_seed(0)
    print("generating inputs ...")
    x = torch.randint(-127, 128, (GROUP * M, K), dtype=torch.int8)
    w = torch.randint(-127, 128, (GROUP * K, N), dtype=torch.int8)
    out = torch.zeros(GROUP * M, N, dtype=torch.int32)

    print(f"compiling: {CUBE_CORES} resident blocks x ~{N_STRIPS // CUBE_CORES} strips each, "
          f"block {TILE_M}x{TILE_N}x{L0_K}, L1 panel K={L1_K} ...")
    t0 = time.perf_counter()
    compiled = ir.compile(GroupMatmulI8Resident, backend_type=_BACKENDS[args.platform])
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
    print(f"OK  bit-exact  {CUBE_CORES} resident blocks  (AIC x{CUBE_CORES})")


if __name__ == "__main__":
    main()
