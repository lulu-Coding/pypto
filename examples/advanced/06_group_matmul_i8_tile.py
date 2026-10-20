# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile-level INT8 group matmul on A5, translated from the AscendC reference.

Reference: hicann/ops-transformer gmm/grouped_matmul, arch35 (Ascend 950) tiling
(grouped_no_quant_matmul_tiling.cpp) and the BlockMmad preload-async loop
(block_mmad_preload_async_with_callback_pergroup.hpp). The mapping:

    AscendC (arch35, INT8)                     this file (PyPTO tile API)
    -----------------------                    ----------------------------
    base block 256x256x128                     TILE_M=128(sim-safe)/256(Huawei), TILE_N=256, L0_K=128
      baseK = L0B/2 / (baseN*1B) = 128           (same derivation, see asserts)
      baseM capped by L0A/2 and L0C/4            (same)
    L1 staging: A/B panels, DB at              L1_K-deep panels loaded into Mat
      panel depth stepK (=4 -> 512 K)            inside pl.pipeline(stage=2) --
      L1_STAGES slots + MTE1_MTE2 flags          LowerPipelineLoops gives the
                                                 load/extract chain depth-2
                                                 membership (the flag discipline
                                                 is the compiler's job here)
    L0A/L0B double buffer, M_MTE1 flags        tile.extract -> Left/Right inside
                                                 the same pipeline (depth 2)
    L0C single 256x256 slot (no dbC):          acc [256,256] INT32 = full L0C,
      full L0C, FIXPIPE drains at k-last         pl.store at the n-block end
    MMAD initC/unitFlag predicate              pl.tile.matmul_acc(acc, a, b,
                                                 kb == 0 and kk == 0)  -- the
                                                 predicated 4-arg form replaces
                                                 the if/else peel (Acc phi is
                                                 illegal in an InCore kernel)
    M x N block grid over aicNum cores         one task per [TILE_M, N] strip
                                                 (1024 tasks ~ 176us each, well
                                                 above the ~1us/task overhead)

    x  : [GROUP*M, K]  INT8   group g at rows [g*M, (g+1)*M)
    w  : [GROUP*K, N]  INT8   group g at rows [g*K, (g+1)*K)
    out: [GROUP*M, N]  INT32  raw INT32 (no dequant; add a vector epilogue on
                              top if needed)

L1_K=256 keeps the double-buffered panels at 256KB of Mat (safe margin);
Huawei's tiling derives stepK=4 (L1_K=512, exactly full L1) -- try 512 by
editing the config if the allocator accepts it (K % 512 == 0 here).

Run:
    python examples/advanced/06_group_matmul_i8_tile.py             # a5
    python examples/advanced/06_group_matmul_i8_tile.py -p a5sim
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
TILE_M, TILE_N = 128, 256               # sim-safe; 256x256 (Huawei exact) crashes this box's a5 sim
L0_K = 128                              # L0 chunk depth: L0B/2/(TILE_N*1B)
L1_K = 256                              # Mat panel depth (Huawei: 512)
# ===================================================

L0C_BYTES = 256 * 1024                  # A5 Acc (L0C)
L0A_BYTES = 64 * 1024                   # A5 Left (L0A)
L0B_BYTES = 64 * 1024                   # A5 Right (L0B)
MAT_BYTES = 512 * 1024                  # A5 Mat (L1)
CUBE_CORES = 18

N_STRIPS = GROUP * M // TILE_M
ROWS_X = GROUP * M
ROWS_W = GROUP * K

# AscendC arch35 tiling derivation + hardware bounds, checked at import.
assert M % TILE_M == 0 and N % TILE_N == 0 and K % L1_K == 0 and L1_K % L0_K == 0
assert K // L1_K >= 2
for _d in (M, K, N, TILE_M, TILE_N, L0_K, L1_K):
    assert _d % 16 == 0, f"{_d} is not a multiple of 16 (cube fractal alignment)"
assert L0_K == (L0B_BYTES // 2) // (TILE_N * 1) & ~15, "L0_K must follow the arch35 derivation"
assert TILE_M * TILE_N * 4 <= L0C_BYTES, "acc block exceeds A5 L0C (256KB)"
# Full-L0C single slot is the deliberate AscendC arch35 choice (no dbC); the
# L0C/2 assert from the earlier examples is intentionally NOT enforced here.
assert 2 * (TILE_M * L0_K) <= L0A_BYTES, "L0A double-buffer does not fit"
assert 2 * (L0_K * TILE_N) <= L0B_BYTES, "L0B double-buffer does not fit"
assert 2 * (TILE_M * L1_K + L1_K * TILE_N) <= MAT_BYTES, \
    "Mat panels x pipeline double-buffer exceed L1 (512KB); lower L1_K"
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"


@pl.program
class GroupMatmulI8Tile:
    @pl.function(type=pl.FunctionType.InCore)
    def gemm_strip(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.InOut[pl.Tensor[[ROWS_X, N], pl.INT32]],
        row: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        """Cube: one [TILE_M, N] strip = N/TILE_N blocks of 256x256xK.

        Per n-block: a full-K accumulation through the AscendC pipeline --
        Mat panels (MTE2) staged double-buffered by pl.pipeline, L1->L0
        extracts (MTE1) per L0_K chunk, predicated matmul_acc (MMAD), and a
        single FIXPIPE drain at the block end (L0C is one full 256x256 slot).
        """
        wbase = (row // M) * K
        for n_i in pl.range(N // TILE_N):
            n0 = n_i * TILE_N
            # Head-peel panel kb==0 (its first L0 chunk defines the accumulator
            # via tile.matmul -- the if/else peel form is illegal in an InCore
            # kernel because its phi merges two Acc-resident producers). The
            # init_values/yield_ canonical form is avoided too: this box's
            # MemoryReuse CreateTileMove alignment hits an INTERNAL_CHECK on
            # loop-carried tile vars (older pypto; pipeline(1, ...) + implicit
            # carry is the shape proven on real hardware by the 05 series).
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
                # GM -> Mat panels (MTE2), double-buffered by the pipeline
                a_panel = pl.load(x, [row, k1], [TILE_M, L1_K], target_memory=pl.MemorySpace.Mat)
                b_panel = pl.load(w, [wbase + k1, n0], [L1_K, TILE_N], target_memory=pl.MemorySpace.Mat)
                for kk in pl.range(L1_K // L0_K):
                    k0 = kk * L0_K
                    # Mat -> L0A/L0B (MTE1)
                    a_l0 = pl.tile.extract(a_panel, 0, k0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left)
                    b_l0 = pl.tile.extract(b_panel, k0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right)
                    # MMAD (accumulates; the head peel already initialised)
                    acc = pl.tile.matmul_acc(acc, a_l0, b_l0)
            # FIXPIPE: L0C -> GM (INT32 -> INT32, no conversion)
            out = pl.store(acc, [row, n0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.Out[pl.Tensor[[ROWS_X, N], pl.INT32]],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        # 1024 strip tasks, zero true dependencies (disjoint outputs): manual
        # scope because one-shot disjoint writes cannot ride the auto
        # tensormap under CHIP_MAX_FANIN=128.
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
    parser.add_argument("-p", "--platform", default="a5", help="target platform (default: a5)")
    parser.add_argument("-d", "--device", type=int, default=0, help="device id (default: 0)")
    args = parser.parse_args()

    cfg = RunConfig(platform=args.platform, device_id=args.device)
    torch.manual_seed(0)
    print("generating inputs ...")
    x = torch.randint(-127, 128, (GROUP * M, K), dtype=torch.int8)
    w = torch.randint(-127, 128, (GROUP * K, N), dtype=torch.int8)
    out = torch.zeros(GROUP * M, N, dtype=torch.int32)

    print(f"compiling: {N_STRIPS} strip tasks, block {TILE_M}x{TILE_N}x{L0_K}, L1 panel K={L1_K} ...")
    t0 = time.perf_counter()
    compiled = ir.compile(GroupMatmulI8Tile, backend_type=_BACKENDS[args.platform])
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
    print(f"OK  bit-exact  tasks = {N_STRIPS}  block = {TILE_M}x{TILE_N}x{L0_K}  (AIC x{CUBE_CORES})")


if __name__ == "__main__":
    main()
