# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile-level INT8 group matmul: dual-n-block pipeline, L0B depth-2 fix.

The 331-354us bottleneck is L0B pipeline depth 1 (PH-MR-001): TILE_N=256
needs 2x128x256 = 64KB per pipeline group = full L0B, so only ONE group
fits and MTE1 serializes with MMAD (measured 1.58x slowdown, matching the
15.2 vs 25.1 TMAC/s/core gap to AscendC).

This variant uses TILE_N=128 so each group needs only 32KB of L0B, allowing
TWO n-blocks to pipeline simultaneously. Additionally, both n-blocks share
the same a-panel L0A extract -- halving L0A traffic per MMAD pair.

Resource budget (A5, INT8):
    L0C: 2 accs [128, 128] x 4B       = 128KB  (half of 256KB, dbC-capable)
    L0A: 2 stages x [128, 128] x 1B   =  32KB  (half of 64KB)
    L0B: 2 groups x 2 stages x 16KB   =  64KB  (exactly full, depth 2 fits)
    Mat: 2 stages x (128+128)KB       = 256KB  (half of 512KB)

Run:
    python examples/advanced/06b_group_matmul_i8_dual.py             # a5
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

TILE_M = 128
TILE_N = 128                             # narrow enough for 2 L0B groups
L0_K = 128                               # L0 chunk depth
L1_K = 512                               # Mat panel depth (deep MTE2 prefetch)
# ======================================

L0C_BYTES = 256 * 1024
L0A_BYTES = 64 * 1024
L0B_BYTES = 64 * 1024
MAT_BYTES = 512 * 1024
CUBE_CORES = 18

N_STRIPS = M // TILE_M                   # 64 strips
N_PAIRS = N // (TILE_N * 2)              # 12 pairs of n-blocks (2 at a time)
K_L1_ITERS = K // L1_K                   # 7 L1 panels
K_L0_ITERS = L1_K // L0_K                # 4 L0 chunks per panel
ROWS_X = M
ROWS_W = GROUP * K

assert M % TILE_M == 0 and N % (TILE_N * 2) == 0 and K % L1_K == 0 and L1_K % L0_K == 0
for _d in (M, K, N, TILE_M, TILE_N, L0_K, L1_K):
    assert _d % 16 == 0, f"{_d} is not a multiple of 16"
# L0B: 2 n-blocks x 2 stages x L0_K*TILE_N bytes must fit
assert 2 * 2 * (L0_K * TILE_N) <= L0B_BYTES, \
    f"2 n-block groups x 2 stages x {L0_K}x{TILE_N} = {4 * L0_K * TILE_N // 1024}KB > L0B {L0B_BYTES // 1024}KB; shrink TILE_N or L0_K"
# L0A: 2 stages x TILE_M*L0_K must fit
assert 2 * (TILE_M * L0_K) <= L0A_BYTES, "L0A double-buffer does not fit"
# L0C: 2 accumulators
assert 2 * (TILE_M * TILE_N * 4) <= L0C_BYTES, "2 accs do not fit in L0C"
# Mat: 2 stages x (a + b0 + b1 panels)
assert 2 * (TILE_M * L1_K + 2 * L1_K * TILE_N) <= MAT_BYTES, \
    f"Mat panels exceed L1; lower L1_K"
assert 127 * 127 * K < 2**31


@pl.program
class GroupMatmulI8Dual:
    @pl.function(type=pl.FunctionType.InCore)
    def gemm_strip(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.InOut[pl.Tensor[[ROWS_X, N], pl.INT32]],
        row: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        """Cube: one [TILE_M, N] strip, processing n-blocks in PAIRS.

        Per pair: two [TILE_M, TILE_N] accumulators live simultaneously,
        sharing the same a-panel L0A extract. The K-pipeline loads a-panel
        once and TWO b-panels, then interleaves MMAD for both n-blocks --
        L0A traffic halved, and the two b-pipeline groups (32KB each) both
        fit in L0B for depth-2 overlap.
        """
        wbase = (row // GROUP_M) * K

        for pair_i in pl.range(N_PAIRS):
            n0 = pair_i * TILE_N * 2
            n1 = n0 + TILE_N

            # Head-peel: first L0 chunk of the first L1 panel defines accs
            a_panel = pl.load(x, [row, 0], [TILE_M, L1_K], target_memory=pl.MemorySpace.Mat)
            b0_panel = pl.load(w, [wbase, n0], [L1_K, TILE_N], target_memory=pl.MemorySpace.Mat)
            b1_panel = pl.load(w, [wbase, n1], [L1_K, TILE_N], target_memory=pl.MemorySpace.Mat)

            acc0 = pl.tile.matmul(
                pl.tile.extract(a_panel, 0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                pl.tile.extract(b0_panel, 0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
            )
            acc1 = pl.tile.matmul(
                pl.tile.extract(a_panel, 0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                pl.tile.extract(b1_panel, 0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
            )

            # Remaining L0 chunks of the head panel
            for kk in pl.range(1, K_L0_ITERS):
                k0 = kk * L0_K
                acc0 = pl.tile.matmul_acc(
                    acc0,
                    pl.tile.extract(a_panel, 0, k0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                    pl.tile.extract(b0_panel, k0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
                )
                acc1 = pl.tile.matmul_acc(
                    acc1,
                    pl.tile.extract(a_panel, 0, k0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                    pl.tile.extract(b1_panel, k0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
                )

            # K-pipeline for the remaining L1 panels
            for kb in pl.pipeline(1, K_L1_ITERS, stage=2):
                k1 = kb * L1_K
                a_panel = pl.load(x, [row, k1], [TILE_M, L1_K], target_memory=pl.MemorySpace.Mat)
                b0_panel = pl.load(w, [wbase + k1, n0], [L1_K, TILE_N], target_memory=pl.MemorySpace.Mat)
                b1_panel = pl.load(w, [wbase + k1, n1], [L1_K, TILE_N], target_memory=pl.MemorySpace.Mat)
                for kk in pl.range(K_L0_ITERS):
                    k0 = kk * L0_K
                    # Two MMADs per a-extract (the compiler CSEs the shared extract)
                    acc0 = pl.tile.matmul_acc(
                        acc0,
                        pl.tile.extract(a_panel, 0, k0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                        pl.tile.extract(b0_panel, k0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
                    )
                    acc1 = pl.tile.matmul_acc(
                        acc1,
                        pl.tile.extract(a_panel, 0, k0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                        pl.tile.extract(b1_panel, k0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
                    )

            # FIXPIPE: drain both accumulators
            out = pl.store(acc0, [row, n0], out)
            out = pl.store(acc1, [row, n1], out)
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

    print(f"compiling: {N_STRIPS} strips, dual-n-block "
          f"[{TILE_M}x{TILE_N}x{L0_K}] x2, L1_K={L1_K} ...")
    t0 = time.perf_counter()
    compiled = ir.compile(GroupMatmulI8Dual, backend_type=_BACKENDS[args.platform])
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
    print(f"OK  bit-exact  tasks = {N_STRIPS}  dual [{TILE_M}x{TILE_N}]  "
          f"total = {M * K * N * 2 / 1e9:.1f} GFLOP  (AIC x{CUBE_CORES})")


if __name__ == "__main__":
    main()
