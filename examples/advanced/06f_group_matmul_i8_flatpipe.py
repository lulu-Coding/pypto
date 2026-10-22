# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""INT8 group matmul, FLAT-K deep pipeline (06 variant F).

06 pipelines at the L1-PANEL granularity: per n-block it does a serial head-peel
(load panel kb=0, seed the accumulator) then ``pl.pipeline(1, K/L1_K, stage=2)``
over 7 more panels, each of which runs a nested ``pl.range`` over 4 L0 chunks.
That head-peel is un-pipelined, so one panel load per n-block cannot overlap
anything.

06f removes both the nesting and the head-peel asymmetry by moving the pipeline
down to the L0-chunk granularity: one flat ``pl.pipeline`` over all ``K/L0_K``
= 28 chunks, seeded once. Every chunk is a uniform GM->Mat load + L0 extract +
MAD, so the pipeliner sees a single regular loop and (with a deeper stage) can
keep that many chunks in flight -- at the cost of the L1 panel reuse that 06's
``L1_K=512`` panel buys (each Mat tile is now exactly one L0 chunk).

Mat budget for the flat loop: stage * (TILE_M*L0_K + L0_K*TILE_N) bytes.
But the pipeliner multi-buffers the L0 tiles as well, and at TILE_M=TILE_N=256,
L0_K=128 every chunk is 32KB in L0A and L0B, so L0 caps the stage at 2
(2*32KB = 64KB = the whole L0A/L0B).  stage=4 on the old compiler died with
"Right buffer usage (131072) exceeds platform limit (65536)".

Shape: M = GROUP_M * GROUP = 8192 TOTAL rows (256 per group, NOT per-group 8192).

Run:
    python examples/advanced/06f_group_matmul_i8_flatpipe.py -p a5sim
    python examples/advanced/06f_group_matmul_i8_flatpipe.py            # a5
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
FLAT_STAGE = 2                          # chunks in flight; L0 is the binding limit here
# ===================================================

L0C_BYTES = 256 * 1024
L0A_BYTES = 64 * 1024
L0B_BYTES = 64 * 1024
MAT_BYTES = 512 * 1024
CUBE_CORES = 18

N_STRIPS = M // TILE_M                   # 32 strips (one per group)
ROWS_X = M                               # 8192
ROWS_W = GROUP * K                       # 114688

assert M % TILE_M == 0 and N % TILE_N == 0 and K % L0_K == 0
for _d in (M, K, N, TILE_M, TILE_N, L0_K):
    assert _d % 16 == 0, f"{_d} is not a multiple of 16"
assert TILE_M * TILE_N * 4 <= L0C_BYTES, "acc block exceeds A5 L0C (256KB)"
assert 2 * (TILE_M * L0_K) <= L0A_BYTES, "L0A double-buffer does not fit"
assert 2 * (L0_K * TILE_N) <= L0B_BYTES, "L0B double-buffer does not fit"
# The flat pipeliner multi-buffers the L0 tiles too, so L0 (not Mat) caps the stage:
# FLAT_STAGE * TILE_M*L0_K <= L0A and FLAT_STAGE * L0_K*TILE_N <= L0B.
assert FLAT_STAGE * (TILE_M * L0_K) <= L0A_BYTES, "flat-pipeline L0A budget exceeds L0A; lower FLAT_STAGE"
assert FLAT_STAGE * (L0_K * TILE_N) <= L0B_BYTES, "flat-pipeline L0B budget exceeds L0B; lower FLAT_STAGE"
assert FLAT_STAGE * (TILE_M * L0_K + L0_K * TILE_N) <= MAT_BYTES, \
    "flat-pipeline Mat budget exceeds L1; lower FLAT_STAGE"
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"


@pl.program
class GroupMatmulI8FlatPipe:
    @pl.function(type=pl.FunctionType.InCore)
    def gemm_strip(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.InOut[pl.Tensor[[ROWS_X, N], pl.INT32]],
        row: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        """Cube: one [TILE_M, N] strip, flat deep L0-chunk pipeline."""
        wbase = (row // GROUP_M) * K
        for n_i in pl.range(N // TILE_N):
            n0 = n_i * TILE_N
            # Seed: the very first L0 chunk defines the accumulator.
            a_seed = pl.load(x, [row, 0], [TILE_M, L0_K], target_memory=pl.MemorySpace.Mat)
            b_seed = pl.load(w, [wbase, n0], [L0_K, TILE_N], target_memory=pl.MemorySpace.Mat)
            acc = pl.tile.matmul(
                pl.tile.extract(a_seed, 0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                pl.tile.extract(b_seed, 0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
            )
            # Flat pipeline over every remaining L0 chunk -- no nested panel loop.
            for kc in pl.pipeline(1, K // L0_K, stage=FLAT_STAGE):
                k0 = kc * L0_K
                a_p = pl.load(x, [row, k0], [TILE_M, L0_K], target_memory=pl.MemorySpace.Mat)
                b_p = pl.load(w, [wbase + k0, n0], [L0_K, TILE_N], target_memory=pl.MemorySpace.Mat)
                acc = pl.tile.matmul_acc(
                    acc,
                    pl.tile.extract(a_p, 0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                    pl.tile.extract(b_p, 0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
                )
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

    print(f"compiling: {N_STRIPS} strip tasks, flat L0 pipeline (stage={FLAT_STAGE}), "
          f"block {TILE_M}x{TILE_N}x{L0_K} ...")
    t0 = time.perf_counter()
    compiled = ir.compile(GroupMatmulI8FlatPipe, backend_type=_BACKENDS[args.platform])
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
          f"flat stage = {FLAT_STAGE}  total = {M * K * N * 2 / 1e9:.1f} GFLOP  (AIC x{CUBE_CORES})")


if __name__ == "__main__":
    main()
