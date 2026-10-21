# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Manual-flag INT8 group matmul: pair-unrolled MTE1/MAD + FIXPIPE/MTE2 overlap.

Requires PR #2997 (pl.system.set_flag / pl.system.wait_flag).

KEY INSIGHT: in SSA, you can't reference "the previous loop iteration's variable"
to implement ping-pong double-buffering. The solution is to UNROLL the K-loop
by 2 and process chunks in PAIRS, with explicit SSA variables for even/odd:

    Per pair (chunks 2i, 2i+1):
        1. MTE1: extract even chunk → L0B slot 0     (async, returns immediately)
        2. MTE1: extract odd chunk  → L0B slot 1     (async, pipelined after #1)
        3. Cube: wait_flag(slot 0 ready) → MAD on even chunk
           ← While cube computes on slot 0, MTE1 is still filling slot 1!
        4. Cube: wait_flag(slot 1 ready) → MAD on odd chunk
           ← By now slot 1 is ready (MTE1 had time during #3)

    This gives TRUE MTE1/MAD overlap without loop-carried variables.

FIXPIPE overlap: after the last chunk's MAD, set_flag(M→FIX) and store.
The store is async (FIXPIPE engine drains L0C). The next n-block's
b_panel load (MTE2) starts immediately — before FIXPIPE finishes.

Flag events (user range [0,13]):
    EV_L0B_0 = 0: L0B slot 0 (MTE1→M: data ready, M→MTE1: slot free)
    EV_L0B_1 = 1: L0B slot 1 (same handshake)
    EV_L0C   = 2: L0C buffer (M→FIX: drain ready, FIX→M: buffer free)

Run:
    python examples/advanced/09_group_matmul_i8_flags.py             # a5
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
M = GROUP_M * GROUP
TILE_M, TILE_N = 128, 256
L0_K = 128
# ======================================

L0C_BYTES = 256 * 1024
L0B_BYTES = 64 * 1024
CUBE_CORES = 18

N_STRIPS = M // TILE_M
N_K_CHUNKS = K // L0_K                   # 28 K-chunks
N_PAIRS = N_K_CHUNKS // 2               # 14 pairs
N_TILES_N = N // TILE_N                  # 12 n-blocks
ROWS_X = M
ROWS_W = GROUP * K

assert M % TILE_M == 0 and N % TILE_N == 0 and K % L0_K == 0
assert N_K_CHUNKS % 2 == 0, "K-chunk count must be even for pair unrolling"
for _d in (M, K, N, TILE_M, TILE_N, L0_K):
    assert _d % 16 == 0
assert TILE_M * TILE_N * 4 <= L0C_BYTES
assert 2 * (L0_K * TILE_N) <= L0B_BYTES
assert 127 * 127 * K < 2**31

EV_L0B_0 = 0
EV_L0B_1 = 1
EV_L0C = 2


@pl.program
class GroupMatmulI8Flags:
    @pl.function(type=pl.FunctionType.InCore)
    def gemm_strip(
        self,
        x: pl.Tensor[[ROWS_X, K], pl.INT8],
        w: pl.Tensor[[ROWS_W, N], pl.INT8],
        out: pl.InOut[pl.Tensor[[ROWS_X, N], pl.INT32]],
        row: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[ROWS_X, N], pl.INT32]:
        """One [TILE_M, N] strip, pair-unrolled with explicit flags."""
        wbase = (row // GROUP_M) * K

        # --- Initialize: all L0B slots and L0C free ---
        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=EV_L0B_0)
        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=EV_L0B_1)
        pl.system.set_flag(pl.PipeType.FIX, pl.PipeType.M, event_id=EV_L0C)

        for n_i in pl.range(N_TILES_N):
            n0 = n_i * TILE_N

            # === MTE2: load b_panel (async — overlaps with previous
            #     n-block's FIXPIPE drain) ===
            b_panel = pl.load(w, [wbase, n0], [L0_K, TILE_N],
                              target_memory=pl.MemorySpace.Mat)

            # === Head: first K-chuck (chunk 0) → L0B slot 0 ===
            pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=EV_L0B_0)
            b_first = pl.tile.extract(b_panel, 0, 0, [L0_K, TILE_N],
                                      target_memory=pl.MemorySpace.Right)
            pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, event_id=EV_L0B_0)

            # MAD on first chunk (defines accumulator)
            a_l0 = pl.tile.extract(
                pl.load(x, [row, 0], [TILE_M, L0_K],
                        target_memory=pl.MemorySpace.Mat),
                0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left)
            pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, event_id=EV_L0B_0)
            acc = pl.tile.matmul(a_l0, b_first)
            pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=EV_L0B_0)

            # === Pair-unrolled steady-state (chunks 1..27, two at a time) ===
            # Each pair: (odd chunk → slot 1, even chunk → slot 0)
            # MAD on even chunk overlaps with MTE1 filling odd chunk's slot.
            for kp in pl.range(N_PAIRS):
                k_even = (2 + kp * 2) * L0_K      # chunk index 2,4,6,...
                k_odd = (3 + kp * 2) * L0_K       # chunk index 3,5,7,...

                # --- MTE1: issue BOTH extracts back-to-back (async) ---
                # Even chunk → slot 0
                pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=EV_L0B_0)
                b_even = pl.tile.extract(b_panel, k_even, 0, [L0_K, TILE_N],
                                         target_memory=pl.MemorySpace.Right)
                pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, event_id=EV_L0B_0)

                # Odd chunk → slot 1 (MTE1 pipelines after even chunk)
                pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=EV_L0B_1)
                b_odd = pl.tile.extract(b_panel, k_odd, 0, [L0_K, TILE_N],
                                        target_memory=pl.MemorySpace.Right)
                pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, event_id=EV_L0B_1)

                # --- Cube: compute on EVEN chunk (slot 0) ---
                # While cube does this, MTE1 is filling slot 1 (b_odd)!
                pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, event_id=EV_L0B_0)
                acc = pl.tile.matmul_acc(acc, a_l0, b_even)
                pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=EV_L0B_0)

                # --- Cube: compute on ODD chunk (slot 1) ---
                # By now, slot 1 is ready (MTE1 had time during the even MAD)
                pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, event_id=EV_L0B_1)
                acc = pl.tile.matmul_acc(acc, a_l0, b_odd)
                pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=EV_L0B_1)

            # === FIXPIPE: drain L0C (async) ===
            pl.system.wait_flag(pl.PipeType.FIX, pl.PipeType.M, event_id=EV_L0C)
            out = pl.store(acc, [row, n0], out)
            pl.system.set_flag(pl.PipeType.M, pl.PipeType.FIX, event_id=EV_L0C)
            # Do NOT wait for FIXPIPE here — next n-block's MTE2 starts now.

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

    print(f"compiling: {N_STRIPS} strips, pair-unrolled flags "
          f"({N_K_CHUNKS} chunks in {N_PAIRS} pairs) ...")
    t0 = time.perf_counter()
    compiled = ir.compile(GroupMatmulI8Flags, backend_type=_BACKENDS[args.platform])
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
    print(f"OK  bit-exact  tasks = {N_STRIPS}  pair-unrolled flags  "
          f"total = {M * K * N * 2 / 1e9:.1f} GFLOP  (AIC x{CUBE_CORES})")


if __name__ == "__main__":
    main()
