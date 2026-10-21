# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Manual-flag INT8 group matmul v3: corrected flag direction semantics.

Key fix from v2: the M (cube) engine can only SET flags on channels where
M is the SOURCE (M→MTE1, M→FIX). It CANNOT set MTE1→M or FIX→M flags —
those are set by the respective hardware engines (MTE1 sets "data ready",
FIX sets "drain complete").

Corrected flag discipline:
  M→MTE1 (M signals MTE1):  "L0B slot is free, you can write"
     - Set by: our code (M engine) after matmul_acc finishes reading a slot
     - Waited by: MTE1 hardware engine (before starting next transfer)

  MTE1→M (MTE1 signals M):  "L0B slot has data, you can compute"
     - Set by: MTE1 HARDWARE ENGINE automatically when transfer completes
     - Waited by: our code (M engine) before matmul_acc
     - ★ We DO NOT manually set these — hardware handles it!

  FIX→M (FIX signals M):    "L0C buffer is free for next accumulator"
     - Set by: our code (after store, because FIXPIPE consumed L0C data
       synchronously even though the GM write is async)
     - Waited by: our code before the next store

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
N_K_CHUNKS = K // L0_K
N_PAIRS = N_K_CHUNKS // 2
N_TILES_N = N // TILE_N
ROWS_X = M
ROWS_W = GROUP * K

assert M % TILE_M == 0 and N % TILE_N == 0 and K % L0_K == 0
assert N_K_CHUNKS % 2 == 0
for _d in (M, K, N, TILE_M, TILE_N, L0_K):
    assert _d % 16 == 0
assert TILE_M * TILE_N * 4 <= L0C_BYTES
assert 2 * (L0_K * TILE_N) <= L0B_BYTES
assert 127 * 127 * K < 2**31

EV_L0B_0 = 0    # M→MTE1: L0B slot 0 free
EV_L0B_1 = 1    # M→MTE1: L0B slot 1 free
EV_L0C = 2      # FIX→M: L0C buffer free


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
        wbase = (row // GROUP_M) * K

        # --- Init: release L0B slots for MTE1, mark L0C free ---
        # M→MTE1: "slots are free" (M is the source — we CAN set these)
        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=EV_L0B_0)
        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=EV_L0B_1)
        # FIX→M: "L0C is free" (set from our code to re-arm for each wait)
        pl.system.set_flag(pl.PipeType.FIX, pl.PipeType.M, event_id=EV_L0C)

        for n_i in pl.range(N_TILES_N):
            n0 = n_i * TILE_N

            # MTE2: load b_panel (async, overlaps with previous FIXPIPE drain)
            b_panel = pl.load(w, [wbase, n0], [L0_K, TILE_N],
                              target_memory=pl.MemorySpace.Mat)

            # === Head: chunk 0 → L0B slot 0 ===
            # (MTE1 hardware auto-waits for M→MTE1 flag 0 = "slot 0 free")
            b_first = pl.tile.extract(b_panel, 0, 0, [L0_K, TILE_N],
                                     target_memory=pl.MemorySpace.Right)
            # ★ Do NOT set MTE1→M flag — hardware sets it when transfer completes

            # MAD: wait for MTE1→M flag 0 ("slot 0 data ready" — hardware-set)
            pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, event_id=EV_L0B_0)
            a_l0 = pl.tile.extract(
                pl.load(x, [row, 0], [TILE_M, L0_K],
                        target_memory=pl.MemorySpace.Mat),
                0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left)
            acc = pl.tile.matmul(a_l0, b_first)
            # Release slot 0 for MTE1 (M→MTE1: M is source — we CAN set this)
            pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=EV_L0B_0)

            # === Pair-unrolled steady-state ===
            for kp in pl.range(N_PAIRS):
                k_even = (2 + kp * 2) * L0_K
                k_odd = (3 + kp * 2) * L0_K

                # MTE1: extract even chunk → slot 0
                # (MTE1 hardware waits for M→MTE1 flag 0 = "slot 0 free")
                b_even = pl.tile.extract(b_panel, k_even, 0, [L0_K, TILE_N],
                                         target_memory=pl.MemorySpace.Right)
                # ★ Hardware auto-sets MTE1→M flag 0 when done

                # MTE1: extract odd chunk → slot 1
                # (MTE1 hardware waits for M→MTE1 flag 1 = "slot 1 free")
                b_odd = pl.tile.extract(b_panel, k_odd, 0, [L0_K, TILE_N],
                                        target_memory=pl.MemorySpace.Right)
                # ★ Hardware auto-sets MTE1→M flag 1 when done

                # MAD on even chunk (slot 0)
                # Wait for MTE1→M flag 0 (hardware-set "data ready")
                pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, event_id=EV_L0B_0)
                acc = pl.tile.matmul_acc(acc, a_l0, b_even)
                # Release slot 0 (M→MTE1: we CAN set this)
                pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=EV_L0B_0)

                # MAD on odd chunk (slot 1)
                pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, event_id=EV_L0B_1)
                acc = pl.tile.matmul_acc(acc, a_l0, b_odd)
                pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=EV_L0B_1)

            # === FIXPIPE: drain L0C ===
            # Wait for L0C free (from init or previous n-block's set below)
            pl.system.wait_flag(pl.PipeType.FIX, pl.PipeType.M, event_id=EV_L0C)
            out = pl.store(acc, [row, n0], out)
            # FIXPIPE consumed L0C data (sync) — L0C is free again.
            # Re-arm FIX→M flag for the next n-block's wait.
            pl.system.set_flag(pl.PipeType.FIX, pl.PipeType.M, event_id=EV_L0C)

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

    print(f"compiling: {N_STRIPS} strips, manual flags v3 "
          f"(hardware-set MTE1→M, manual M→MTE1 + FIX→M) ...")
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
    print(f"OK  bit-exact  tasks = {N_STRIPS}  manual flags v3  "
          f"total = {M * K * N * 2 / 1e9:.1f} GFLOP  (AIC x{CUBE_CORES})")


if __name__ == "__main__":
    main()
