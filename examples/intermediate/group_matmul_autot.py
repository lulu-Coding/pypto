# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Group-level SPMD INT8 group matmul: one block per group, K-pipelined.

Corrected from the user's transcription. Structure preserved:
    pl.spmd(core_num=G) -> one block per group
    get_block_idx()     -> group index
    unroll over N-tiles, pipeline over K-chunks

Shape: x [8192, 3584] int8, weight [G*K, N] int8, out [8192, N] int32.
Total work = 8192 * 3584 * 3072 = 90.3 GMAC (NOT 2.89 TMAC -- the earlier
sessions used M=8192 PER GROUP giving x=[262144, K]; this file's M=8192 is
the TOTAL across all 32 groups, 256 rows each).
"""

import argparse
import time

import pypto.language as pl
import torch
from pypto.runtime import RunConfig

# ============ shape config ============
G = 32
GROUP_M = 256
K = 3584
N = 3072
M = GROUP_M * G                    # 8192 total rows

TILE_K = 256
TILE_N = 256
# ======================================

N_TILES_N = N // TILE_N            # 12
K_CHUNKS = K // TILE_K             # 14


@pl.jit.incore
def _matmul_group(x: pl.Tensor, weight: pl.Tensor, y: pl.Tensor):
    """One SPMD block = one group's full [GROUP_M, N] output."""
    i = pl.tile.get_block_idx()
    x_base = i * GROUP_M
    w_base = i * K
    for j in pl.range(N_TILES_N):
        n0 = j * TILE_N
        # Head-peel kb==0: defines the accumulator (if/else peel is illegal
        # in InCore -- Acc phi; create_tensor is also forbidden in InCore).
        a = x[x_base : x_base + GROUP_M, 0:TILE_K]
        b = weight[w_base : w_base + TILE_K, n0 : n0 + TILE_N]
        acc = pl.matmul(a, b, out_dtype=pl.INT32)
        for kb in pl.pipeline(1, K_CHUNKS, stage=2):
            k0 = kb * TILE_K
            a = x[x_base : x_base + GROUP_M, k0 : k0 + TILE_K]
            b = weight[w_base + k0 : w_base + k0 + TILE_K, n0 : n0 + TILE_N]
            acc = pl.matmul_acc(acc, a, b)
        y = pl.assemble(y, acc, [x_base, n0])
    return y


@pl.jit
def matmul_group_autot(x: pl.Tensor, weight: pl.Tensor, y: pl.InOut[pl.Tensor]):
    with pl.spmd(core_num=G):
        y = _matmul_group(x, weight, y)
    return y


def grouped_matmul_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """weight is [G, K, N] 3D for the reference."""
    y = torch.zeros((M, N), dtype=torch.int32)
    for g in range(G):
        y[g * GROUP_M : (g + 1) * GROUP_M] = (
            x[g * GROUP_M : (g + 1) * GROUP_M, :].int() @ weight[g, :, :].int()
        )
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description="Group Matmul (group-level SPMD)")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim", "a2a3sim", "a2a3"])
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    cfg = RunConfig(platform=args.platform, device_id=args.device)
    torch.manual_seed(0)
    x = torch.randint(-127, 128, (M, K), dtype=torch.int8)
    weight = torch.randint(-127, 128, (G, K, N), dtype=torch.int8)
    weight_2d = weight.reshape(G * K, N)
    y = torch.zeros((M, N), dtype=torch.int32)

    print(f"compiling: {G} spmd blocks (one per group), "
          f"tile {GROUP_M}x{TILE_N}x{TILE_K} ...")
    t0 = time.perf_counter()
    matmul_group_autot(x, weight_2d, y, config=cfg)
    print(f"done in {time.perf_counter() - t0:.4f}s")

    print("verifying ...")
    expected = grouped_matmul_ref(x, weight)
    assert torch.equal(expected, y), (
        f"max err = {(expected - y).abs().max().item()}"
    )
    print(f"OK  bit-exact  {G} blocks x [{GROUP_M}, {N}]  "
          f"total = {M * K * N * 2 / 1e9:.1f} GFLOP")


if __name__ == "__main__":
    main()
