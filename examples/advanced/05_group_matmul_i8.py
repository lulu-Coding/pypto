# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""High-parallelism INT8 group matmul on A5: INT8 x INT8 -> INT32 acc -> dequant -> FP16.

Band-stacked layout (group g owns a contiguous band in each tensor):

    x    : [GROUP*M, K]   INT8   group g's activations at rows [g*M, (g+1)*M)
    w    : [GROUP*K, N]   INT8   group g's weights      at rows [g*K, (g+1)*K)
    scale: [GROUP*M, 1]   FP32   per-row scale (fill a band with one value for per-group scale)
    out  : [GROUP*M, N]   FP16

Parallel structure (one output tile per task, GROUP x M_TILES x N_TILES tasks):

    g  -> pl.unroll    compile-time expansion; band offsets fold to literals
    m0 -> pl.parallel  runtime row-tile dispatch
    n0 -> pl.unroll    compile-time column-tile expansion
    kb -> pl.pipeline  stage=2 K loop (operand double buffering)

Dequant epilogue — PyPTO 3.0 has no scaled fixpipe dequant (pto.tdequant lacks a
frontend path, and the unscaled fixpipe writeback converts only FP32 -> FP16/BF16,
so an INT32 accumulator may only leave L0C as INT32). Two supported modes instead:

    mixed  ONE task per tile: the INT32 acc crosses the C2V FIFO to the vector
           lane of the same core group; cast -> row-scale -> FP16 never leaves
           the chip (cf. tests/st/runtime/cross_core/test_c2v_narrowed_acc_epilogue.py,
           _MixedEpilogueCase).
    staged TWO tasks per tile: the cube scope stores INT32 acc into a GM
           intermediate; a separate vector kernel dequantizes it. Simplest
           structure, and deq(i) can overlap cube(i+1) via SSA dependencies
           (_StagedKSplitCase in the same test).

A5 (Ascend950) resources: L0C 256KB, L0A/L0B 64KB, Mat 512KB, 18 cube cores/die.

Run:
    python examples/advanced/05_group_matmul_i8.py                # a5, device 0, mixed
    python examples/advanced/05_group_matmul_i8.py -p a5sim      # simulator, no device
    python examples/advanced/05_group_matmul_i8.py -e staged     # two-kernel epilogue
"""

import argparse

import pypto.language as pl
import torch
from pypto.runtime import RunConfig

# ============ shape config (edit freely) ============
GROUP = 9                            # groups (9*2*2 = 36 = 2x18 cores, full card)
M, K, N = 256, 512, 256              # per-group matrix shape
TILE_M, TILE_K, TILE_N = 128, 128, 128   # output tile / K chunk
OUT_DTYPE = pl.FP16
EPILOGUE = "mixed"                   # default epilogue mode ("mixed" | "staged")
# ===================================================

L0C_BYTES = 256 * 1024               # A5 Acc (L0C)
CUBE_CORES = 18                      # A5 cube cores per die

# Config sanity (fails fast at import with a clear message).
assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0, "tiles must divide their dims"
assert K // TILE_K >= 2, "K // TILE_K must be >= 2 (single-block pipelines are degenerate)"
for _d in (M, K, N, TILE_M, TILE_K, TILE_N):
    assert _d % 16 == 0, f"{_d} is not a multiple of 16 (cube fractal alignment)"
_acc_bytes = TILE_M * TILE_N * 4     # INT32 accumulator footprint
assert _acc_bytes <= L0C_BYTES, "acc tile exceeds A5 L0C (256KB)"
assert _acc_bytes * 2 <= L0C_BYTES, "acc tile > L0C/2: dbC ping-pong will not fit (still runs)"


@pl.jit
def group_matmul_mixed(x: pl.Tensor, w: pl.Tensor, scale: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Mixed epilogue: cube GEMM + vector dequant in ONE scope, acc crosses C2V."""
    for g in pl.unroll(GROUP):
        for m0 in pl.parallel(0, M, TILE_M):
            for n0 in pl.unroll(0, N, TILE_N):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="gmm"):
                    acc = pl.create_tensor([TILE_M, TILE_N], dtype=pl.INT32)
                    for kb in pl.pipeline(0, K // TILE_K, stage=2):
                        k0 = kb * TILE_K
                        a = x[g * M + m0 : g * M + m0 + TILE_M, k0 : k0 + TILE_K]
                        b = w[g * K + k0 : g * K + k0 + TILE_K, n0 : n0 + TILE_N]
                        if k0 == 0:
                            acc = pl.matmul(a, b, out_dtype=pl.INT32)
                        else:
                            acc = pl.matmul_acc(acc, a, b)
                    s = scale[g * M + m0 : g * M + m0 + TILE_M, :]
                    deq = pl.row_expand_mul(pl.cast(acc, target_type=pl.FP32, mode="none"), s)
                    out = pl.assemble(out, pl.cast(deq, target_type=OUT_DTYPE), [g * M + m0, n0])
    return out


@pl.jit
def group_matmul_staged(x: pl.Tensor, w: pl.Tensor, scale: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Staged epilogue: cube stores INT32 acc to GM; a separate vector kernel dequants.

    Dependencies are inserted per SSA value: deq(g, m0, n0) waits only on
    cube(g, m0, n0), so the scheduler may overlap deq(i) with cube(i+1).
    """
    acc_gm = pl.create_tensor([GROUP * M, N], dtype=pl.INT32)   # control-plane GM scratch
    for g in pl.unroll(GROUP):
        for m0 in pl.parallel(0, M, TILE_M):
            for n0 in pl.unroll(0, N, TILE_N):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="gmm_cube"):
                    acc = pl.create_tensor([TILE_M, TILE_N], dtype=pl.INT32)
                    for kb in pl.pipeline(0, K // TILE_K, stage=2):
                        k0 = kb * TILE_K
                        a = x[g * M + m0 : g * M + m0 + TILE_M, k0 : k0 + TILE_K]
                        b = w[g * K + k0 : g * K + k0 + TILE_K, n0 : n0 + TILE_N]
                        if k0 == 0:
                            acc = pl.matmul(a, b, out_dtype=pl.INT32)
                        else:
                            acc = pl.matmul_acc(acc, a, b)
                    acc_gm = pl.assemble(acc_gm, acc, [g * M + m0, n0])
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="gmm_deq"):
                    t = acc_gm[g * M + m0 : g * M + m0 + TILE_M, n0 : n0 + TILE_N]
                    s = scale[g * M + m0 : g * M + m0 + TILE_M, :]
                    deq = pl.row_expand_mul(pl.cast(t, target_type=pl.FP32, mode="none"), s)
                    out = pl.assemble(out, pl.cast(deq, target_type=OUT_DTYPE), [g * M + m0, n0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", default="a5", help="target platform (default: a5)")
    parser.add_argument("-d", "--device", type=int, default=0, help="device id (default: 0)")
    parser.add_argument(
        "-e", "--epilogue", choices=["mixed", "staged"], default=EPILOGUE,
        help="dequant epilogue mode (default: %(default)s)",
    )
    args = parser.parse_args()

    cfg = RunConfig(platform=args.platform, device_id=args.device)
    torch.manual_seed(0)
    x = torch.randint(-127, 128, (GROUP * M, K), dtype=torch.int8)
    w = torch.randint(-127, 128, (GROUP * K, N), dtype=torch.int8)
    # FP16 tops out at 65504: |acc|max = 127*127*K (~8.3e6 at K=512), so scale=0.005
    # keeps the worst case (~41K) in range. Recompute when raising K.
    scale = torch.full((GROUP * M, 1), 0.005, dtype=torch.float32)
    out = torch.zeros(GROUP * M, N, dtype=torch.float16)

    fn = group_matmul_mixed if args.epilogue == "mixed" else group_matmul_staged
    fn(x, w, scale, out, config=cfg)

    ref = torch.zeros(GROUP * M, N, dtype=torch.float32)
    for g in range(GROUP):
        ref[g * M : (g + 1) * M] = x[g * M : (g + 1) * M].float() @ w[g * K : (g + 1) * K].float()
    ref = (ref * scale).half()
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2), (
        f"max err = {(out.float() - ref.float()).abs().max().item()}"
    )

    n_tiles = GROUP * (M // TILE_M) * (N // TILE_N)
    n_tasks = n_tiles if args.epilogue == "mixed" else n_tiles * 2
    print(f"OK  [{args.epilogue}] tasks = {n_tasks}  platform = {args.platform}  device = {args.device}")


if __name__ == "__main__":
    main()
