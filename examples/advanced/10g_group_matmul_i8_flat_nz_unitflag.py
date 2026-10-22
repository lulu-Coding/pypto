# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit-flag phased flat-L0 pipeline: flat NZ INT8 group matmul.

10b/10c showed that cube and MTE2 already do AscendC's work (cube 208.9us,
MTE2 ~206us), but the generated kernel still contains a software
``wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0)`` before each L0C store and a
``set_flag(PIPE_FIX, PIPE_M, EVENT_ID0)`` after it. Those pipe-wide events force
the cube and fixpipe to serialize. AscendC instead marks the MMA instructions
with the hardware unit flag:

    mmadParams.unitFlag = isLastK ? 3 : 2;   // middle K / last K
    fixParams.unitFlag  = 3;                 // L0C store

PTOAS and pto-isa support the same mechanism through ``acc_phase`` and
``st_phase``: a ``TMATMUL_ACC<AccPhase::Final>`` performs the producer-side
check-and-set, and a ``TSTORE<STPhase::Final>`` performs the consumer-side
check-and-clear. PyPTO's PTO emitter already translates a matmul accumulation's
``acc_phase`` kwarg into ``accPhase``; this example uses that path after adding
it to ``tile.matmul_acc``:

    non-last L0 K chunk: pl.AccPhase.Partial   -> TMATMUL_ACC<AccPhase::Partial>
    last L0 K chunk:     pl.AccPhase.Final     -> TMATMUL_ACC<AccPhase::Final>
    L0C store:           st_phase=Final        -> TSTORE<..., STPhase::Final>

The final accumulation must be a straight-line producer that the store consumes;
the phase verifier rejects a Final producer whose obligation crosses a loop
boundary. Therefore the last K chunk is peeled out of the flat pipeline: the
loop covers chunks 1..N_CHUNKS-2 with Partial, and chunk N_CHUNKS-1 is the Final
producer followed by the Final store.

The seed chunk stays a base ``tile.matmul`` (no phase). AscendC also sets
unitFlag=2 on its first MMAD; if the hardware/PTOAS path needs that seed on the
unit-flag path, extend ``tile.matmul`` with acc_phase in the same way. This
example isolates the last-K + store pairing first.

Reference points on Ascend950DT_9582 (32 AIC, 8192x3584x3072 INT8):

    10b manual flat-L0, no unit flag       315us total / 572.7 TFLOPS
    10c 2-D AutoTile flat NZ, no unit flag AICore 259-263.6us
    AscendC GMM (msprof kernel time)       215-220us

Weight layout caveat: this prototype uses the flat ``[G*K, N]`` NZ form, not the
AscendC per-group ``[G, N/32, K/16, 16, 32]`` form. The layout is addressed after
the unit-flag schedule is validated.

This example requires a PyPTO build that includes acc_phase on
``tile.matmul_acc``. Run on A5 (profile first, then benchmark):

    python examples/advanced/10g_group_matmul_i8_flat_nz_unitflag.py -p a5 -d 0 --profile
    python examples/utils/swimlane_summary.py <printed record path>

    python examples/advanced/10g_group_matmul_i8_flat_nz_unitflag.py -p a5 -d 0 --rounds 20

In the generated kernel, look for ``TMATMUL_ACC<AccPhase::Partial/Final>`` and
``TSTORE<..., STPhase::Final>``, and for the absence of the M->FIX
``wait_flag``/``set_flag`` pair around the store.
"""

import argparse
import time
from pathlib import Path

import pypto.language as pl
import torch

from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime import RunConfig, benchmark

# ---------------------------------------------------------------------------
# Shape and AscendC arch35 base block
# ---------------------------------------------------------------------------

GROUP = 32
GROUP_M = 256
K, N = 3584, 3072
M = GROUP_M * GROUP
ROWS_W = GROUP * K

TILE_M, TILE_N = 256, 256
L0_K = 128
FLAT_STAGE = 2
N_TILES = N // TILE_N
N_CHUNKS = K // L0_K
NZ_C0 = 32

L0C_BYTES = 256 * 1024
L0A_BYTES = 64 * 1024
L0B_BYTES = 64 * 1024
MAT_BYTES = 512 * 1024

assert GROUP_M == TILE_M and M % TILE_M == 0
assert K % L0_K == 0 and N % TILE_N == 0
assert TILE_M * TILE_N * 4 <= L0C_BYTES, "accumulator exceeds A5 L0C"
assert FLAT_STAGE * (TILE_M * L0_K) <= L0A_BYTES, "flat-pipeline L0A budget exceeds L0A"
assert FLAT_STAGE * (L0_K * TILE_N) <= L0B_BYTES, "flat-pipeline L0B budget exceeds L0B"
assert FLAT_STAGE * (TILE_M * L0_K + L0_K * TILE_N) <= MAT_BYTES, "flat-pipeline Mat budget exceeds L1"
assert ROWS_W % 16 == 0 and N % NZ_C0 == 0, "flat NZ fractal alignment"
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"
assert N_CHUNKS >= 2, "need a seed and at least one peeled final chunk"


# ---------------------------------------------------------------------------
# Kernel: one resident SPMD block per group, phased flat L0 pipeline
# ---------------------------------------------------------------------------


@pl.jit.incore
def _gemm_strip(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[ROWS_W, N], pl.INT8, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One [TILE_M, N] strip: seed + partial pipeline + peeled final chunk."""
    # Keep both offsets as plain products of the block index. The NZ offset
    # prover tracks non-negativity through get_block_idx and products, but not
    # through the equivalent ``(row // GROUP_M) * K`` floor-division chain.
    block = pl.tile.get_block_idx()
    row = block * TILE_M
    wbase = block * K

    for n_i in pl.range(N_TILES):
        n0 = n_i * TILE_N

        # Seed: the first L0 chunk defines the accumulator. The base matmul
        # stays on the legacy path; the accumulating chunks below carry phases.
        a_seed = pl.load(x, [row, 0], [TILE_M, L0_K], target_memory=pl.MemorySpace.Mat)
        b_seed = pl.load(w, [wbase, n0], [L0_K, TILE_N], target_memory=pl.MemorySpace.Mat)
        acc = pl.tile.matmul(
            pl.tile.extract(a_seed, 0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
            pl.tile.extract(b_seed, 0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
        )

        # Chunks 1..N_CHUNKS-2 are interior accumulations: Partial keeps the
        # producer unit flag not-ready (AscendC unitFlag=2).
        for kc in pl.pipeline(1, N_CHUNKS - 1, stage=FLAT_STAGE):
            k0 = kc * L0_K
            a_p = pl.load(x, [row, k0], [TILE_M, L0_K], target_memory=pl.MemorySpace.Mat)
            b_p = pl.load(w, [wbase + k0, n0], [L0_K, TILE_N], target_memory=pl.MemorySpace.Mat)
            acc = pl.tile.matmul_acc(
                acc,
                pl.tile.extract(a_p, 0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
                pl.tile.extract(b_p, 0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
                acc_phase=pl.AccPhase.Partial,
            )

        # Peel the last chunk: Final is the producer-side check-and-set. Keeping
        # it outside the pipeline also satisfies the straight-line-region rule
        # that pairs this SSA value with the Final store below.
        k_tail = (N_CHUNKS - 1) * L0_K
        a_tail = pl.load(x, [row, k_tail], [TILE_M, L0_K], target_memory=pl.MemorySpace.Mat)
        b_tail = pl.load(w, [wbase + k_tail, n0], [L0_K, TILE_N], target_memory=pl.MemorySpace.Mat)
        acc = pl.tile.matmul_acc(
            acc,
            pl.tile.extract(a_tail, 0, 0, [TILE_M, L0_K], target_memory=pl.MemorySpace.Left),
            pl.tile.extract(b_tail, 0, 0, [L0_K, TILE_N], target_memory=pl.MemorySpace.Right),
            acc_phase=pl.AccPhase.Final,
        )

        # Consumer-side check-and-clear: the fixpipe waits on the Final flag and
        # releases it, instead of a pipe-wide M->FIX wait/set pair.
        out = pl.store(acc, [row, n0], out, st_phase=pl.STPhase.Final)

    return out


@pl.jit
def group_matmul_i8_flat_nz_unitflag(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[ROWS_W, N], pl.INT8, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One SPMD launch: 32 blocks, one group strip per block."""
    with pl.spmd(GROUP):
        out = _gemm_strip(x, w, out)
    return out


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def to_nz_flat_int8(w: torch.Tensor) -> torch.Tensor:
    """Reorder a logical ``[G*K, N]`` INT8 weight into flat NZ bytes.

    The logical shape is preserved; the contiguous bytes follow
    ``[N/32, R/16, 16, 32]`` for the whole flattened matrix. This is NOT the
    per-group NZ layout used by AscendC; it is the layout PyPTO supports today
    without a 3-D view.

    Args:
        w: Logical row-major INT8 weight of shape ``[G*K, N]``.

    Returns:
        A tensor with the same shape as ``w``, holding flat NZ bytes.

    Raises:
        AssertionError: If rows are not 16-aligned or columns are not 32-aligned.
    """
    rows, cols = w.shape
    assert rows % 16 == 0, f"NZ needs 16-aligned rows, got {rows}"
    assert cols % NZ_C0 == 0, f"NZ needs {NZ_C0}-aligned columns, got {cols}"
    return (
        w.reshape(rows // 16, 16, cols // NZ_C0, NZ_C0)
        .permute(2, 0, 1, 3)
        .contiguous()
        .reshape(rows, cols)
    )


def grouped_matmul_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Exact INT32 reference: per-group ``x_g @ w_g`` in FP64, narrowed once."""
    out = torch.empty((M, N), dtype=torch.int32)
    for g in range(GROUP):
        rows = slice(g * GROUP_M, (g + 1) * GROUP_M)
        out[rows] = (x[rows].double() @ w[g].double()).to(torch.int32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Unit-flag phased flat-L0 pipeline, flat NZ grouped matmul")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim", "a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--planner",
        choices=["pypto", "dsa_rp", "ptoas"],
        default="pypto",
        help="on-chip memory planner; the manual L0 tiles are 32 KB so all three fit",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="single DFX run (swimlane level 4 + PMU) instead of benchmark; use with swimlane_summary.py",
    )
    parser.add_argument("--artifacts-dir", default="build_output/gmm_flat_nz_unitflag_profile")
    args = parser.parse_args()

    if args.platform == "a5sim":
        print("warning: 256x256 accumulator fills L0C; older a5sim executors may crash")

    planners = {
        "pypto": MemoryPlanner.PYPTO,
        "dsa_rp": MemoryPlanner.DSA_RP,
        "ptoas": MemoryPlanner.PTOAS,
    }
    planner = planners[args.planner]

    if args.profile:
        cfg = RunConfig(
            platform=args.platform,
            device_id=args.device,
            memory_planner=planner,
            enable_chip_swimlane=4,
            enable_pmu=2,
            enable_dep_gen=True,
            save_kernels=True,
            save_kernels_dir=args.artifacts_dir,
        )
    else:
        cfg = RunConfig(platform=args.platform, device_id=args.device, memory_planner=planner)

    torch.manual_seed(0)
    print("generating inputs ...")
    x = torch.randint(-127, 128, (M, K), dtype=torch.int8)
    w = torch.randint(-127, 128, (GROUP, K, N), dtype=torch.int8)
    w_flat = w.reshape(ROWS_W, N)
    w_nz = to_nz_flat_int8(w_flat)
    out = torch.zeros((M, N), dtype=torch.int32)

    print(
        f"compiling: {GROUP} SPMD blocks, unit-flag phased flat L0 pipeline stage={FLAT_STAGE}, "
        f"block {TILE_M}x{TILE_N}x{L0_K}, flat NZ weight, planner={args.planner} ..."
    )
    t0 = time.perf_counter()
    compiled = group_matmul_i8_flat_nz_unitflag.compile(x, w_nz, out, config=cfg)
    print(f"compiled in {time.perf_counter() - t0:.1f}s; artifacts: {compiled.output_dir}")

    compiled(x, w_nz, out, config=cfg)

    print("verifying bit-exact against FP64 reference ...")
    expected = grouped_matmul_ref(x, w)
    if not torch.equal(out, expected):
        diff = (out - expected).abs()
        raise SystemExit(f"FAIL: max abs diff {diff.max().item()}, mismatches {int((diff != 0).sum())}")
    print("OK  bit-exact")

    if args.profile:
        roots = [Path(args.artifacts_dir)]
        output_dir = getattr(compiled, "output_dir", None)
        if output_dir:
            roots.append(Path(output_dir))
        records = []
        for root in roots:
            if root.exists():
                records.extend(root.rglob("chip_swimlane_records.json"))
        print("profile artifacts:")
        for record in sorted(set(records)):
            print(f"  {record}")
        if records:
            print("\nsummarize the AICore span with:")
            print(f"  python examples/utils/swimlane_summary.py {sorted(set(records))[0]}")
        else:
            print(f"no chip_swimlane_records.json under {[str(root) for root in roots]}")
        print("\ncheck the generated unit-flag path with:")
        print(
            "  grep -R -E 'TMATMUL_ACC<AccPhase|STPhase::Final|wait_flag\\(PIPE_M, PIPE_FIX\\)' "
            f"{args.artifacts_dir} | head -20"
        )
        return

    print(f"benchmark: {args.warmup} warmup + {args.rounds} measured launches ...")
    stats = benchmark(compiled, [x, w_nz, out], rounds=args.rounds, warmup=args.warmup, config=cfg)

    flops = 2.0 * M * K * N
    median_us = stats.device_us_median
    tflops = flops / (median_us * 1e-6) / 1e12 if median_us > 0 else 0.0
    print(
        f"device_us: median={median_us:.2f} min={stats.device_us_min:.2f} "
        f"mean={stats.device_us_mean:.2f} stdev={stats.device_us_stdev:.2f} max={stats.device_us_max:.2f}"
    )
    print(f"device TFLOPS at median: {tflops:.1f}  ({flops / 1e9:.1f} GFLOP total)")
    print(f"raw device_us samples: {[round(v, 2) for v in stats.samples]}")
    print(
        "reference: 10b manual (no unit flag) 315us; "
        "10c AutoTile flat NZ AICore 259-263.6us; AscendC kernel 215-220us"
    )


if __name__ == "__main__":
    main()
