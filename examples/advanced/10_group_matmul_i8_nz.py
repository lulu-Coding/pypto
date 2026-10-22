# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Per-group NZ INT8 grouped matmul on Ascend950DT (A5, 32 AIC).

Benchmark shape:

    x  : [8192, 3584]    INT8   M = 32 groups x 256 rows
    w  : [32, 3584, 3072] INT8  logical [G, K, N], each group stored in its own
                                PTO NZ fractal block
    out: [8192, 3072]    INT32  raw accumulator, no dequant

Weight layout. This version matches the AscendC reference: the NZ blocks are
built per group, i.e. the physical order is
``[G, N/32, K/16, 16, 32]`` for INT8. The DSL keeps the logical ``[G, K, N]``
shape and annotates the tensor as ``pl.NZ``; ``to_nz_per_group`` produces the
bytes.

Kernel layout. One SPMD block per group (32 groups = 32 AIC on this device):

    - K is split into 256-deep panels; the head panel seeds the accumulator and
      the rest run under ``pl.pipeline(stage=2)``. This is the fastest measured
      configuration (312.83us). AscendC's A8W8 tiling also resolves to
      base 256x256x128, stepK=2, depth=4, which this panel/pipeline reproduces.
    - At L1_K=128 all three planners fit (the L0 A tile drops to 256x128), but
      PYPTO/DSA_RP/PTOAS all measured ~315-317us: the smaller panel raises MTE2
      while the planner overlap does not recover it. L1_K=256 stays the default.
    - N is split into twelve 256-wide blocks; each block is one [1, 256, 256]
      INT32 accumulator, which is the AscendC arch35 base block.
    - L0 tiling inside each panel is left to AutoTileMatmulL0. This is required
      by the current NZ milestone: a 2-D rank-reduced view of a 3-D NZ tensor is
      not a supported load source, so the supported grouped path is the
      tensor-level ``2-D @ 3-D`` matmul, which AutoTile then lowers to L0.

Run on A5:

    # device latency + bit-exact check (default PYPTO planner)
    python examples/advanced/10_group_matmul_i8_nz.py -p a5 -d 0 --rounds 20

    # pipe-utilization PMU + swimlane profiling (skips the benchmark)
    SIMPLER_DFX=1 python examples/advanced/10_group_matmul_i8_nz.py -p a5 -d 0 --profile --planner pypto

    # Planner sweep: set L1_K=128 above first. All three then fit, but measured
    # PYPTO ~317.5us, DSA_RP ~316.1us, PTOAS ~315.1us -- none beats L1_K=256.

    # recorded: PYPTO L1_K=512 -> 362.69us / 497.4 TFLOPS
    #           PYPTO L1_K=256 -> 312.83us / 576.6 TFLOPS  (best)
    #           L1_K=128, all planners -> 315.1-317.5us

The [256, 256] INT32 output block fills L0C exactly, so the legacy PYPTO planner
runs it with a single accumulator. The DSA_RP / PTOAS planners let AutoTile
choose dbC=2 and overlap the L0C drain with the next block, which is the main
thing the default-planner profile shows as serialized.
"""

import argparse
import csv
import time
from pathlib import Path

import pypto.language as pl
import torch

from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime import RunConfig, benchmark

# ---------------------------------------------------------------------------
# Shape and arch35 design point
# ---------------------------------------------------------------------------

GROUP = 32
GROUP_M = 256
K, N = 3584, 3072
M = GROUP_M * GROUP

N_TILE = 256                             # AscendC baseN
L1_K = 256                               # best measured; AscendC stepK=2, depth=4
N_TILES = N // N_TILE                    # 12 N blocks per group
N_PANELS = K // L1_K                     # 14 K panels

NZ_C0 = 32                               # 256 bits / INT8 = 32 elements

assert GROUP_M == 256
assert K % L1_K == 0 and N % N_TILE == 0
assert K % 16 == 0 and N % NZ_C0 == 0, "NZ fractal alignment"
assert 127 * 127 * K < 2**31, "INT32 accumulator overflow at this K"


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@pl.jit
def group_matmul_i8_nz(
    x: pl.Tensor[[M, K], pl.INT8],
    w: pl.Tensor[[GROUP, K, N], pl.INT8, pl.NZ],
    out: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    """One block per group; per-group NZ weight; K-panel pipeline.

    The 3-D weight slice keeps its leading unit dim so the NZ tensor view stays
    rank-3, which is the shape the current grouped-matmul NZ lowering supports.
    The matmul broadcasts the 2-D activation against that unit batch dim, and
    AutoTileMatmulL0 selects the L1->L0 split inside each 512-deep K panel.
    """
    for g in pl.spmd(GROUP):
        row = g * GROUP_M

        for n_i in pl.range(N_TILES):
            n0 = n_i * N_TILE

            x_head = x[row : row + GROUP_M, 0:L1_K]
            w_head = w[g : g + 1, 0:L1_K, n0 : n0 + N_TILE]
            acc = pl.matmul(x_head, w_head, out_dtype=pl.INT32)

            for kb in pl.pipeline(1, N_PANELS, stage=2):
                k1 = kb * L1_K
                x_k = x[row : row + GROUP_M, k1 : k1 + L1_K]
                w_k = w[g : g + 1, k1 : k1 + L1_K, n0 : n0 + N_TILE]
                acc = pl.matmul_acc(acc, x_k, w_k)

            out[row : row + GROUP_M, n0 : n0 + N_TILE] = pl.reshape(acc, [GROUP_M, N_TILE])

    return out


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def to_nz_per_group(w: torch.Tensor) -> torch.Tensor:
    """Reorder a logical ``[G, K, N]`` INT8 weight into per-group NZ bytes.

    The logical shape is preserved; only the byte order changes. For each group
    the blocked order is ``[N/32, K/16, 16, 32]``.

    Args:
        w: Logical row-major INT8 weight of shape ``[G, K, N]``.

    Returns:
        A tensor with the same shape as ``w``, holding per-group NZ bytes.

    Raises:
        AssertionError: If K is not 16-aligned or N is not 32-aligned.
    """
    groups, rows, cols = w.shape
    assert rows % 16 == 0, f"NZ needs 16-aligned rows, got {rows}"
    assert cols % NZ_C0 == 0, f"NZ needs {NZ_C0}-aligned columns, got {cols}"
    return (
        w.reshape(groups, rows // 16, 16, cols // NZ_C0, NZ_C0)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .reshape(groups, rows, cols)
    )


def grouped_matmul_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Exact INT32 reference: per-group ``x_g @ w_g`` in FP64, narrowed once."""
    out = torch.empty((M, N), dtype=torch.int32)
    for g in range(GROUP):
        rows = slice(g * GROUP_M, (g + 1) * GROUP_M)
        acc = x[rows].double() @ w[g].double()
        out[rows] = acc.to(torch.int32)
    return out


def _as_float(value: str | None) -> float | None:
    """Parse a PMU field as a number, accepting decimal or 0x-prefixed values."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            return float(int(value, 0))
        except ValueError:
            return None


def _print_pmu_summary(path: Path) -> None:
    """Print per-column mean/max for a PMU CSV so it can be pasted back."""
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except OSError as exc:
        print(f"  could not read {path}: {exc}")
        return

    fields = reader.fieldnames or []
    print(f"  rows={len(rows)} columns={fields}")
    for field in fields:
        values = [v for v in (_as_float(row.get(field, "")) for row in rows) if v is not None]
        if values:
            mean = sum(values) / len(values)
            print(f"    {field}: mean={mean:.3f} max={max(values):.3f}")


def _print_profile_artifacts(compiled: object, artifacts_dir: str) -> None:
    """Locate and print the small DFX artefacts produced by a profile run."""
    roots = [Path(artifacts_dir)]
    output_dir = getattr(compiled, "output_dir", None)
    if output_dir:
        roots.append(Path(output_dir))
    seen: set[Path] = set()

    for root in roots:
        if not root.exists():
            continue
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)

        files = [path for path in sorted(root.rglob("*")) if path.is_file()]
        if not files:
            continue

        interesting = [
            path
            for path in files
            if path.name in {"pmu.csv", "perf_hints.log", "chip_swimlane_records.json", "deps.json"}
            or path.name.startswith("merged_swimlane_")
        ]
        print(f"\nprofile artifacts under {root} ({len(files)} files):")
        if not interesting:
            print("  no pmu.csv / perf_hints.log / swimlane files found")
        for path in interesting:
            print(f"  {path.relative_to(root)}  ({path.stat().st_size} bytes)")
            if path.name == "pmu.csv":
                _print_pmu_summary(path)
            elif path.name == "perf_hints.log":
                text = path.read_text(errors="replace").splitlines()
                print("  perf_hints.log:")
                for line in text[:120]:
                    print(f"    {line}")
                if len(text) > 120:
                    print(f"    ... {len(text) - 120} more lines")
            elif path.name.startswith("merged_swimlane_"):
                print("  (merged swimlane trace; keep this path for a converter run)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-group NZ INT8 grouped matmul (A5)")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim", "a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="collect PMU (pipe utilization) + swimlane and skip the benchmark",
    )
    parser.add_argument("--artifacts-dir", default="build_output/gmm_nz_profile")
    parser.add_argument(
        "--planner",
        choices=["pypto", "dsa_rp", "ptoas"],
        default="pypto",
        help="on-chip memory planner; at L1_K=256 only pypto fits, dsa_rp/ptoas need the L1_K=128 variant",
    )
    args = parser.parse_args()

    if args.platform == "a5sim":
        print("warning: 256x256 INT32 output blocks fill L0C; older a5sim executors crash on this tiling")
    if args.planner != "pypto" and L1_K > 128:
        print("note: dsa_rp/ptoas need L1_K=128 in this kernel; expect the L0A placement failure")

    planners = {
        "pypto": MemoryPlanner.PYPTO,
        "dsa_rp": MemoryPlanner.DSA_RP,
        "ptoas": MemoryPlanner.PTOAS,
    }
    planner = planners[args.planner]
    artifacts_dir = f"{args.artifacts_dir}_{args.planner}"

    if args.profile:
        cfg = RunConfig(
            platform=args.platform,
            device_id=args.device,
            memory_planner=planner,
            enable_chip_swimlane=4,
            enable_pmu=2,
            enable_dep_gen=True,
            save_kernels=True,
            save_kernels_dir=artifacts_dir,
        )
    else:
        cfg = RunConfig(platform=args.platform, device_id=args.device, memory_planner=planner)

    torch.manual_seed(0)
    print("generating inputs ...")
    x = torch.randint(-127, 128, (M, K), dtype=torch.int8)
    w = torch.randint(-127, 128, (GROUP, K, N), dtype=torch.int8)
    w_nz = to_nz_per_group(w)
    out = torch.zeros((M, N), dtype=torch.int32)

    print(
        f"compiling: {GROUP} groups x {N_TILES} N blocks, "
        f"K panel {L1_K}, per-group NZ weight, planner={args.planner} ..."
    )
    t0 = time.perf_counter()
    compiled = group_matmul_i8_nz.compile(x, w_nz, out, config=cfg)
    print(f"compiled in {time.perf_counter() - t0:.1f}s; artifacts: {compiled.output_dir}")

    compiled(x, w_nz, out, config=cfg)

    print("verifying bit-exact against FP64 reference ...")
    expected = grouped_matmul_ref(x, w)
    if not torch.equal(out, expected):
        diff = (out - expected).abs()
        raise SystemExit(f"FAIL: max abs diff {diff.max().item()}, mismatches {int((diff != 0).sum())}")
    print("OK  bit-exact")

    if args.profile:
        _print_profile_artifacts(compiled, artifacts_dir)
        print(
            "\nprofile run complete. Send the PMU summary and perf_hints.log above "
            "(plus any compile buffer-limit warnings)."
        )
        return

    print(f"benchmark: {args.warmup} warmup + {args.rounds} measured launches ...")
    stats = benchmark(
        compiled,
        [x, w_nz, out],
        rounds=args.rounds,
        warmup=args.warmup,
        config=cfg,
    )

    flops = 2.0 * M * K * N
    median_us = stats.device_us_median
    tflops = flops / (median_us * 1e-6) / 1e12 if median_us > 0 else 0.0
    print(
        f"device_us: median={median_us:.2f} min={stats.device_us_min:.2f} "
        f"mean={stats.device_us_mean:.2f} stdev={stats.device_us_stdev:.2f} "
        f"max={stats.device_us_max:.2f}"
    )
    print(f"device TFLOPS at median: {tflops:.1f}  ({flops / 1e9:.1f} GFLOP total)")
    print(f"raw device_us samples: {[round(v, 2) for v in stats.samples]}")
    if stats.all_zero_device:
        print("note: raw device timings are all zero — expected on simulator, not on a5")


if __name__ == "__main__":
    main()
